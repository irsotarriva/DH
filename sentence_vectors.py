import ast
import re
import sys
from typing import List, Dict
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import os
import numpy as np
import kaggle_session_manager
import logging
log = logging.getLogger(__name__) # Set up logging
#Try to import the necessary libraries. If they are not found, suggest the user how to install them.
try:
    import torch
except ImportError:
    log.critical("PyTorch not found.You might try installing it via 'pip install torch'.")
    sys.exit(1)
try:
    import pandas as pd
except ImportError:
    log.critical("Pandas not found. You might try installing it via 'pip install pandas'.")
    sys.exit(1)

class VectorEncoder:
    """ Interface for dealing with the vector encoder.
    @brief This class uses a pandas DataFrame to store the encodings of the data. If the encodings are not found, they will be created.
    """

    def __init__(self, device: str = "cpu") -> None:
        """ Initialize the vector encoder.
        @brief This function will initialize the vector encoder.
        """
        self._encoder = SentenceTransformer("paraphrase-xlm-r-multilingual-v1")
        self._device = device
        self._data: pd.DataFrame = None
        self._quickSearchDict: Dict[torch.Tensor, List[int]] = None
        self._corpus_encodings: torch.Tensor = None

    def encode(self, data: pd.DataFrame) -> Dict[torch.Tensor, List[int]]:
        """ Encode the data.
        @brief This function will encode the data using the SentenceTransformer.
        @param data The data to encode.
        @return The encoded data.
        """
        self._data = self._clean_data(data)

        #the dictionary will use the tensor as key and return the index of the article in the dataframe plus the paragraph index.
        log.info("Encoding data...")
        total_rows = len(self._data)
        self._quickSearchDict = {}
        batch_size = 32  # Adjust the batch size according to your GPU memory
        for start_idx in tqdm(range(0, total_rows, batch_size), desc="Encoding data"):
            end_idx = min(start_idx + batch_size, total_rows)
            batch_data = self._data.iloc[start_idx:end_idx]
            #the text will be encoded in paragraphs to help the retrieval be more accurate
            text_vectors = []
            for text in batch_data["text"].astype(str).tolist():
                paragraphs = self._split_text_into_paragraphs(text)
                text_vectors.append(self._encoder.encode(paragraphs, show_progress_bar=False, batch_size=batch_size, device=self._device, convert_to_tensor=True))
            for i in range(start_idx, end_idx):
                for j, encoded_paragraph in enumerate(text_vectors[i - start_idx]):
                    tuple_encoded_paragraph = tuple(encoded_paragraph.tolist())
                    self._quickSearchDict[tuple_encoded_paragraph] = [i, j]
            if start_idx > 1000:
                break
        self._corpus_encodings = torch.stack([torch.tensor(key) for key in self._quickSearchDict.keys()]).to(self._device)
        log.info("Data encoded.")
        return self._quickSearchDict

    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """ Clean the data.
        @brief This function will clean the data by removing known error messages from the crawled data.
        @param data The data to clean.
        @return The cleaned data.
        """
        log.info("Cleaning data...")
        #ensure that the dtypes are all strings
        data = data.astype(str)
        #remove known error messages common on the dataset
        error_messages = [
            "This could be due to a conflict with your ad-blocking or security software.",
            "Please addjapantimes.co.jpandpiano.ioto your list of allowed sites.",
            "If this does not resolve the issue or you are unable to add the domains to your allowlist, please see outthis support page. We humbly apologize for the inconvenience."
        ]
        for error_message in error_messages:
            data = data[~data["text"].str.contains(error_message)]
        #remove articles with empty text or with less than 50 characters.
        data = data[data["text"].str.len() > 50]
        log.info("Data cleaned.")
        return data
    def get(self) -> pd.DataFrame:
        """ Get the encoded data.
        @brief This function will return the encoded data.
        @return The encoded data.
        """
        return self._data

    def save(self, path: str) -> None:
        """ Save the encoded data.
        @brief This function will save the encoded data to a file.
        @param path The path to save the encoded data.
        @return None
        """
        log.info("Saving encoded data...")
        #save the quick search dictionary
        with open(path, 'w') as f:
            for key, value in self._quickSearchDict.items():
                key_str = list(key)  # Convert tensor to list
                f.write(f"{key_str}\t{value[0]}\t{value[1]}\n")
        #save the data
        log.info("Encoded data saved.")

    def load(self, path: str, data: pd.DataFrame) -> None:
        """ Load the encoded data.
        @brief This function will load the encoded data from a file.
        @param path The path to load the encoded data.
        @return None
        """
        log.info("Loading encoding...")
        #load the quick search dictionary
        try:
            self._quickSearchDict = {}
            with open(path) as f:
                for line in f:
                    key, value1, value2 = line.split("\t")
                    key_tuple = tuple(ast.literal_eval(key))
                    self._quickSearchDict[key_tuple] = [int(value1), int(value2)]
            self._corpus_encodings = torch.stack([torch.tensor(key) for key in self._quickSearchDict.keys()]).to(self._device)
            self._data = self._clean_data(data)
            #load the data
            log.info("Encoded data loaded.")
        except FileNotFoundError as e:
            log.critical("Could not load the encoded data. Please make sure the data is available.")
            log.critical(e)
            sys.exit(1)
    
    def _split_text_into_paragraphs(self, text: str) -> list[str]:
        """ Split the text into paragraphs.
        @brief This function will split the text into paragraphs.
        @param text The text to split.
        @return The paragraphs of the text.
        """
        #identify the language of the text either Japanese or English
        japanese_char_count = len(re.findall(r'[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]', text))
        japanese_char_ratio = japanese_char_count / len(text)
        min_paragraph_length = 150 #helps to avoid splitting the text into too many paragraphs.
        if japanese_char_ratio > 0.5:
            min_paragraph_length = 50
            paragraphs = re.split(r'\n', text)
        else:
            paragraphs = re.split(r'\n|\.', text)
        filtered_paragraphs = []
        buffer = ""
        for paragraph in paragraphs:
            buffer += paragraph
            if len(buffer) > min_paragraph_length:#if the paragraph is too short, it will be merged with the next paragraph
                filtered_paragraphs.append(buffer)
                buffer = ""
        if buffer:
            if not filtered_paragraphs:
                filtered_paragraphs.append(buffer)
            else:
                filtered_paragraphs[-1] += buffer
        return filtered_paragraphs

    def find_similar(self, keywork: str, n=20) -> pd.DataFrame:
        """ Find paragraphs most similar to the keyword.
        @brief This function will find the n articles with a paragraph most similar to the keyword.
        @param query The keyword to search for.
        @return a tuple containing the source, date, title, author, paragraph, and similarity score of the n most similar articles. 
        @note: if the keyword is found on the date, title, author or source, the paragraph field returned will be the first paragraph of the article.
        """
        sources, dates, titles, authors, texts, uids, full_texts = [], [], [], [], [], [], []
        query_vector = self._encoder.encode(keywork, show_progress_bar=False,convert_to_tensor=True).to(self._device)
        similarity_scores = util.pytorch_cos_sim(query_vector, self._corpus_encodings)[0]
        scores, indices = torch.topk(similarity_scores, n)
        for i in indices:
            key_tuple = tuple(self._corpus_encodings[i].tolist())
            article_id, paragraph_id = self._quickSearchDict[key_tuple]
            article = self._data.iloc[article_id]
            sources.append(article["source"])
            dates.append(article["date"])
            titles.append(article["title"])
            authors.append(article["author"])
            full_texts.append(article["text"])
            texts.append(self._split_text_into_paragraphs(article["text"])[paragraph_id])
            uids.append(f"@{article_id}/{paragraph_id}")
        return pd.DataFrame({
            "UID": uids,
            "source": sources,
            "date": dates,
            "title": titles,
            "author": authors,
            "paragraph": texts,
            "full_text": full_texts,
            "similarity": scores.cpu().numpy()
        })

    def search(self, keywords: list[str], n=20) -> pd.DataFrame:
        """ Search for articles simultaneously satisfying all the keywords.
        @brief This function will search for the n articles that simultaneously satisfy all the keywords
        @param keywords The keywords to search for.
        @return The articles that satisfy all the keywords.
        """
        #the search will be done by calling the find_similar function for each keyword and then calculating the combined query score defined as the mean of the similarity scores.
        log.debug("Searching for articles...")
        potential_articles = pd.DataFrame()
        for keyword in keywords:
            potential_articles = pd.concat([potential_articles, self.find_similar(keyword, n*20)], axis=0)
        potential_articles = potential_articles.drop_duplicates(subset="UID").reset_index(drop=True)
        potential_articles["query_score"] = 0
        #evaluate the cosine similarity of the paragraph with each keyword and calculate the query score as the sum of the similarity scores.
        paragraph_vectors = self._encoder.encode(potential_articles["paragraph"].tolist(), show_progress_bar=False, convert_to_tensor=True).to(self._device)
        for keyword in keywords:
            keyword_vector = self._encoder.encode(keyword, show_progress_bar=False, convert_to_tensor=True).to(self._device)
            similarity_scores = util.pytorch_cos_sim(keyword_vector, paragraph_vectors)
            potential_articles["query_score"] += similarity_scores.cpu().numpy()
        potential_articles = potential_articles.sort_values("query_score", ascending=False).head(n)
        return potential_articles

    def load_data(self,kaggle_manager: kaggle_session_manager.KaggleSessionManager) -> None:
        """ Load the data from Kaggle.
        @brief This function will load the data and encodings, if no econodings are found, they will be created.
        @return None
        """
        log.info("Loading data...")
        #if no encodings are found, encode the data
        dataset = os.path.join("vyhuholl","japanese-newspapers-20052021")
        english_news, japanese_news = kaggle_manager.download_data(dataset)
        self._data = pd.concat([english_news, japanese_news], axis=0).reset_index(drop=True)
        encodings_path = "encoded_data.csv"
        if os.path.exists(encodings_path):
            self.load(encodings_path, self._data)
        else:
            self.encode(self._data)
            self.save(encodings_path)