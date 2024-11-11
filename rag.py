"""
    @file rag.py
    @brief This file contains the implementation of the RAG model using Gemma 2 2b JPN IT.
    @author Sotarriva Alvarez Isai Roberto
    @date 21/10/2021
	@version 1.0
    @email sotarriva.i.aa@titech.ac.jp
"""
# -----------------------------------------------------------------------------IMPORTS-----------------------------------------------------------------------------#
import locale
import ast
import os
import sys
import subprocess
import contextlib
import logging
from enum import Enum
from UI import kaggle_login as kl
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from tqdm import tqdm
log = logging.getLogger(__name__) # Set up logging
#Try to import the necessary libraries. If they are not found, suggest the user how to install them.
try:
    import torch
except ImportError:
    log.critical("PyTorch not found.You might try installing it via 'pip install torch'.")
    sys.exit(1)
try:
    import kagglehub
    from kagglehub.exceptions import KaggleApiHTTPError, UnauthenticatedError
except ImportError:
    log.critical("Kagglehub not found. You might try installing it via 'pip install kagglehub'.")
    sys.exit(1)
try:
    import pandas as pd
except ImportError:
    log.critical("Pandas not found. You might try installing it via 'pip install pandas'.")
    sys.exit(1)

#check if is running in google colab if so, set the Kaggle API keys from the environment variables
IS_COLAB = False
try:
    from google.colab import userdata
    IS_COLAB = True
except ImportError:
    IS_COLAB = False
    log.info("Colab enviroment not detected")
except AttributeError:
    IS_COLAB = True
    log.info("Colab found, but the secrets could not be accessed. Be sure the user name is stored as COLAB_SECRET_USERNAME and the password as COLAB_SECRET_PASSWORD")
try:#PyQt5 will only be used outside of Google Colab. However, it is necessary to import it here to avoid errors.
    from PyQt5.QtWidgets import QApplication
except ImportError:
    log.critical("PyQt5 not found. You might try installing it via 'pip install PyQt5'.")
    sys.exit(1)
if not IS_COLAB:
    gemma_path = os.path.join(os.getcwd(), "gemma_pytorch")
else:
    gemma_path = "/content/DH/gemma_pytorch"
if not os.path.exists(gemma_path):
    log.critical("Gemma not found.Try running git submodule init and git submodule update to get the Gemma repository.")
    log.critical(" If you are still finding troubles after trying the previous solution,you might try installing by cloning the repository from github: git clone https://github.com/google/gemma_pytorch.git on the working directory.")
try:
    sys.path.append(gemma_path)
    from gemma.config import GemmaConfig, get_model_config
    from gemma.model import GemmaForCausalLM
    from gemma.tokenizer import Tokenizer
except ImportError:
    log.critical("Error importing Gemma. The gemmma_pytorch repository was found but the it is not installed yet.")
    log.critical("You might try installing it by cd into the gemma_pytorch folder and running 'pip install -e .'.")
    sys.exit(1)
#check if the machine has cuda available, if not set the machine type to cpu
MACHINE_TYPE: str = "cpu"
if torch.cuda.is_available():
    log.info("CUDA available. Using GPU.")
    MACHINE_TYPE = "cuda"
else:
    log.warning("Using CPU, this might lead to longer response times.")
# -----------------------------------------------------------------------------BODY-----------------------------------------------------------------------------#

def _handle_login(username: str, password: str) -> bool:
    """ Handle the Kaggle login.
    @brief This function will handle the Kaggle login by setting the Kaggle credentials.
    @param username The Kaggle username.
    @param password The Kaggle password.
    @return True if the login was successful, False otherwise.
    """
    kagglehub.config.set_kaggle_credentials(username, password)
    try:
        user= kagglehub.auth.whoami()
        log.info("Welcome " + user["username"])
        return True
    except UnauthenticatedError:
        log.error("Could not log in to Kaggle. Please try again.")
        return False

def _authenticateToKaggleViaGUI():
    """ Authenticate to Kaggle via GUI.
    @brief This function will authenticate to Kaggle via the GUI.
    @return None
    """
    #Try to get the Kaggle credentials from the user.
    app = QApplication(sys.argv)
    login_window = kl.KaggleLogin()
    login_window.login_signal.connect(_handle_login)
    app.exec_()

class VectorEncoder:
    """ Interface for dealing with the vector encoder.
    @brief This class uses a pandas DataFrame to store the encodings of the data. If the encodings are not found, they will be created.
    """
    _data: pd.DataFrame = None
    _encoder: SentenceTransformer = None
    _quickSearchDict: dict[torch.Tensor,[int, int]] = None
    _corpus_encodings: torch.Tensor = None
    def __init__(self) -> None:
        """ Initialize the vector encoder.
        @brief This function will initialize the vector encoder.
        """
        self._encoder = SentenceTransformer("paraphrase-xlm-r-multilingual-v1")

    def encode(self, data: pd.DataFrame) -> dict[torch.Tensor,[int, int]]:
        """ Encode the data.
        @brief This function will encode the data using the SentenceTransformer.
        @param data The data to encode.
        @return The encoded data.
        """
        self._data = data

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
                paragraphs = text.split("\n")
                text_vectors.append(self._encoder.encode(paragraphs, show_progress_bar=False, batch_size=batch_size, device=MACHINE_TYPE, convert_to_tensor=True))
            for i, row in batch_data.iterrows():
                for j, encoded_paragraph in enumerate(text_vectors[i-start_idx]):
                    tuple_encoded_paragraph = tuple(encoded_paragraph.tolist())
                    self._quickSearchDict[tuple_encoded_paragraph] = [i, j]
            if start_idx > 100:
                break
        self._corpus_encodings = torch.stack([torch.tensor(key) for key in self._quickSearchDict.keys()]).to(MACHINE_TYPE)
        log.info("Data encoded.")
        return self._quickSearchDict


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
            self._corpus_encodings = torch.stack([torch.tensor(key) for key in self._quickSearchDict.keys()]).to(MACHINE_TYPE)
            self._data = data
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
        return text.split("\n")

    def find_similar(self, keywork: str, n=20) -> pd.DataFrame:
        """ Find paragraphs most similar to the keyword.
        @brief This function will find the n articles with a paragraph most similar to the keyword.
        @param query The keyword to search for.
        @return a tuple containing the source, date, title, author, paragraph, and similarity score of the n most similar articles. 
        @note: if the keyword is found on the date, title, author or source, the paragraph field returned will be the first paragraph of the article.
        """
        sources, dates, titles, authors, texts = [], [], [], [], []
        log.debug("Finding similar articles...")
        query_vector = self._encoder.encode(keywork, show_progress_bar=False,convert_to_tensor=True).to(MACHINE_TYPE)
        similarity_scores = util.pytorch_cos_sim(query_vector, self._corpus_encodings)[0]
        scores, indices = torch.topk(similarity_scores, n)
        for i, score in zip(indices, scores):
            key_tuple = tuple(self._corpus_encodings[i].tolist())
            article_id, paragraph_id = self._quickSearchDict[key_tuple]
            sources.append(self._data.at[article_id, "source"])
            dates.append(self._data.at[article_id, "date"])
            titles.append(self._data.at[article_id, "title"])
            authors.append(self._data.at[article_id, "author"])
            texts.append(self._data.at[article_id, "text"].split("\n")[paragraph_id])
        return pd.DataFrame({
            "source": sources,
            "date": dates,
            "title": titles,
            "author": authors,
            "paragraph": texts,
            "similarity": scores
        })

    def search(self, keywords: list[str], n=20) -> pd.DataFrame:
        """ Search for articles simultaneously satisfying all the keywords.
        @brief This function will search for the n articles that simultaneously satisfy all the keywords
        @param keywords The keywords to search for.
        @return The articles that satisfy all the keywords.
        """
        #the search will be done by calling the find_similar function for each keyword and then calculating the combined query score defined as the mean of the similarity scores.
        log.debug("Searching for articles...")
        self._data["query_score"] = None
        for keyword in keywords:
            similar_articles = self.find_similar(keyword)
            for i, row in similar_articles.iterrows():
                if self._data.at[i, "query_score"] is None:
                    self._data.at[i, "query_score"] = row["similarity"]
                else:
                    self._data.at[i, "query_score"] += row["similarity"]
        self._data["query_score"] /= len(keywords)

class USERTYPES(Enum):
    """ Enum for the user types. """
    CHATBOT = 1
    USER = 2

class Chat:
    """ interface for dealing with the chatbot """
    _model: torch.nn.Module = None
    _chatHistory :list[USERTYPES, str] = []
    _instructions: str = ""

    def __init__(self, model: torch.nn.Module, instructions: str = "") -> None:
        """ Initialize the chat.
        @brief This function will initialize the chat.
        @param model The model to use for the chat.
        """
        self._model = model
        self._instructions = instructions

    def write(self, message: str, user: USERTYPES = USERTYPES.USER) -> None:
        """ Write a message.
        @brief This function will write a message to the chat history.
        @param message The message to write.
        @return None
        """
        self._chatHistory.append([user, message])

    def read(self,int) -> list[USERTYPES, str]:
        """ Read the nth message on the chat history.
        @brief This function will read the nth message on the chat history, negative numbers will read from the end.
        @param n The index of the message to read.
        @return The nth message on the chat history.
        """
        return self._chatHistory[n]
    
    def __len__(self) -> int:
        """ Get the length of the chat history.
        @brief This function will return the length of the chat history.
        @return The length of the chat history.
        """
        return len(self._chatHistory)

    def query(self, query: str) -> str:
        """ Query the chatbot.
        @brief This function will write the query to the chat history and then query the chatbot using the whole chat history.
        @param query The query to ask the chatbot.
        @return The response from the chatbot.
        @note: The chatbot response will also be written to the chat history.
        """
        self.write(query, USERTYPES.USER)
        
        #templates for the query
        USER_TEMPLATE = "<start_of_turn>user\n{prompt}<end_of_turn><eos>\n"
        CHATBOT_TEMPLATE = "<start_of_turn>chatbot\n{prompt}<end_of_turn><eos>\n"
        #get the chat history and format it
        str_query = self._instructions #start with the instructions for the chatbot
        for user, message in self._chatHistory:#add the chat history to the query
            if user == USERTYPES.USER:
                str_query += USER_TEMPLATE.format(prompt=message)
            else:
                str_query += CHATBOT_TEMPLATE.format(prompt=message)
        str_query += "<start_of_turn>model\n"
        prompt = (str_query)
        response = self._model.generate(USER_TEMPLATE.format(prompt=prompt),device=MACHINE_TYPE, output_len=128,)
        self.write(response, USERTYPES.CHATBOT)
        return response

    def clear(self) -> None:
        """ Clear the chat history.
        @brief This function will clear the chat history.
        @return None
        """
        self._chatHistory = []

class RAG:
    """ Implements the RAG model using Gemma 2 2b JPN IT"""
    _vector_encoder: VectorEncoder = None
    _model: torch.nn.Module = None
    def _kaggle_login(self) -> None:
        """ Log in to Kaggle.
        @brief This function will log in to Kaggle using the Kaggle API.
        @return None
        """
        #Check if kaggle has been logged in. If not, ask the user to log in.
        #If not running in Google Colab, check if the Kaggle credentials are stored. If not, ask the user to log in.
        try:
            credentials = kagglehub.config.get_kaggle_credentials()
        except Exception as e:
            credentials = None
            if IS_COLAB:
                log.critical("Could not get the Kaggle credentials. Please make sure the Kaggle API keys have been set in the environment variables KAGGLE_USERNAME and KAGGLE_KEY.")
                log.critical(e)
                sys.exit(1)
        if not credentials:
            #This will happen the first time the program is run.
            log.info("This program uses the Kaggle API to download the data. You will need to have a Kaggle account and API credentials.")
            log.info("Learn how to obtain your Kaggle API credentials by going to https://www.kaggle.com/docs/api#authentication")
            log.info("Please log in to Kaggle.")
            _authenticateToKaggleViaGUI()
        #The program will try to log in to Kaggle using the stored credentials.
        #In the case they have been become invalid, the user will be asked to log in again with the new credentials.
        user = None
        while user is None: 
            #While the credentials are invalid, the user will not be able to proceed.
            try:
                user = kagglehub.auth.whoami()
            except UnauthenticatedError:
                kagglehub.config.clear_kaggle_credentials()
                log.info("The Kaggle credentials seem to be invalid. Please log in to Kaggle using your Kaggle API credentials.")
                _authenticateToKaggleViaGUI()
        #If the user is logged in, the program will continue.
        log.info("Logged in to Kaggle as " + user["username"])

    def _load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """ Load the data.
        @brief This function will download the data from Kaggle and load it into two pandas DataFrames.
        @return A tuple containing two pandas DataFrames with the English and Japanese news, respectively.
        """
        #Download the data from Kaggle.
        data_path = os.path.join(kagglehub.config.DEFAULT_CACHE_FOLDER,"datasets","vyhuholl","japanese-newspapers-20052021","versions")
        if not os.path.exists(data_path):
            log.info("Downloading data...")
            try:
                data_path = kagglehub.dataset_download("vyhuholl/japanese-newspapers-20052021")
            except Exception as e:
                log.critical("Could not download the data. Please make sure the data is available and you have the necessary permissions to download it.")
                log.critical(e)
                sys.exit(1)
        else:
            log.debug("Data already downloaded. Using cached data.")
            #append the version number to the path
            subfolders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f)) and f.isdigit()]
            if subfolders:
                latest_version = max(subfolders, key=int)
                data_path = os.path.join(data_path, latest_version)
            else:
                log.critical("No version subfolders found in the data path.")
                raise FileNotFoundError("No version subfolders found in the data path.")
        log.info("Loading data...")
        log.debug("The data has been located at: " + data_path)
        english_news = pd.read_csv(os.path.join(data_path, "english_news.csv"), sep="	", header=(0), dtype={
            "source": str,
            "date": str,
            "title": str,
            "author": str,
            "text": str
        }, low_memory=False)
        japanese_news = pd.read_csv(os.path.join(data_path, "japanese_news.csv"), sep="	", header=(0), dtype={
            "source": str,
            "date": str,
            "title": str,
            "author": str,
            "text": str
        }, low_memory=False)
        #add and id column to the dataframes following the dataIndex. This will be used to identify articles on an unique way (The title might not be unique)
        english_news["id"] = english_news.index
        japanese_news["id"] = japanese_news.index
        log.info("Data loaded.")
        return english_news, japanese_news

    def _load_model(self) -> torch.nn.Module:
        """ Load the model. 
        @brief This function will download the Gemma 2 2b JPN IT model from Kaggle and load it into a PyTorch model.
        @return A PyTorch model.
        """
        # check that inmutable dict and sentencepiece are installed
        try:
            import immutabledict
            import sentencepiece
        except ImportError:
            log.critical("Immutabledict and sentencepiece not found. You might try installing them via 'pip install immutabledict sentencepiece'.")
        VARIANT : str = "2b-v2"
        if os.path.exists(os.path.join(os.getcwd(), "modelWeights")):
            model_path = os.path.join(os.getcwd(), "modelWeights")
            log.info("Using cached model weights.")
        else:
            log.info("Downloading model weights...")
            try:
                model_path = kagglehub.model_download("google/gemma-2-2b-jpn-it/pyTorch/gemma-2-2b-jpn-it")
            except KaggleApiHTTPError as e:
                log.critical("Could not download the model weights. Please make sure the model is available and you have the necessary permissions to download it.")
                log.critical("Hint: You are required to accept a separate license agreement via Kaggle to download this model. Please go to https://www.kaggle.com/models/google/gemma-2-2b-jpn-it/ and accept the license with your accout")
                log.critical(e)
                sys.exit(1)
            except Exception as e:
                log.critical("Could not download the model weights. Due to an unexpected error.")
                log.critical(e)
                sys.exit(1)
        log.debug("Weights found at: " + model_path)
        weights_file = os.path.join(model_path, "model.ckpt")
        if not os.path.exists(weights_file):
            log.critical("The model weights were not found.")
            raise FileNotFoundError("The model weights were not found.")
        @contextlib.contextmanager
        def _set_default_tensor_type(dtype: torch.dtype):
            """Sets the default torch dtype to the given dtype."""
            torch.set_default_dtype(dtype)
            yield
            torch.set_default_dtype(torch.float)

        model_config = get_model_config(VARIANT)
        model_config.tokenizer = os.path.join(model_path, "tokenizer.model")
        if not os.path.exists(model_config.tokenizer):
            log.critical("The tokenizer model was not found.")
            raise FileNotFoundError("The tokenizer model was not found.")
        device = torch.device(MACHINE_TYPE)
        with _set_default_tensor_type(model_config.get_dtype()):
            model = GemmaForCausalLM(model_config)
            model.load_weights(weights_file)
            model = model.to(device).eval()
        return model
    
    def __init__(self) -> None:
        """ Initialize the RAG model. """
        log.debug("Initializing RAG model...")
        locale.getpreferredencoding = lambda: 'UTF-8' # Set up locale
        self._kaggle_login() # Log in to Kaggle
        self._vector_encoder = VectorEncoder()
        #look for any stored vector encodings of the data. If not found, create them.
        english_news, japanese_news = None, None
        try:
            english_news, japanese_news = self._load_data() # Load the data
        except FileNotFoundError as e:
            log.critical("Could not load the data. Please make sure the data is available.")
            log.critical(e)
            sys.exit(1)
            log.debug("Encoded data not found. An encoding will be created.")
            log.info("Creating vector encodings, this might take a while...")
        #concatenate the dataframes to encode them together
        data = english_news #pd.concat([english_news, japanese_news], ignore_index=True)
        if os.path.exists(os.path.join(os.getcwd(), "encoded_data.csv")):
            log.debug("Encoded data found. Loading...")
            self._vector_encoder.load(os.path.join(os.getcwd(), "encoded_data.csv"), data)
        else:
            self._vector_encoder.encode(data)
            self._vector_encoder.save(os.path.join(os.getcwd(), "encoded_data.csv"))
        self._model = self._load_model() # Load the model
        log.debug("RAG model initialized.")
    
    def query(self, query: str) -> str:
        """ Query the RAG model.
        @brief This function will query the RAG model with the given query and return the response.
        @param query The query to ask the RAG model.
        @return The response from the RAG model.
        """
        #use the LLM to generate a comma separated list of keywords to be used on the search engine to provide context for the chatbot to answer the query
        keywordsChat = Chat(self._model,"")
        keywordsChat.clear()
        log.debug("Querying the chatbot for keywords...")
        keywords= ""#keywordsChat.query(f"Given the query: '{query}', give a comma separated list of keywords that can be used to search for news articles to help answer the query, keywords can be in Japanese or English. Example:'Japan, 開会式, オリンピック'")
        log.debug("Keywords found: " + str(keywords))
        keywords = keywords.split(",")
        #search for similar articles which can be used as context for the chatbot
        similar_articles = self._vector_encoder.find_similar(keywords)
        #get the top 5 similar articles
        similar_articles = similar_articles.head(5)
        #get the source, data, title, author and text of the articles
        sources = similar_articles["source"].tolist()
        dates = similar_articles["date"].tolist()
        titles = similar_articles["title"].tolist()
        authors = similar_articles["author"].tolist()
        texts = similar_articles["paragraph"].tolist()
        #Prepare the instructions for the chatbot
        instructions = "Answer the querry from the user using the following news articles as context:\n=============================================================="
        for i in range(len(sources)):
            instructions += f"Source: {sources[i]}\nDate: {dates[i]}\nTitle: {titles[i]}\nAuthor: {authors[i]}\nParagraph: {texts[i]}\n\n"
        instructions += "==============================================================\n\n"
        instructions += "Respond to the user on the same language he/she is addressing you regardless of the language of the news articles. Cite your sources whenever possible."
        log.debug("Instructions: " + instructions)
        #Start a new chat with the instructions 
        chat = Chat(self._model, instructions)
        chat.clear() #Clear the chat history
        return chat.query(query)