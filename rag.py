"""
    @file rag.py
    @brief This file contains the implementation of the RAG model using Gemma 2 2b JPN IT.
    @version 2.0
    @date 14/11/2024
    @authot Sotarriva Alvarez Isai Roberto
    @contact sotarriva.i.aa@titech.ac.jp
"""
# -----------------------------------------------------------------------------IMPORTS-----------------------------------------------------------------------------#
import os
import sys
import logging
import contextlib
import torch
import pandas as pd
from kaggle_session_manager import KaggleSessionManager
import sentence_vectors
from chatbot import Chat
log = logging.getLogger(__name__) # Set up logging

class RAG:
    def __init__(self, is_colab: bool = False) -> None:
        self.is_colab = is_colab
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.kaggle_manager = KaggleSessionManager(is_colab)
        self._vector_encoder = sentence_vectors.VectorEncoder(device=self._device)
        self._vector_encoder.load_data(self.kaggle_manager)
        self._model = self._load_model()
        self._rag_chat = Chat(self._model, "", device=self._device)
        self._finder_chat = Chat(self._model, "Your role is to be a search assistant. Provide a comma separated list of topics or keywords in Japanese related to the query. Example: For the query 'When was the opening ceremony of the Tokyo Olympics?' you could answer '東京、オリンピック、開会式, 2020年、開催、日本', or for the query 'Is abortion legal in Japan?' you could answer '中絶、合法、日本'. Return at least one keyword.\n Do not provide an explanation or details on the topics, just the keywords or topics since they will be automatically submitted to a search engine.", device=self._device)

    def _load_model(self) -> torch.nn.Module:
        """ Load the model from Kaggle.
        @brief This function will load the model from Kaggle.
        @return The loaded model.
        """
        if os.path.exists(os.path.join(os.getcwd(), "modelWeights")):
            model_path = os.path.join(os.getcwd(), "modelWeights")
            log.info("Using cached model weights.")
        else:
            log.info("Downloading model weights...")
            model_path = self.kaggle_manager.download_model(os.path.join("google","gemma-2-2b-jpn-it","pyTorch","gemma-2-2b-jpn-it"))
            log.info("Model weights downloaded.")
        sys.path.append(model_path)
        from gemma.config import GemmaConfig, get_model_config
        from gemma.model import GemmaForCausalLM
        from gemma.tokenizer import Tokenizer
        log.debug("Weights found at: " + model_path)
        weights_file = None
        for root, dirs, files in os.walk(model_path):
            if "model.ckpt" in files:
                weights_file = os.path.join(root, "model.ckpt")
                break
        if not os.path.exists(weights_file):
            log.critical("The model weights were not found.")
            raise FileNotFoundError("The model weights were not found.")
        @contextlib.contextmanager
        def _set_default_tensor_type(dtype: torch.dtype):
            """Sets the default torch dtype to the given dtype."""
            torch.set_default_dtype(dtype)
            yield
            torch.set_default_dtype(torch.float)
        model_config = get_model_config("2b-v2")
        model_config.tokenizer = os.path.join(os.path.dirname(weights_file), "tokenizer.model")
        if not os.path.exists(model_config.tokenizer):
            log.critical("The tokenizer model was not found.")
            raise FileNotFoundError("The tokenizer model was not found.")
        with _set_default_tensor_type(model_config.get_dtype()):
            model = GemmaForCausalLM(model_config)
            model.load_weights(weights_file)
            model = model.to(self._device).eval()
        return model

    def find_similar_articles(self, query: str, n: int = 20) -> pd.DataFrame:
        """ Find similar articles based on the query.
        @brief This function will find similar articles based on the query.
        @param query The query to search for.
        @param n The number of similar articles to return.
        @return A DataFrame containing the similar articles.
        """
        log.debug("Querying the chatbot for keywords...")
        keywords = self._finder_chat.query("I want to know about:'" + query + "'. What topics or keywords should I search for in the database?", desired_lenght=32)#A short desired_lenght provides more consize keywords and is faster to process
        log.debug("Keywords response: " + str(keywords))

        if not keywords:
            log.warning("No keywords found. Using the query as keywords. This might lead less accurate results.")
            keywords = [query]
        else:
            #find the location of \n<end_of_turn> in the keywords and remove the instructions from the chatbot
            keywords = keywords.split("\n<end_of_turn>")[0]
            log.debug("Keywords found: " + keywords)
            #split the keywords into a list. The list is comma separated, however the commas cna be western (,) or japanese (、)
            keywords = keywords.replace("、", ",").split(",")

        #search for similar articles which can be used as context for the chatbot
        similar_articles = self._vector_encoder.find_similar(keywords, n=n)
        return similar_articles

    def generate_response(self, query: str) -> tuple[str, pd.DataFrame]:
        """ Generate a response to the query using the RAG model.
        @brief This function will generate a response to the query using the RAG model.
        @param query The query to generate a response for.
        @return The generated response and the dataFrame with the similar articles.
        """
        similar_articles = self.find_similar_articles(query, n=10)
        UIDs = similar_articles["UID"].tolist()
        sources = similar_articles["source"].tolist()
        dates = similar_articles["date"].tolist()
        titles = similar_articles["title"].tolist()
        texts = similar_articles["paragraph"].tolist()

        #Prepare the instructions for the chatbot
        instructions = "You are an assistant for a social studies research project. You have been asked to read the following news articles and provide a response to the user's question based on the information in the articles.(The articles are the result of an automatic search based on the user's query, some articles may not be relevant to the user's question)\n\n"
        instructions += "==============================================================\n\n"
        #ignore the fields (title, date, Source ) id they are empty, "", "None" or "nan"
        for i in range(len(sources)):
            instructions += f"UID: {UIDs[i]}\n"
            if titles[i] not in ["", "None", "nan"]:
                instructions += f"Title: {titles[i]}\n"
            if dates[i] not in ["", "None", "nan"]:
                instructions += f"Date: {dates[i]}\n"
            if sources[i] not in ["", "None", "nan"]:
                instructions += f"Source: {sources[i]}\n"
            instructions += f"Content snippet: {texts[i]}\n\n"
        instructions += "==============================================================\n\n"
        instructions += "Answer the question on the same language it is written (japanese or english).\n" 
        instructions += "It is very important to include citations\n"
        instructions += "Use [UID] to provide citations to the news articles, links in this format will automatically be transformed into hyperlinks to the original articles and transformed to look on the format desired by the user. Example, to cite the article with UID: @123/4, write [@123/4], not [UID:@123/4]. For multiple citations, place each citation in a separate pair of brackets. Example: [@123/4][@456/7] do not write them together like [@123/4, @456/7].\n"
        instructions += "Use HTML tags to format the text. Example: <b>bold text</b>, <i>italic text</i>, <ul><li>unordered list item</li></ul>, <ol><li>ordered list item</li></ol>, <a href='link'>hyperlink</a>\n"

        #Use the rag_chat since it contains all the chat history.
        self._rag_chat._instructions = instructions
        response = self._rag_chat.query(query, desired_lenght=1024) #The maximum length of the response is 8192 tokens. Longer lengths might take a long time to process.
        log.debug("Response: " + response)
        return response, similar_articles
        
    def clear(self) -> None:
        """ Clear the chat history.
        @brief This function will clear the chat history.
        @return None
        """
        self._rag_chat.clear()
        self._finder_chat.clear()