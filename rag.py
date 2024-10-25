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
import os
import sys
import subprocess
import contextlib
import logging
from enum import Enum
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

class USERTYPES(Enum):
    """ Enum for the user types. """
    CHATBOT = 1
    USER = 2

class Chat:
    """ interface for dealing with the chatbot """
    _model: torch.nn.Module = None
    _chatHistory :list[USERTYPES, str] = []

    def __init__(self, model: torch.nn.Module) -> None:
        """ Initialize the chat.
        @brief This function will initialize the chat.
        @param model The model to use for the chat.
        """
        self._model = model

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
        str_query = ""
        for user, message in self._chatHistory:
            if user == USERTYPES.USER:
                str_query += USER_TEMPLATE.format(prompt=message)
            else:
                str_query += CHATBOT_TEMPLATE.format(prompt=message)
        str_query += "<start_of_turn>model\n"
        prompt = (str_query)
        log.debug("Querying the model with: " + prompt)
        response = self._model.generate(USER_TEMPLATE.format(prompt=prompt),device=MACHINE_TYPE, output_len=128,)
        log.debug("Response from the model: " + response)
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
    _english_news: pd.DataFrame = None
    _japanese_news: pd.DataFrame = None
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
        english_news = pd.read_csv(os.path.join(data_path, "english_news.csv"), sep="	", header=(0))
        japanese_news = pd.read_csv(os.path.join(data_path, "japanese_news.csv"), sep="	", header=(0))
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
        try:
            self._english_news, self._japanese_news = self._load_data() # Load the data
        except FileNotFoundError as e:
            log.critical("Could not load the data. Please make sure the data is available.")
            log.critical(e)
            sys.exit(1)
        self._model = self._load_model() # Load the model
        log.debug("RAG model initialized.")
    
    def query(self, query: str) -> str:
        """ Query the RAG model.
        @brief This function will query the RAG model with the given query and return the response.
        @param query The query to ask the RAG model.
        @return The response from the RAG model.
        """
        chat = Chat(self._model) #Start a new chat
        return chat.query(query)