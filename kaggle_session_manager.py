import os
import sys
import logging
import kagglehub
from kagglehub.exceptions import KaggleApiHTTPError, UnauthenticatedError
import pandas as pd

log = logging.getLogger(__name__)

class KaggleSessionManager:
    def __init__(self, is_colab: bool) -> None:
        self.is_colab = is_colab
        self._is_authenticated = False

    def _handle_login(self, username: str, password: str) -> bool:
        """ Handle the Kaggle login.
        @brief This function will handle the Kaggle login by setting the Kaggle credentials.
        @param username The Kaggle username.
        @param password The Kaggle password.
        @return True if the login was successful, False otherwise.
        """
        kagglehub.config.set_kaggle_credentials(username, password)
        try:
            user = kagglehub.auth.whoami()
            log.info("Welcome " + user["username"])
            return True
        except UnauthenticatedError:
            log.error("Could not log in to Kaggle. Please try again.")
            return False

    def _authenticate_to_kaggle_via_gui(self) -> None:
        """ Authenticate to Kaggle via GUI.
        @brief This function will authenticate to Kaggle via the GUI.
        @return None
        """
        from PyQt5.QtWidgets import QApplication
        from UI import kaggle_login as kl

        app = QApplication(sys.argv)
        login_window = kl.KaggleLogin()
        login_window.login_signal.connect(self._handle_login)
        app.exec_()

    def _kaggle_login(self) -> None:
        """ Log in to Kaggle.
        @brief This function will log in to Kaggle using the Kaggle API.
        @return None
        """
        try:
            credentials = kagglehub.config.get_kaggle_credentials()
        except Exception as e:
            credentials = None
            if self.is_colab:
                log.critical("Could not get the Kaggle credentials. Please make sure the Kaggle API keys have been set in the environment variables KAGGLE_USERNAME and KAGGLE_KEY.")
                log.critical(e)
                sys.exit(1)

        if not credentials:
            log.info("This program uses the Kaggle API to download the data. You will need to have a Kaggle account and API credentials.")
            log.info("Learn how to obtain your Kaggle API credentials by going to https://www.kaggle.com/docs/api#authentication")
            log.info("Please log in to Kaggle.")
            self._authenticate_to_kaggle_via_gui()

        user = None
        while user is None:
            try:
                user = kagglehub.auth.whoami()
            except UnauthenticatedError:
                kagglehub.config.clear_kaggle_credentials()
                log.info("The Kaggle credentials seem to be invalid. Please log in to Kaggle using your Kaggle API credentials.")
                self._authenticate_to_kaggle_via_gui()

        log.info("Logged in to Kaggle as " + user["username"])
        self._is_authenticated = True

    def download_data(self, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """ Download the data from Kaggle.
        @brief This function will download the data from Kaggle and load it into two pandas DataFrames.
        @param dataset The Kaggle dataset identifier.
        @param cache_folder The folder to cache the downloaded data.
        @return A tuple containing two pandas DataFrames with the English and Japanese news, respectively.
        """
        cache_folder = kagglehub.config.get_cache_folder()
        data_path = os.path.join(cache_folder, "datasets", dataset, "versions")
        if not os.path.exists(data_path):
            if not self._is_authenticated:
                self._kaggle_login()
            log.info("Downloading data...")
            try:
                data_path = kagglehub.dataset_download(dataset)
            except Exception as e:
                log.critical("Could not download the data. Please make sure the data is available and you have the necessary permissions to download it.")
                log.critical(e)
                sys.exit(1)
        else:
            log.debug("Data already downloaded. Using cached data.")
            subfolders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f)) and f.isdigit()]
            if subfolders:
                latest_version = max(subfolders, key=int)
                data_path = os.path.join(data_path, latest_version)
            else:
                log.critical("No version subfolders found in the data path.")
                raise FileNotFoundError("No version subfolders found in the data path.")

        log.info("Loading data...")
        log.debug("The data has been located at: " + data_path)
        english_news = pd.read_csv(os.path.join(data_path, "english_news.csv"), sep="\t", header=0, dtype={
            "source": str,
            "date": str,
            "title": str,
            "author": str,
            "text": str
        }, low_memory=False)
        japanese_news = pd.read_csv(os.path.join(data_path, "japanese_news.csv"), sep="\t", header=0, dtype={
            "source": str,
            "date": str,
            "title": str,
            "author": str,
            "text": str
        }, low_memory=False)

        english_news["id"] = english_news.index
        japanese_news["id"] = japanese_news.index
        log.info("Data loaded.")
        return english_news, japanese_news

    def download_model(self, model: str) -> str:
        """ Download the model from Kaggle.
        @brief This function will download the model from Kaggle.
        @param model The Kaggle model identifier.
        @param cache_folder The folder to cache the downloaded model.
        @return The path to the downloaded model.
        """
        cache_folder = kagglehub.config.get_cache_folder()
        model_path = os.path.join(cache_folder, "models", model)
        if not os.path.exists(model_path):
            if not self._is_authenticated:
                self._kaggle_login()
            log.info("Downloading model weights...")
            try:
                model_path = kagglehub.model_download(model)
            except KaggleApiHTTPError as e:
                log.critical("Could not download the model weights. Please make sure the model is available and you have the necessary permissions to download it.")
                log.critical("Hint: You are required to accept a separate license agreement via Kaggle to download this model. Please go to https://www.kaggle.com/models/google/gemma-2-2b-jpn-it/ and accept the license with your account")
                log.critical(e)
                sys.exit(1)
            except Exception as e:
                log.critical("Could not download the model weights due to an unexpected error.")
                log.critical(e)
                sys.exit(1)
        else:
            log.debug("Model weights already downloaded. Using cached model weights.")
        return model_path