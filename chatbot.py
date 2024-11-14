
"""
    @file chatbot.py
    @brief This file contains the chatbot interface.
    @version 1.0
    @date 14/11/2024
    @author Sotarriva Alvarez Isai Roberto
    @contact sotarriva.i.aa@titech.ac.jp
"""
import logging
from enum import Enum
import torch
log = logging.getLogger(__name__)
class USERTYPES(Enum):
    """ Enum for the user types. """
    CHATBOT = 1
    USER = 2

class Chat:
    """ interface for dealing with the chatbot """
    def __init__(self, model: torch.nn.Module, instructions: str = "", device: str = "cuda" if torch.cuda.is_available() else "cpu") -> None:
        """ Initialize the chat.
        @brief This function will initialize the chat.
        @param model The model to use for the chat.
        """
        self._model = model
        self._instructions:str = instructions
        self._model.to(device)
        self._device = device
        self._chatHistory :list[USERTYPES, str] = []

    def write(self, message: str, user: USERTYPES = USERTYPES.USER) -> None:
        """ Write a message.
        @brief This function will write a message to the chat history.
        @param message The message to write.
        @return None
        """
        self._chatHistory.append([user, message])

    def read(self, n: int) -> list[USERTYPES, str]:
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

    def query(self, query: str,desired_lenght=128) -> str:
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
        str_query = self._instructions + "\n" #start with the instructions for the chatbot
        for user, message in self._chatHistory:#add the chat history to the query
            if user == USERTYPES.USER:
                str_query += USER_TEMPLATE.format(prompt=message)
            else:
                str_query += CHATBOT_TEMPLATE.format(prompt=message)
        str_query += "<start_of_turn>model\n"
        prompt = (str_query)
        log.debug("Prompt: " + prompt)
        response = self._model.generate(USER_TEMPLATE.format(prompt=prompt),device=self._device, output_len=desired_lenght)
        self.write(response, USERTYPES.CHATBOT)
        return response

    def clear(self) -> None:
        """ Clear the chat history.
        @brief This function will clear the chat history.
        @return None
        """
        self._chatHistory = []
        
    def setInstructions(self, instructions: str) -> None:
        """ Set the instructions.
        @brief This function will set the instructions for the chatbot.
        @param instructions The instructions for the chatbot.
        @return None
        """
        self._instructions = instructions