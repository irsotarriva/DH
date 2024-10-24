"""
    @file objects.py
    @brief This file contains the implementation of the classes used in the RAG model.
    @author Sotarriva Alvarez Isai Roberto
    @date 21/10/2021
    @version 1.0
    @email sotarriva.i.aa@m.titech.ac.jp
"""
# -----------------------------------------------------------------------------IMPORTS-----------------------------------------------------------------------------#
import datetime
from decorator import cache
import logging
log = logging.getLogger(__name__) # Set up logging
try:
    import pandas as pd
except ImportError:
    log.critical("Pandas not found. You might try installing it via 'pip install pandas'.")
    sys.exit(1)

# -----------------------------------------------------------------------------BODY-----------------------------------------------------------------------------#
class TextSnippet:
    """
    @class TextSnippet
    @brief This class is used to pass texts and highlights ingormation to the UI.
    @note A list of int pairs is used to represent the highlighted sections. The UI will use this information to highlight the text and show the user the important parts of the text.
    """
    text: str = ""
    highlights: list[tuple[int, int]] = []

    def __init__(self, text: str, highlights: List[Tuple[int, int]]):
        """
        @fn __init__
        @brief Constructor of the class.
        @param text: The text to be shown in the UI.
        @param highlights: A list of int pairs representing the highlighted sections of the text.
        """
        self.text = text
        self.highlights = highlights
    def show(characterLimit: int = 100):
        """
        @fn show
        @brief This function returns the text to be shown in the UI.
        @param characterLimit: The maximum number of characters to be shown in the UI.
        @return Preformated string of text to be shown in the UI. (The highlighted parts are in bold)
        @details If the text is too long, the text is cut and "..." is added at the end. The highlights are prioritized on the cut text. If there is still some space left over after covering all the highlights the text after the last highlight is added and if there is still space left the text before the first highlight is added. The program will try to show as much text as possible while keeping the highlights visible, while cutting the text only at the beginning or end of a word.
        """
        # If the text is too long, the text is cut and "..." is added at the end. The highlights are prioritized on the cut text.
        if len(self.text) > characterLimit:
            effectiveCharacterLimit:int = characterLimit - 3  # 3 is the length of "..."
            base_text:str = self.text
            formated_text:str = ""
            offset:int = 0
            invisibleCharacters:int = 0
            lastCharPositionOnOriginalText:int = 0
            for start, end in self.highlights:
                if offset > 0:
                    formated_text += "..."
                    offset += 3
                start += offset
                end += offset
                #if the highlight is longer than the leftover space availible, the highlight is cut. Otherwise, the highlight is added fully.
                spaceStillLeft:int = effectiveCharacterLimit-len(formated_text)+invisibleCharacters
                if end-start > spaceStillLeft-3:
                    textToBeAdded:str = base_text[start:start+spaceStillLeft]
                    #end at the end of a word
                    if textToBeAdded[-1] != " ":
                        nextWordPosition:int = textToBeAdded.rfind(" ")
                        textToBeAdded = textToBeAdded[:nextWordPosition]
                    formated_text += "<b>" + textToBeAdded + "</b>..."
                    offset += 10  # length of "<b></b>..." is 10
                    invisibleCharacters += 7 #the tags "<b>" and "</b>"" are invisible
                    lastCharPositionOnOriginalText = start+spaceStillLeft
                    #exit the for loop since there is no more space left
                    break
                else:
                    formated_text += "<b>" + base_text[start:end] + "</b>"
                offset += 7  # length of "<b></b>" is 7
                invisibleCharacters += 7 #the tags "<b>" and "</b>"" are invisible
                lastCharPositionOnOriginalText = end
            #if there is still some space left over after covering all the highlights. The last ... is removed and the text after the last highlight is added until the limit is reached or we reach the end of the text.
            spaceStillLeft:int = effectiveCharacterLimit-len(formated_text)+invisibleCharacters
            if len(formated_text)-lastCharPositionOnOriginalText < spaceStillLeft:
                #enough space left to show the rest of the text
                formated_text += base_text[lastCharPositionOnOriginalText:]
                #With the space left, we can now try to show text before the first highlight, until either the limit is reached or we reach the start of the text.
                firstHighlightPosition:int = self.highlights[0][0]
                spaceStillLeft:int = effectiveCharacterLimit-len(formated_text)+invisibleCharacters
                if firstHighlightPosition < spaceStillLeft:
                    formated_text = base_text[:firstHighlightPosition] + formated_text
                else:
                    #if the space left is more than 1 word+3 characters, we show ... and as many characters as possible before the first highlight.
                    prevWordPosition:int = base_text.rfind(" ", 0, firstHighlightPosition)
                    if firstHighlightPosition-prevWordPosition < spaceStillLeft-3:
                        #enough space left to show some of the text before the first highlight. The text is cut and "..." is added at the beginning.
                        textToBeAdded:str = base_text[prevWordPosition:firstHighlightPosition]
                        #start from the begining of a word
                        if textToBeAdded[0] != " ":
                            prevWordPosition = textToBeAdded.find(" ")
                            textToBeAdded = textToBeAdded[prevWordPosition:]
                        formated_text = "..." + textToBeAdded + formated_text
                    else:
                        #not enough space left to show something decent so we just add "..." at the beginning if the first highlight is not at the start of the text.
                        if firstHighlightPosition > 0:
                            formated_text = "..." + formated_text
            else:
                #not enough space left to show the rest of the text. The text is cut and "..." is added at the end.
                testToBeAdded:str = base_text[lastCharPositionOnOriginalText:lastCharPositionOnOriginalText+spaceStillLeft-3]
                #end at the end of a word
                if testToBeAdded[-1] != " ":
                    nextWordPosition:int = testToBeAdded.rfind(" ")
                    testToBeAdded = testToBeAdded[:nextWordPosition]
                formated_text += testToBeAdded + "..."
        else:
            formatted_text = self.text
            offset = 0
            for start, end in self.highlights:
                start += offset
                end += offset
                formatted_text = formatted_text[:start] + "<b>" + formatted_text[start:end] + "</b>" + formatted_text[end:]
                offset += 7  # length of "<b></b>" is 7
        return formatted_text

class Article:
    """
    @class Article
    @brief This class is used to define a new article.
    @note An article has a title, a text, a list of authors, and a list of publishers.
    """
    publisher: str| None = ""
    title: str| None = ""
    date: datetime.datetime| None = None
    authors: list[str] = [] #Names of the authors that have written the article
    content: TextSnippet = None
    def __init__(self, publisher: str| None, title: str| None, date: datetime.datetime| None, authors: List[str], content: TextSnippet| None):
        """
        @fn __init__
        @brief Constructor of the class.
        @param publisher: The name of the publisher.
        @param title: The title of the article.
        @param date: The date the article was published.
        @param authors: A list of authors that have written the article.
        @param content: The content of the article.
        """
        self.publisher = publisher
        self.title = title
        self.date = date
        self.authors = authors
        self.content = content
    
    @staticmethod
    def byID(ID: int, database:pd.DataFrame, highlights: List[Tuple[int, int]] = []):
        """
        @fn byID
        @brief This is a factory method that returns an article by its ID from the database.
        @param ID: The ID of the article.
        @param database: The database containing the articles.
        @return An article object.
        """
        #find the article by its ID
        article = database.iloc[ID]
        # if a field is empty, the value should be None
        publishers = article["source"] if article["source"] else None
        title = article["title"] if article["title"] else None
        date = datetime.datetime.strptime(article["date"], "%Y-%m-%d") if article["date"] else None
        authors = article["author"].split(",") if article["author"] else []
        content = TextSnippet(article["text"], highlights) if article["text"] else None
        return Article(publishers, title, date, authors, content)

    @staticmethod
    @cache
    def byTitle(title: str, database:pd.DataFrame, highlights: List[Tuple[int, int]] = []):
        """
        @fn byTitle
        @brief This is a factory method that returns an article by its title from the database.
        @param title: The title of the article.
        @param database: The database containing the articles.
        @return An article object.
        """
        #find the article by its title
        article = database[database["title"] == title].iloc[0]
        # if a field is empty, the value should be None
        publishers = article["source"] if article["source"] else None
        title = article["title"] if article["title"] else None
        date = datetime.datetime.strptime(article["date"], "%Y-%m-%d") if article["date"] else None
        authors = article["author"].split(",") if article["author"] else []
        content = TextSnippet(article["text"], highlights) if article["text"] else None
        return Article(publishers, title, date, authors, content)

class Author:
    """
    @class Author
    @brief This class is used to define a new author.
    @note An author has a name and a list of articles written by the author.
    """
    name: str = ""
    article: dict[int, str] = {} #ID and title of the articles written by the author
    publishers: list[str] = [] #Names of the publishers that have published articles written by the author on the database
    def __init__(self, name: str, articles: List[TextSnippet], publishers: List[str]):
        """
        @fn __init__
        @brief Constructor of the class.
        @param name: The name of the author.
        @param articles: A list of articles written by the author.
        """
        self.name = name
        self.articles = articles
        self.publishers = publishers
    @staticmethod
    @cache
    def byName(name: str, database:pd.DataFrame):
        """
        @fn byName
        @brief This is a factory method that returns an author by its name from the database.
        @param name: The name of the author.
        @param database: The database containing the authors.
        @return An author object.
        """
        #find all the articles written by the author
        authorArticles = database[database["author"] == name]
        # if a field is empty, the articles and publishers should be empty
        if authorArticles.empty:
            return Author(name, {}, [])
        articles = {article["id"]: article["title"] for index, article in authorArticles.iterrows()} #create a dictionary with the ID and title of the articles written by the author
        publishers = authorArticles["source"].unique()  #get the list of publishers that have published articles written by the author
        return Author(name, articles, publishers)

class NewsPublisher:
    """
    @class NewsPublisher
    @brief This class is used to pass news information to the UI.
    @note The class contains the name of the publisher and the URL of the publisher's website.
    """
    name: str = ""
    articles: dict[int, str] = {} #ID and title of the articles published by the publisher
    authors: list[str] = [] #Names of the authors that have written articles published by the publisher on the database
    def __init__(self, name:str , articles: List[TextSnippet], authors: List[str]):
        """
        @fn __init__
        @brief Constructor of the class.
        @param name: The name of the publisher.
        @param articles: A list of articles published by the publisher.
        @param authors: A list of authors that have written articles published by the publisher.
        """
        self.name
        self.articles = articles
        self.authors = authors
    @staticmethod
    @cache
    def byName(name: str, database:pd.DataFrame):
        """
        @fn byName
        @brief This is a factory method that returns a publisher by its name from the database.
        @param name: The name of the publisher.
        @param database: The database containing the publishers.
        @return A publisher object.
        """
        #find all the articles published by the publisher
        publisherArticles = database[database["source"] == name]
        # if a field is empty, the articles and authors should be empty
        if publisherArticles.empty:
            return NewsPublisher(name, {}, [])
        #create a dictionary with the ID and title of the articles published by the publisher
        articles = {article["id"]: article["title"] for index, article in publisherArticles.iterrows()}
        #get the list of authors that have written articles published by the publisher
        authors = publisherArticles["author"].unique()
        return NewsPublisher(name, articles, authors)