"""
    @file article_snippet.py
    @brief This file contains the class ArticleSnippet, which is a class that represents a snippet of an article. It contains the publisher, the title, date, author, and the content of the article all of that defined as attributes of the News class.
    @author Sotarriva Alvarez Isai Roberto
    @date 22/10/2021
    @version 1.0
    @email sotarriva.i.aa@titech.ac.jp
"""
# ______________________________________________________________________________ IMPORTS ______________________________________________________________________________
import datetime
import logging
log = logging.getLogger(__name__)
try:
    from PyQt5 import QtWidgets, QtGui, QtCore
except ImportError:
    log.critical("PyQt5 is not installed")
from objects import TextSnippet, Article
# ______________________________________________________________________________ BODY ______________________________________________________________________________
class ArticleSnippet(QWidget):
    """
    @class ArticleSnippet
    @brief This class is used to pass article information to the UI.
    @note The Article class contains the publisher, the title, date, author, and the content of the article to be displayed in the UI.
    """
    article : obj.Article = None
    def __init__(self, article: Article):
        """
        @fn __init__
        @brief Constructor of the class.
        @param article: Article object that contains the information of the article.
        """
        self.article = article
        super().__init__()
        self.initUI()

    def initUI(self):
        """
        @fn initUI
        @brief This function initializes the UI of the article snippet.
        """
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        self.layout.setAlignment(Qt.AlignTop)
        self.setLayout(self.layout)
        self.layout.addWidget(TextSnippet(self.article.publisher, 16, True))
        self.layout.addWidget(TextSnippet(self.article.title, 20, True))
        self.layout.addWidget(TextSnippet(self.article.date.strftime("%d/%m/%Y"), 12, True))
        self.layout.addWidget(TextSnippet(self.article.author, 12, True))
        self.layout.addWidget(TextSnippet(self.article.content.show(200), 14, False))
        self.layout.addStretch(1)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(10)
        self.setStyleSheet("background-color: #f0f0f0; border-radius: 10px; padding: 10px;")
        self.setFixedWidth(400)
        self.setFixedHeight(400)
        self.show()
