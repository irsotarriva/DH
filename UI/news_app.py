import sys
import logging
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QLineEdit,
    QPushButton,
    QMessageBox,
    QTextBrowser,
    QVBoxLayout,
    QDialog,
    QListWidget,
    QListWidgetItem,
    QSizePolicy,
    QSplitter,
    QToolTip,
    QDialog
)
from PyQt5.QtGui import QTextCursor
from PyQt5.QtCore import Qt, QEvent
import re
import pandas as pd
log = logging.getLogger(__name__)

class ArticleViewer(QDialog):
    def __init__(self, article_data, paragraph, parent=None):
        super().__init__(parent)
        self.setWindowTitle(article_data['title'])
        self.resize(1600, 1080)

        result_text = QTextBrowser()
        full_text = article_data['full_text']

        # Highlight the cited paragraph
        highlighted_text = self.highlight_paragraph(full_text, paragraph)
        body = ""
        if(article_data['title'] != None):
            if(article_data['title'] != "nan"):
                body = f"<h1>{article_data['title']}</h1>"
            else:
                body = "<h1>(No title)</h1>"
        else:
            body = "<h1>(No title)</h1>"
        if(article_data['date'] != None and article_data['date'] != "nan"):
            body += f"<p><strong>Date:</strong> {article_data['date']}</p>"
        if(article_data['author'] != None and article_data['author'] != "nan"):
            body += f"<p><strong>Author:</strong> {article_data['author']}</p><hr>"
        if(article_data['source'] != None and article_data['source'] != "nan"):
            body += f"<p><strong>Source:</strong> {article_data['source']}</p>"
        body += f"<p>{highlighted_text}</p>"
        result_text.setHtml(body)
        result_text.moveCursor(QTextCursor.Start)

        layout = QVBoxLayout()
        layout.addWidget(result_text)
        self.setLayout(layout)

        # Scroll to the highlighted paragraph
        cursor = result_text.textCursor()
        cursor.movePosition(QTextCursor.Start)
        result_text.setTextCursor(cursor)

    def highlight_paragraph(self, text, paragraph):
        # Escape special characters in paragraph
        escaped_paragraph = re.escape(paragraph.strip())
        # Highlight the paragraph
        highlighted_paragraph = f'<span style="background-color: yellow;">{paragraph.strip()}</span>'
        # Replace the paragraph in the text
        highlighted_text = re.sub(escaped_paragraph, highlighted_paragraph, text, flags=re.MULTILINE)
        return highlighted_text
    
    

class NewsApp(QWidget):
    def __init__(self, rag):
        super().__init__()
        self.rag = rag
        self.similar_articles = None
        self.initUI()
        self._reference_articles = None

    def initUI(self):
        self.setWindowTitle('News App')
        self.query_label = QLabel('Enter your query:')
        self.query_input = QLineEdit()
        self.query_button = QPushButton('Submit', self)
        self.result_text = QTextBrowser()
        self.clear_chat_button = QPushButton('Clear Chat', self)
        self.chat_history = QListWidget()

        self.query_button.clicked.connect(self.get_news)
        self.query_input.returnPressed.connect(self.query_button.click)
        self.clear_chat_button.clicked.connect(self.clear_chat)
        self.clear_chat_button.move(10, 50)  # Adjust the position as needed

        # Set size policies and maximum heights
        self.chat_history.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.chat_history.setMaximumHeight(300)
        self.result_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.result_text.setMaximumHeight(300)

      # Create a splitter to allow resizing between chat_history and result_text
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self.chat_history)
        splitter.addWidget(self.result_text)
        splitter.setSizes([100, 500])  # Initial sizes

        layout = QVBoxLayout()
        layout.addWidget(self.chat_history)
        layout.addWidget(self.result_text)
        layout.addWidget(self.query_label)
        layout.addWidget(self.query_input)
        layout.addWidget(self.query_button)
        layout.addWidget(self.clear_chat_button)

        self.setLayout(layout)

        # Handle link clicks
        self.result_text.setOpenLinks(False)
        self.result_text.anchorClicked.connect(self.open_article)

        # Connect hover event
        self.result_text.viewport().installEventFilter(self)
        self.floating_window = None

    def get_news(self):
        #if there is already a result_text, move it to the chat history
        if self.result_text.toPlainText() != '':
            self.add_to_chat_history(self.result_text.toPlainText())
            self.result_text.clear()
        self.add_to_chat_history(self.query_input.text())
        query = self.query_input.text()
        if query:
            result, similar_articles = self.rag.generate_response(query)
            #add the similar articles to the list of reference articles.
            if self._reference_articles is None:
                self._reference_articles = similar_articles
            else:
                self._reference_articles = pd.concat([self._reference_articles, similar_articles], ignore_index=True)
            result_with_links = self.replace_uids_with_links(result, self._reference_articles)
            result_with_links = self.format_to_html(result_with_links)
            self.result_text.setHtml(result_with_links)
        else:
            QMessageBox.warning(self, 'Input Error', 'Please enter a query.')
    
    def format_to_html(self, text:str)->str:
        #The AI uses "* " to indicate a new unordered list item (literal asterisk followed by a space)
        text = re.sub(r'\* (.*?)\n', r'<ul><li>\1</li></ul>', text)
        #the AI uses "\n" to indicate a new line
        text = text.replace("\n", "<br>")
        #the AI uses double asterix "**text**" to indicate a bold text
        # Find all bold text and replace with <b>bold text</b>
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        return text

    def replace_uids_with_links(self, text:str, reference_articles:pd.DataFrame)->str:
        # Find all citations. Each citation is on the format @{article_id}/{paragraph_id} where @{article_id}/{paragraph_id} is the UID, article_id is an integer and paragraph_id is also an integer. Example: @123/4
        # The AI uses [UID] to indicate a citation. Replace [UID] with a hyperlink to the article.
        #The AI sometimes writes multiple citations in the same pair of brackets. These should be separated into individual pairs of brackets. Ex. [UID:@123/4, @456/7] should be replaced with [@123/4][@456/7]
        # Define a regex pattern to match all the possible UID formats
        # sometimes the AI writes [UID:123/4] , or [123/4] or [UID:@123/4] or [UID: @123/4] instead of [@123/4]. These should also be replaced.
        # Define a regex pattern to match all the possible UID formats
        pattern = re.compile(r'\[@?(\d+/\d+)\]|\[UID:?\s?@?(\d+/\d+)\]|\[UID:?\s?(\d+/\d+)\]')

        # Function to replace matched UIDs with hyperlinks
        def replace_match(match):
            uid = match.group(1) or match.group(2) or match.group(3)
            uid = "@" + uid  # Add the "@" symbol to the UID to match the format in the reference_articles DataFrame
            matched_articles = self._reference_articles[self._reference_articles['UID'] == uid]
            
            if matched_articles.empty:
                title = uid  # Fallback to using the UID if no matching article is found
            else:
                article = matched_articles.iloc[0]
                title_text = article['title']
                if title_text == 'nan':
                    title_text = uid
                elif title_text == None:
                    title_text = uid
                title = title_text
            
            return f'<a href="{uid}">[{title}]</a>'

        # Substitute all matches in the text
        result_text = pattern.sub(replace_match, text)
        return result_text

    def open_article(self, link: str):
        uid = link.toString()
        log.debug(f'Opening article with UID: {uid}')
        if uid in self._reference_articles['UID'].values:
            article_data = self._reference_articles[self._reference_articles['UID'] == uid].iloc[0]
            article_viewer = ArticleViewer(article_data, article_data['paragraph'])
            article_viewer.exec_()
        else:
            QMessageBox.warning(self, 'Article Not Found', 'The article you are trying to access is not in the reference')
    def clear_chat(self):
        self.query_input.clear()
        self.result_text.clear()
        self.chat_history.clear()
        self._reference_articles = None
        self.rag.clear()

    def add_to_chat_history(self, text):
            query_item = QListWidgetItem(text)
            self.chat_history.addItem(query_item)
            self.chat_history.scrollToBottom()

    def eventFilter(self, source, event):
        if event.type() == QEvent.HoverMove and source is self.result_text.viewport():
            cursor = self.result_text.cursorForPosition(event.pos())
            cursor.select(QTextCursor.WordUnderCursor)
            word = cursor.selectedText()
            if word.startswith('@'):  # Assuming word is a reference
                self.show_floating_window(event.globalPos(), word)
            else:
                self.hide_floating_window()
        elif event.type() == QEvent.Leave and source is self.result_text.viewport():
            self.hide_floating_window()
        return super().eventFilter(source, event)

    def show_floating_window(self, pos, reference):
        if self.floating_window is None:
            self.floating_window = ArticleViewer(self.article_data, self.paragraph, self)
        self.floating_window.move(pos)
        self.floating_window.show()

    def hide_floating_window(self):
        if self.floating_window is not None:
            self.floating_window.hide()

if __name__ == '__main__':
    from rag import RAG

    app = QApplication(sys.argv)
    rag = RAG()
    news_app = NewsApp(rag)
    news_app.show()
    sys.exit(app.exec_())