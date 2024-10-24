"""
    @file news_app.py
    @brief This file contains the implementation of the News App window using PyQt5.
    @author Sotarriva Alvarez Isai Roberto
    @date 21/10/2021
    @version 1.0
    @email sotarriva.i.aa@titech.ac.jp
"""
import sys
import logging
log = logging.getLogger(__name__) # Set up logging
try:
    from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QMessageBox
    from PyQt5.QtCore import pyqtSignal
except ImportError:
    log.critical("PyQt5 not found. You might try installing it via 'pip install PyQt5'.")
    sys.exit(1)
class NewsApp(QWidget):
    def __init__(self, rag):
        super().__init__()
        self.rag = rag
        self.initUI()

    def initUI(self):
        self.setWindowTitle('News App')
        self.query_label = QLabel('Enter your query:')
        self.query_input = QLineEdit()
        self.query_button = QPushButton('Get News')
        self.result_label = QLabel('Result will be shown here.')
        self.query_button.clicked.connect(self.get_news)
        self.query_input.returnPressed.connect(self.query_button.click)

        layout = QVBoxLayout()
        layout.addWidget(self.query_label)
        layout.addWidget(self.query_input)
        layout.addWidget(self.query_button)
        layout.addWidget(self.result_label)

        self.setLayout(layout)

    def get_news(self):
        query = self.query_input.text()
        if query:
            result = self.rag.query(query)
            self.result_label.setText(result)
        else:
            QMessageBox.warning(self, 'Input Error', 'Please enter a query.')