"""
    @file kaggle_login.py
    @brief This file contains the implementation of the Kaggle login window using PyQt5.
    @author Sotarriva Alvarez Isai Roberto
    @date 21/10/2021
	@version 1.0
    @email sotarriva.i.aa@titech.ac.jp
"""
# -----------------------------------------------------------------------------IMPORTS-----------------------------------------------------------------------------#
import sys
import logging
log = logging.getLogger(__name__) # Set up logging
try:
    from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QMessageBox
    from PyQt5.QtCore import pyqtSignal
except ImportError:
    log.critical("PyQt5 not found. You might try installing it via 'pip install PyQt5'.")
    sys.exit(1)

# -----------------------------------------------------------------------------BODY-----------------------------------------------------------------------------#
class KaggleLogin(QWidget):
    login_signal = pyqtSignal(str, str)
    def __init__(self):
        super().__init__()
        self.initUI()
    def initUI(self):
        layout = QVBoxLayout()

        self.user_label = QLabel('Username:', self)
        layout.addWidget(self.user_label)

        self.user_input = QLineEdit(self)
        layout.addWidget(self.user_input)

        self.pass_label = QLabel('Password:', self)
        layout.addWidget(self.pass_label)

        self.pass_input = QLineEdit(self)
        self.pass_input.setEchoMode(QLineEdit.Password)
        layout.addWidget(self.pass_input)

        self.login_button = QPushButton('Login', self)
        self.login_button.clicked.connect(self.handle_login)
        layout.addWidget(self.login_button)

        self.error_label = QLabel('The username and/or password are incorrect', self)
        self.error_label.setStyleSheet("color: red;")
        self.error_label.hide()  # Initially hide the error label
        layout.addWidget(self.error_label)

        self.setLayout(layout)
        self.setWindowTitle('Kaggle Login')
        self.show()

    def handle_login(self):
        username = self.user_input.text()
        password = self.pass_input.text()
        if not self.login_signal.emit(username, password):
            # Successfully logged in
            self.close()
        else:
            # Unsuccessful login attempt
            self.error_label.show() # Show the error label
            self.pass_input.setFocus() # Focus on the password input field