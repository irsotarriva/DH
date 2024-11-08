"""
    @file main.py
    @brief Main file, it contains the GUI and calls to the backend located in the rag.py file.
    @author Sotarriva Alvarez Isai Roberto
    @date 21/10/2021
    @version 1.0
    @email sotarriva.i.aa@titech.ac.jp
"""
# -----------------------------------------------------------------------------IMPORTS-----------------------------------------------------------------------------#
import sys
import logging
import log.log as log # specific configuration for the logging.
import UI.news_app as ui # implementation of the GUI.
try:
    from PyQt5.QtWidgets import QApplication
except ImportError:
    log.critical("PyQt5 not found. You might try installing it via 'pip install PyQt5'.")
    sys.exit(1)

# -----------------------------------------------------------------------------BODY-----------------------------------------------------------------------------#
def main() -> None:
    """ Main function to run the program. """
    log.setup_logging()
    logging.basicConfig(level="INFO")
    import rag #implementation of the RAG model.
    MyRAG = rag.RAG()
    app = QApplication(sys.argv)
    news_app = ui.NewsApp(MyRAG)
    news_app.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
