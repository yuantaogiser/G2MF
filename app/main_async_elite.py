from utils_main import create_main_window
from PyQt5.QtWidgets import QApplication
from utils_model import *
import sys

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.threads = []
    window = create_main_window()
    window.show()
    sys.exit(app.exec_())