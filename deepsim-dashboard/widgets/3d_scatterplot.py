import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QGridLayout, QLineEdit
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt


from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from widgets.wordcloud_widget import WordcloudWidget
from widgets.embbedding_scatterplot_widget import ScatterplotWidget

import configparser
import h5py
import numpy as np
import umap


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Viewer")

        layout = QGridLayout()
        self.setLayout(layout)


def main():
    app = QApplication(sys.argv)
    with open("theme1.css","r") as file:
        app.setStyleSheet(file.read())

    window = Window()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
