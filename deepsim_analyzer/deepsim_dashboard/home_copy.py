import sys
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QFileDialog,
    QGridLayout,
    QLineEdit,
    QMessageBox,
    QFrame
)
from PyQt6.QtWidgets import QGroupBox, QScrollArea
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt, QRect
from PyQt6 import QtWidgets, QtCore

from transformers import ViltProcessor, ViltForQuestionAnswering

from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from .widgets.wordcloud_widget import WordcloudWidget
from .widgets.embbedding_scatterplot_widget import ScatterplotWidget
from .widgets.tree_widget import HistoryTimelineWidget
from .widgets.model_vis_widget import ModelVis
from .widgets.timeline_widget import TimelineWindow
from .col_functions import Col1Widget, Col2Widget,Col3Widget

import configparser
from pathlib import Path
import numpy as np
import random

import deepsim_analyzer as da


class DeepSimDashboard(QMainWindow):
    def __init__(self, key_dict, datafile_path, images_filepath):
        super().__init__()
        np.random.seed(3)  # Set the seed to a specific value

        self.setWindowTitle("Image Viewer")
        # self.resize(1366, 768)
        # self.resize(1566, 768)
        self.resize(1530, 786)

        # load the config file 'config.ini'
        self.config = configparser.ConfigParser()
        basepath = Path(__file__).parent
        config_filepath = str(basepath / "config.ini")
        self.config.read(config_filepath)

        num_samples = int(self.config["main"]["num_samples"])
        sample_selection = str(self.config["main"]["sample_selection"])

        self.image_key_dict = key_dict
        self.image_keys = [key for key in key_dict.keys()] # hashes
        self.image_indices = [i for i in range(len(key_dict))]
        self.image_paths = [str(Path(images_filepath) / image_name) for image_name in key_dict.values()]

        # dataloading
        first_image_key = list(key_dict.keys())[0]
        first_image_path = key_dict[first_image_key]
        default_img = str(Path(images_filepath) / first_image_path)

        self.feature_name = "dummy"
        test_feature = da.io.read_feature(
            datafile_path, first_image_key, self.feature_name, read_projection=False
        )

        self.data_dict = {}
        self.data_dict[self.feature_name] = {}
        self.data_dict[self.feature_name]["projection"] = np.zeros((len(self.image_keys), 2))
        self.data_dict[self.feature_name]["full"] = np.zeros((len(self.image_keys), test_feature.shape[0]))
        self.metadata = {}

        # fill projection and full size data
        for i, k in enumerate(self.image_keys):
            point = da.io.read_feature(
                datafile_path, k, self.feature_name, read_projection=True
            )
            self.data_dict[self.feature_name]["projection"][i] = point   

            point = da.io.read_feature(
                datafile_path, k, self.feature_name, read_projection=False
            )
            self.data_dict[self.feature_name]["full"][i] = point   
        
            metadata_image = da.read_metadata(datafile_path, k)
            self.metadata[k] = metadata_image

        self.filename=''

        main_widget = QWidget()
        Hbox = QHBoxLayout()

        self.col1 = Col1Widget( self.upload_photo, self.get_QA)
        self.col2 = Col2Widget(self.data_dict,self.feature_name, self.image_indices,self.image_paths, self.config,self.on_canvas_click,self.trigger_scatterplot)
        self.col3 = Col3Widget(self.display_photo, self.trigger_scatterplot, self.filename)

        init_id = np.random.randint(len(self.image_indices))
        self.filename=self.image_paths[init_id]

        self.initialize_images(
            self.data_dict[self.feature_name]["projection"][init_id], self.image_paths[init_id], self.image_keys[init_id]
        )
        self.display_photo(self.image_paths[init_id], left=True)

        # Hbox.addLayout(self.col1)
        # Hbox.addLayout(self.col2)
        # Hbox.addLayout(self.col3)

        Hbox.addLayout(self.col1.layout())
        Hbox.addLayout(self.col2.layout())
        Hbox.addLayout(self.col3.layout())

        # Hbox.addWidget(self.col1)
        # Hbox.addWidget(self.col2)
        # Hbox.addWidget(self.col3)
        main_widget.setLayout(Hbox)

        # self.setCentralWidget(main_widget)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(main_widget)
        self.setCentralWidget(scroll_area)

    def upload_photo(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        file_dialog.setNameFilter("Images (*.png *.xpm *.jpg *.jpeg)")

        if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
            filenames = file_dialog.selectedFiles()
            if len(filenames) > 0:
                self.filename = filenames[0]
                # get point for new image
                new_point = self.get_point_new_img(self.filename)
                self.initialize_images(new_point, self.filename)

    def get_point_new_img(self, filename):
        # image_features = da.io.read_feature(filename)
        random_array = np.random.uniform(low=-10.0, high=10.0, size=(2,))
        print('random_array', random_array)
        image_features = random_array
        new_point = image_features
        return new_point

    def initialize_images(self, init_point, filepath, init_key=None, left=False):
        self.display_photo(filepath)
        nearest_indices = self.col2.scatterplot.find_nearest_neighbors(init_point, n=3)
        print("nearest_indices", nearest_indices)

        nearest_images = []
        for near_idx in nearest_indices:
            nearest_images.append(self.image_paths[near_idx])
        self.display_top_photo(nearest_images)

        if init_key is None:
            print("here when upload image")
            self.update_image_info("unk", "unk", "unk", "unk")
        else:
            print("here when clicked in plot")
            self.update_image_info(
                self.metadata[init_key]['date'],
                self.metadata[init_key]['artist_name'],
                self.metadata[init_key]['style'],
                self.metadata[init_key]['tags'],
            )

    def update_image_info(self, date, artist, style, tags):
        # Update the label texts
        self.col1.date_label.setText(f"Date: {int(date)}")
        self.col1.artist_label.setText(f"Artist: {artist}")
        self.col1.style_label.setText(f"Style: {style}")
        self.col1.tag_label.setText(f"Tags: {tags}")

    def display_photo(self, filename, left=False):
        if left:
            label = self.col1.photo_label
        else:
            label = self.col3.selected_photo
        print('filename',filename)
        pixmap = QPixmap(filename)
        label.setPixmap(pixmap.scaledToWidth(label.width()))

    def display_top_photo(self, filenames):
        for i, filename in enumerate(filenames):
            if i < len(self.col3.preview_photo_labels):
                label = self.col3.preview_photo_labels[i]
                pixmap = QPixmap(filename)
                label.setPixmap(pixmap.scaledToWidth(label.width()))

    def get_QA(self):
        text = self.col1.input_text.text()
        image = Image.open(self.filename)
        processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        model = ViltForQuestionAnswering.from_pretrained(
            "dandelin/vilt-b32-finetuned-vqa"
        )
        # prepare inputs
        encoding = processor(image, text, return_tensors="pt")
        # forward pass
        outputs = model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        predicted_answer = model.config.id2label[idx]
        # Display the predicted answer
        self.col1.answer_label.setText(f"Predicted answer: {predicted_answer}")

    def trigger_scatterplot(self, button_num):
        if button_num == 1:
            print("Button 1 clicked")
            # Create a new scatterplot with different data or settings
            self.col2.scatterplot.draw_scatterplot()
        elif button_num == 2:
            print("Button 2 clicked")
            self.col2.scatterplot.draw_scatterplot()
        elif button_num == 3:
            print("Button 3 clicked")
            self.col2.scatterplot = ScatterplotWidget(...)
        elif button_num == 4:
            print("Button 4 clicked")
            self.col2.scatterplot = ScatterplotWidget(...)
        elif button_num == 5:
            print("Button 5 clicked")
            self.col2.scatterplot = ScatterplotWidget(...)

        elif button_num == 10:
            print("Button 10 sliders submitted")
            self.col2.scatterplot.draw_scatterplot()

    def on_canvas_click(self, ev):
        pos = ev.scenePos()
        print("on canvas click:", pos)
        if ev.button() == Qt.MouseButton.LeftButton:
            # print("self.col2.scatterplot.image_items", self.col2.scatterplot.image_items)
            for idx, index, item in self.col2.scatterplot.image_items:
                # print("item.mapFromScene(pos)", item, item.mapFromScene(pos))
                if item.contains(item.mapFromScene(pos)):
                    self.scol2.catterplot.selected_point = int(pos.x()), int(pos.y())
                    self.col2.scatterplot.selected_index = index
                    self.col2.scatterplot.plot_index = idx
                    print('selected_index==plot_index?',index==idx)
                    self.clicked_on_point()
                    break

    def clicked_on_point(self):
        print("point/ image clicked, load on the left")
        self.filename= self.image_paths[self.col2.scatterplot.selected_index]
        self.initialize_images(
            self.col2.scatterplot.selected_point, self.filename, self.image_keys[self.col2.scatterplot.plot_index]
        )

    # def closeEvent(self, event):
    #     reply = QMessageBox.question(self, "Close Window", "Are you sure you want to close the window?",
    #                                  QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
    #     if reply == QMessageBox.StandardButton.Yes:
    #         event.accept()
    #     else:
    #         event.ignore()

def start_dashboard(key_dict, dataset_filepath, images_filepath):
    app = QApplication(sys.argv)
    basepath = Path(__file__)
    css_filepath = str(basepath.parent / "theme1.css")
    with open(css_filepath, "r") as file:
        app.setStyleSheet(file.read())
    window = DeepSimDashboard(key_dict, dataset_filepath, images_filepath)
    window.show()
    sys.exit(app.exec())

