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
from .widgets.model_vis_widget import ModelVis

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
        self.resize(1563, 780)

        # load the config file 'config.ini'
        self.config = configparser.ConfigParser()
        basepath = Path(__file__).parent
        config_filepath = str(basepath / "config.ini")
        self.config.read(config_filepath)

        num_samples = int(self.config["main"]["num_samples"])
        sample_selection = str(self.config["main"]["sample_selection"])

        self.image_key_dict = key_dict
        self.image_keys = [key for key in key_dict.keys()]
        self.image_indices = [i for i in range(len(key_dict))]
        self.image_paths = [str(Path(images_filepath) / image_name) for image_name in key_dict.values()]

        # dataloading
        first_image_key = list(key_dict.keys())[0]
        first_image_path = key_dict[first_image_key]
        default_img = str(Path(images_filepath) / first_image_path)

        feature_name = "dummy"
        
        test_feature = da.io.read_feature(
            datafile_path, first_image_key, feature_name, read_projection=False
        )

        self.data_dict = {}
        self.data_dict[feature_name] = {}
        self.data_dict[feature_name]["projection"] = np.zeros((len(self.image_keys), 2))
        self.data_dict[feature_name]["full"] = np.zeros((len(self.image_keys), test_feature.shape[0]))
        self.metadata = {}

        # fill projection and full size data
        for i, k in enumerate(self.image_keys):
            point = da.io.read_feature(
                datafile_path, k, feature_name, read_projection=True
            )
            self.data_dict[feature_name]["projection"][i] = point   

            point = da.io.read_feature(
                datafile_path, k, feature_name, read_projection=False
            )
            self.data_dict[feature_name]["full"][i] = point   
        
            metadata_image = da.read_metadata(datafile_path, k)
            self.metadata[k] = metadata_image

        # Create a layout and add the label, button, input field, and timeline
        main_widget = QWidget()
        Hbox = QHBoxLayout()

        ############## col 1 ##################

        # Create a label to display the photo
        left_img_size = 300
        self.photo_label = QLabel(self)
        self.photo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.photo_label.setFixedSize(left_img_size, left_img_size)

        self.date_label = QLabel("", self)
        self.artist_label = QLabel("", self)
        self.style_label = QLabel("", self)
        self.tag_label = QLabel("", self)
        # Set word wrap for the labels
        self.date_label.setWordWrap(True)
        self.artist_label.setWordWrap(True)
        self.style_label.setWordWrap(True)
        self.tag_label.setWordWrap(True)

        img_info_layout = QGridLayout()
        img_info_layout.addWidget(self.date_label, 0, 0)
        img_info_layout.addWidget(self.artist_label, 0, 1)
        img_info_layout.addWidget(self.style_label, 0, 2)
        img_info_layout.addWidget(self.tag_label, 1, 0, 1, 3)
        img_info = QWidget()
        img_info.setLayout(img_info_layout)
        img_info.setFixedWidth(left_img_size)
        img_info.setFixedHeight(50)

        # Create a button to upload a photo
        self.upload_button = QPushButton("Upload Photo", self)
        self.upload_button.clicked.connect(self.upload_photo)

        # Create a text field for input
        self.input_text = QLineEdit(self)
        self.input_text.setPlaceholderText("Ask a question about the image")

        # Create a label to display the predicted answer
        self.answer_label = QLabel(self)
        self.answer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.submit_button = QPushButton("Submit question", self)
        self.submit_button.clicked.connect(self.get_QA)

        col1 = QVBoxLayout()
        col1.addWidget(self.photo_label)
        col1.addWidget(img_info)
        col1.addWidget(self.upload_button)
        col1.addWidget(self.input_text)
        col1.addWidget(self.submit_button)
        col1.addWidget(self.answer_label)

        self.timeline = HistoryTimelineWidget()
        self.timeline.populate_tree(["Item 1", "Item 2", "Item 3"])
        col1.addWidget(self.timeline)

        ############## col 2 ##################

        self.scatterplot = ScatterplotWidget(
            self.data_dict[feature_name]["projection"], self.image_indices, self.image_paths, self.config
        )
        # self.scatterplot.setGeometry(QtCore.QRect(330, 20, 331, 351))
        # self.scatterplot.point_clicked.connect(self.on_canvas_click)
        # self.scatterplot.plot_widget.scene().sigMouseClicked.connect(self.scatterplot.point_clicked.emit)
        self.scatterplot.plot_widget.scene().sigMouseClicked.connect(
            self.on_canvas_click
        )

        self.dot_plot = QPushButton("Dot Scatterplot")
        self.dot_plot.clicked.connect(self.scatterplot.draw_scatterplot_dots)
        self.images_button = QPushButton("Images Scatterplot")
        self.images_button.clicked.connect(self.scatterplot.draw_scatterplot)

        col2 = QVBoxLayout()
        col2.addWidget(self.scatterplot)
        buttons = QHBoxLayout()
        buttons.addWidget(self.images_button)
        buttons.addWidget(self.dot_plot)
        col2.addLayout(buttons)

        # Create a scroll area for the button container
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        buttons_widget = QWidget()

        button_texts = [
            "Semantic Similarity",
            "Low-Level Similarity",
            "High-Level Similarity",
            "Color Similarity",
            "Selected Object Similarity",
        ]
        button_layout = QHBoxLayout()
        for text in button_texts:
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(
                lambda checked, t=text: self.trigger_scatterplot(
                    button_texts.index(t) + 1
                )
            )
            button_layout.addWidget(button)

        buttons_widget.setLayout(button_layout)
        scroll_area.setWidget(buttons_widget)
        scroll_area.setMinimumHeight(buttons_widget.sizeHint().height() + 18)
        scroll_area.setMaximumHeight(buttons_widget.sizeHint().height() + 18)
        col2.addWidget(scroll_area)

        self.model_vis = ModelVis()
        col2.addWidget(self.model_vis)

        ############## col 3 ##################

        # left_img_size=300
        # self.title_col3 = QLabel("Selected image and its top most similar images" ,self)
        # self.selected_photo = QLabel(self)
        # self.selected_photo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # self.selected_photo.setFixedSize(left_img_size, left_img_size)

        # self.to_left_button = QPushButton("To left", self)
        # self.to_left_button.setFixedSize(50, 50)

        # # Create a container widget to hold the label and button
        # container_widget = QWidget()
        # container_layout = QVBoxLayout(container_widget)
        # container_layout.addWidget(self.selected_photo)
        # container_layout.addWidget(self.to_left_button, alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)

        # col3.addWidget(container_layout)

        # # Create the scroll area for the preview photos
        # scroll_area = QScrollArea()
        # scroll_area.setWidgetResizable(True)

        # # Create a container widget for the preview photos
        # preview_widget = QWidget()
        # preview_layout = QHBoxLayout(preview_widget)

        # # Add smaller preview photos to the container layout
        # preview_photo_paths = ["path/to/preview1.png", "path/to/preview2.png", "path/to/preview3.png"]

        # for preview_photo_path in preview_photo_paths:
        #     preview_photo_label = QLabel()
        #     preview_photo_pixmap = QPixmap(preview_photo_path)
        #     preview_photo_label.setPixmap(preview_photo_pixmap.scaledToWidth(100))  # Scale the preview photo width
        #     preview_layout.addWidget(preview_photo_label)

        # # Set the container widget as the scroll area's widget
        # scroll_area.setWidget(preview_widget)

        # col3.addWidget(scroll_area)

        top_img_size = 150
        self.top1 = QLabel(self)
        self.top1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.top1.setFixedSize(top_img_size, top_img_size)

        self.top2 = QLabel(self)
        self.top2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.top2.setFixedSize(top_img_size, top_img_size)

        self.top3 = QLabel(self)
        self.top3.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.top3.setFixedSize(top_img_size, top_img_size)

        col3 = QVBoxLayout()
        self.top = QHBoxLayout()
        self.top.addWidget(self.top1)
        self.top.addWidget(self.top2)
        self.top.addWidget(self.top3)
        col3.addLayout(self.top)

        # Create a container for the sliders
        slider_container = QGroupBox("Sliders")
        slider_layout = QVBoxLayout()

        sliders = [
            {
                "label": "Semantic Similarity",
                "slider": QtWidgets.QSlider(Qt.Orientation.Horizontal),
            },
            {
                "label": "Color Similarity",
                "slider": QtWidgets.QSlider(Qt.Orientation.Horizontal),
            },
            {
                "label": "Selected Object Similarity",
                "slider": QtWidgets.QSlider(Qt.Orientation.Horizontal),
            },
            {
                "label": "Another Similarity",
                "slider": QtWidgets.QSlider(Qt.Orientation.Horizontal),
            },
        ]

        for slider in sliders:
            label = QLabel(slider["label"])
            slider_widget = slider["slider"]
            slider_widget.setMinimumWidth(40)  # Adjust the minimum width as needed
            # Create a label to display the slider value
            slider_label = QLabel("0")
            slider_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

            # Connect the slider's valueChanged signal to a lambda function that updates the label text
            slider_widget.valueChanged.connect(
                lambda value, label=slider_label: label.setText(str(value))
            )

            slider_layout.addWidget(label)
            slider_layout.addWidget(slider_widget)
            slider_layout.addWidget(slider_label)

        slider_button = QtWidgets.QPushButton("Submit similarity importance")
        slider_button.clicked.connect(lambda: self.trigger_scatterplot(10))
        slider_layout.addWidget(slider_button)
        slider_container.setLayout(slider_layout)
        col3.addWidget(slider_container)

        ############################

        init_id = np.random.randint(len(self.image_indices))

        self.initialize_images(
            self.data_dict[feature_name]["projection"][init_id], self.image_paths[init_id], self.image_keys[init_id]
        )

        Hbox.addLayout(col1)
        Hbox.addLayout(col2)
        Hbox.addLayout(col3)
        main_widget.setLayout(Hbox)

        # self.setCentralWidget(main_widget)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(main_widget)
        self.setCentralWidget(scroll_area)


    def initialize_images(self, init_point, filepath, init_key=None):
        self.display_photo(filepath)
        nearest_indices = self.scatterplot.find_nearest_neighbors(init_point, n=3)
        print("nearest_indices", nearest_indices)

        nearest_images = []
        for near_idx in nearest_indices:
            nearest_images.append(self.image_paths[near_idx])
        self.display_top_photo(nearest_images)

        if init_key is None:
            print("hier when upload image")
            self.update_image_info("unk", "unk", "unk", "unk")
        else:
            print("hier when clicked in plot")
            self.update_image_info(
                self.metadata[init_key]['date'],
                self.metadata[init_key]['artist_name'],
                self.metadata[init_key]['style'],
                self.metadata[init_key]['tags'],
            )

    def upload_photo(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        file_dialog.setNameFilter("Images (*.png *.xpm *.jpg *.jpeg)")

        if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
            filenames = file_dialog.selectedFiles()
            if len(filenames) > 0:
                self.filename = filenames[0]
                # self.display_photo(self.filename)
                # get point aka embedding for new image
                new_point = self.get_point_new_img(self.filename)
                self.initialize_images(new_point, self.filename)

    def update_image_info(self, date, artist, style, tags):
        # Update the label texts
        self.date_label.setText(f"Date: {date}")
        self.artist_label.setText(f"Artist: {artist}")
        self.style_label.setText(f"Style: {style}")
        self.tag_label.setText(f"Tags: {tags}")

    def get_point_new_img(self, filename):
        # image_features = da.io.read_feature(filename)
        # image_features = np.array([[-0.2143612 , 7.560753 ]])
        # image_features = self.compute_umap(image_features)[0]

        random_array = np.random.uniform(low=-10.0, high=10.0, size=(2,))
        print(random_array)
        image_features = random_array
        new_point = image_features
        return new_point

    def display_photo(self, filename):
        pixmap = QPixmap(filename)
        self.photo_label.setPixmap(pixmap.scaledToWidth(self.photo_label.width()))

    def display_top_photo(self, filenames):
        top_photos = [self.top1, self.top2, self.top3]
        for i, filename in enumerate(filenames):
            pixmap = QPixmap(filename)
            top_photos[i].setPixmap(pixmap.scaledToWidth(top_photos[i].width()))

    def get_QA(self):
        text = self.input_text.text()
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
        self.answer_label.setText(f"Predicted answer: {predicted_answer}")

    def trigger_scatterplot(self, button_num):
        if button_num == 1:
            print("Button 1 clicked")
            # Create a new scatterplot with different data or settings
            self.scatterplot.draw_scatterplot()
        elif button_num == 2:
            print("Button 2 clicked")
            # # If you create new instance, remove the existing scatterplot widget from the layout

            # # Create a new scatterplot with different data or settings
            # new_scatterplot = ScatterplotWidget(self.points,self.indices, self.image_paths, self.config)
            # new_scatterplot.setGeometry(QtCore.QRect(330, 20, 331, 351))
            # new_scatterplot.plot_widget.scene().sigMouseClicked.connect(self.on_canvas_click)

            # # Replace the old scatterplot with the new one in the layout
            # self.layout().replaceWidget(self.scatterplot, new_scatterplot)

            # self.scatterplot.deleteLater()
            # # Add the new scatterplot widget to the layout
            # self.scatterplot = new_scatterplot

        elif button_num == 3:
            print("Button 3 clicked")
            # Create a new scatterplot with different data or settings
            self.scatterplot = ScatterplotWidget(...)
        elif button_num == 4:
            print("Button 4 clicked")
            # Create a new scatterplot with different data or settings
            self.scatterplot = ScatterplotWidget(...)
        elif button_num == 5:
            print("Button 5 clicked")
            # Create a new scatterplot with different data or settings
            self.scatterplot = ScatterplotWidget(...)

        elif button_num == 10:
            print("Button 10 sliders submitted")
            # Create a new scatterplot with different data or settings
            self.scatterplot.draw_scatterplot()

    def on_canvas_click(self, ev):
        pos = ev.scenePos()
        print("on canvas click:", pos)
        if ev.button() == Qt.MouseButton.LeftButton:
            print("self.scatterplot.image_items", self.scatterplot.image_items)
            for idx, index, item in self.scatterplot.image_items:
                print("item.mapFromScene(pos)", item, item.mapFromScene(pos))
                if item.contains(item.mapFromScene(pos)):
                    self.scatterplot.selected_point = int(pos.x()), int(pos.y())
                    self.scatterplot.selected_index = index
                    self.scatterplot.plot_index = idx
                    self.clicked_on_point()
                    break

    def clicked_on_point(self):
        print("point/ image clicked, load on the left")
        # Get the selected point
        self.filename = self.scatterplot.get_image_path(self.scatterplot.selected_index)
        self.initialize_images(
            self.scatterplot.selected_point, self.filename, self.scatterplot.plot_index
        )


def start_dashboard(key_dict, dataset_filepath, images_filepath):
    app = QApplication(sys.argv)
    basepath = Path(__file__)
    css_filepath = str(basepath.parent / "theme1.css")
    with open(css_filepath, "r") as file:
        app.setStyleSheet(file.read())
    window = DeepSimDashboard(key_dict, dataset_filepath, images_filepath)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    start_dashboard()
