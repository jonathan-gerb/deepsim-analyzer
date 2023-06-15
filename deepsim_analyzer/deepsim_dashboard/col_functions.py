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

import configparser
from pathlib import Path
import numpy as np
import random

import deepsim_analyzer as da

class Col1Widget(QWidget):
    def __init__(self, upload_photo, get_QA):
        super().__init__()

        self.upload_photo=upload_photo
        self.get_QA=get_QA

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

        # col1 = QVBoxLayout()
        # self.setLayout(col1)
        # col1.addWidget(self.photo_label)
        # col1.addWidget(img_info)
        # col1.addWidget(self.upload_button)
        # col1.addWidget(self.input_text)
        # col1.addWidget(self.submit_button)
        # col1.addWidget(self.answer_label)

        main_photo=''
        self.timeline = TimelineWindow(main_photo)
        self.timeline.draw_timeline( main_photo)
        # col1.addWidget(self.timeline)

        # Create a layout for Col1Widget
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.photo_label)
        self.layout.addWidget(img_info)
        self.layout.addWidget(self.upload_button)
        self.layout.addWidget(self.input_text)
        self.layout.addWidget(self.submit_button)
        self.layout.addWidget(self.answer_label)
        self.layout.addWidget(self.timeline)
    
   

class Col2Widget(QWidget):
    def __init__(self, data_dict,feature_name, image_indices, image_paths, config,on_canvas_click,trigger_scatterplot):
        super().__init__()

        self.data_dict,self.feature_name, self.image_indices,self.image_paths,self.config,self.on_canvas_click,self.trigger_scatterplot=data_dict,feature_name, image_indices,image_paths,config,on_canvas_click,trigger_scatterplot

        self.scatterplot = ScatterplotWidget(
            self.data_dict[self.feature_name]["projection"], self.image_indices, self.image_paths, self.config
        )
        self.scatterplot.plot_widget.scene().sigMouseClicked.connect(
            self.on_canvas_click
        )

        self.dot_plot = QPushButton("Dot Scatterplot")
        self.dot_plot.clicked.connect(self.scatterplot.draw_scatterplot_dots)
        self.images_button = QPushButton("Images Scatterplot")
        self.images_button.clicked.connect(self.scatterplot.draw_scatterplot)

        # col2 = QVBoxLayout()
        # col2.addWidget(self.scatterplot)
        # buttons = QHBoxLayout()
        # buttons.addWidget(self.images_button)
        # buttons.addWidget(self.dot_plot)
        # col2.addLayout(buttons)

        # Create a layout for Col2Widget
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.scatterplot)
        buttons = QHBoxLayout()
        buttons.addWidget(self.images_button)
        buttons.addWidget(self.dot_plot)
        self.layout.addLayout(buttons)

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
        # col2.addWidget(scroll_area)
        self.layout.addWidget(scroll_area)

        self.model_vis = ModelVis()
        # col2.addWidget(self.model_vis)
        self.layout.addWidget(self.model_vis)

        


class Col3Widget(QWidget):
    def __init__(self,display_photo, trigger_scatterplot,filename):
        super().__init__()

        self.display_photo, self.trigger_scatterplot, self.filename=display_photo, trigger_scatterplot,filename

        col3 = QVBoxLayout()

        left_img_size=300
        self.title_col3 = QLabel("Selected image and its top most similar images" ,self)
        self.selected_photo = QLabel(self)
        self.selected_photo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.selected_photo.setFixedSize(left_img_size, left_img_size)

        self.to_left_button = QPushButton("To left", self)
        self.to_left_button.setFixedSize(50, 50)
        self.to_left_button.clicked.connect(lambda: self.display_photo(self.filename, left=True))

        # Create a frame as the container
        container_frame = QFrame(self)
        # Create layout for the container
        container_layout = QVBoxLayout(container_frame)
        container_layout.addWidget(self.selected_photo)
        container_layout.addWidget(self.to_left_button, alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)
        
        col3.addWidget(container_frame)

        # Create the scroll area for the preview photos
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        # Create a container widget for the preview photos
        preview_widget = QWidget()
        preview_layout = QHBoxLayout(preview_widget)

        # Add smaller preview photos to the container layout
        num_of_previews = 3 # remove hard code later
        self.preview_photo_labels = []
        for i in range(num_of_previews):
            preview_photo_label = QLabel()
            preview_layout.addWidget(preview_photo_label)
            self.preview_photo_labels.append(preview_photo_label)

        # Set the container widget as the scroll area's widget
        scroll_area.setWidget(preview_widget)

        col3.addWidget(scroll_area)


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