import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QGridLayout, QLineEdit
from PyQt6.QtWidgets import QGroupBox, QScrollArea
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt, QRect
from PyQt6 import QtWidgets, QtCore 

from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from widgets.wordcloud_widget import WordcloudWidget
from widgets.embbedding_scatterplot_widget import ScatterplotWidget
from widgets.tree_widget import HistoryTimelineWidget

import configparser
import h5py
import numpy as np
import umap

import deepsim_analyzer as da

class MainWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()
        np.random.seed(3)  # Set the seed to a specific value

        self.setWindowTitle("Image Viewer")
        self.resize(1100, 800)

        # Create a label to display the photo
        self.photo_label = QLabel(self)
        self.photo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.photo_label.setFixedSize(300, 300)  # Set the fixed size to 200x200

        # Create a label to display the photo
        self.top1 = QLabel(self)
        self.top1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.top1.setFixedSize(200, 200)  # Set the fixed size to 200x200

        # Create a label to display the photo
        self.top2 = QLabel(self)
        self.top2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.top2.setFixedSize(200, 200)  # Set the fixed size to 200x200

        # Create a label to display the photo
        self.top3 = QLabel(self)
        self.top3.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.top3.setFixedSize(200, 200)  # Set the fixed size to 200x200

        # Create a button to upload a photo
        self.upload_button = QPushButton("Upload Photo", self)
        self.upload_button.clicked.connect(self.upload_photo)

        # Create a text field for input
        self.input_text = QLineEdit(self)
        self.input_text.setPlaceholderText("Enter text")

        # Create labels for the timeline
        self.date_label = QLabel("Date:", self)
        self.event_label = QLabel("Event:", self)
        self.image_label = QLabel("Image:", self)

        # # Create a layout for the timeline
        img_info = QGridLayout()
        img_info.addWidget(self.date_label, 0, 0)
        img_info.addWidget(self.event_label, 0, 1)
        img_info.addWidget(self.image_label, 0, 2)

        # Create a label to display the predicted answer
        self.answer_label = QLabel(self)
        self.answer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.submit_button = QPushButton("Submit question", self)
        self.submit_button.clicked.connect(self.get_QA)

        # load the config file 'config.ini'
        config = configparser.ConfigParser()
        config.read('config.ini')

        data_path = config['main']['pkl_path']
        images_path = config['main']['images_path']

        num_samples = int(config['main']['num_samples'])
        sample_selection = str(config['main']['sample_selection'])
        # dataloading
        df = pd.read_pickle(data_path)

        with h5py.File(images_path, "r") as hf:
            image_features = hf["image_features"][:]

        # image_features=  da.io.read_feature(datafile_path, img_hash, feature_name)

        # select num_samples samples between
        if sample_selection == 'random':
            random_indices = np.random.choice(
                len(df), num_samples, replace=False)
            tags = df['tags'].iloc[random_indices].values
            points = df[['umap_x', 'umap_y']].iloc[random_indices].values
            image_features = image_features[random_indices]
            img_paths = df['image'].iloc[random_indices].values
            indices=random_indices

        if sample_selection == 'first':
            tags = df['tags'].iloc[:num_samples].values
            points = df[['umap_x', 'umap_y']].iloc[:num_samples].values
            image_features = image_features[:num_samples]
            img_paths = df['image'].iloc[:num_samples].values
            indices=num_samples

        # recompute the tag embedding coordinates to umap
        if bool(config['main']['recompute_embedding']):
            points = self.compute_umap(image_features)

        self.config=config
        self.points=points
        self.indices=indices
        self.img_paths=img_paths
        init_id =np.random.randint(len(points))
        print(points[init_id], img_paths[init_id])
        self.scatterplot = ScatterplotWidget(points,indices, img_paths, config)
        self.scatterplot.setGeometry(QtCore.QRect(330, 20, 331, 351))
        # self.scatterplot.point_clicked.connect(self.on_canvas_click)
        # self.scatterplot.plot_widget.scene().sigMouseClicked.connect(self.scatterplot.point_clicked.emit)
        self.scatterplot.plot_widget.scene().sigMouseClicked.connect(self.on_canvas_click)

        # self.reset_button = QPushButton("Reset Scatterplot")
        # self.reset_button.clicked.connect(self.scatterplot.reset_scatterplot)

        self.dot_plot = QPushButton("Dot Scatterplot")
        self.dot_plot.clicked.connect(self.scatterplot.draw_scatterplot_dots)
        self.images_button = QPushButton("Images Scatterplot")
        self.images_button.clicked.connect(self.scatterplot.draw_scatterplot)


        self.timeline = HistoryTimelineWidget()
        # read_feature(datafile_path, img_hash, feature_name)
        self.initialize_images(points[init_id], img_paths[init_id])

        # Create a layout and add the label, button, input field, and timeline
        main_widget = QWidget()
        Hbox = QHBoxLayout()
        col1 = QVBoxLayout()
        col1.addWidget(self.photo_label)
        col1.addLayout(img_info)
        col1.addWidget(self.upload_button)
        col1.addWidget(self.input_text)
        col1.addWidget(self.answer_label)
        col1.addWidget(self.submit_button)

        col2 = QVBoxLayout()
        col2.addWidget(self.scatterplot)
        buttons = QHBoxLayout()
        buttons.addWidget(self.images_button)
        buttons.addWidget(self.dot_plot)
        # buttons.addWidget(self.reset_button)
        col2.addLayout(buttons)

        # Create a scroll area for the button container
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        # Create a widget to hold the buttons
        buttons_widget = QWidget()

        # Create push buttons and connect them to the scatter plot
        self.pushButton = QtWidgets.QPushButton("Semantic Similarity")
        self.pushButton.clicked.connect(lambda: self.trigger_scatterplot(1))

        self.pushButton_2 = QtWidgets.QPushButton("Low-Level Similarity")
        self.pushButton_2.clicked.connect(lambda: self.trigger_scatterplot(2))

        self.pushButton_3 = QtWidgets.QPushButton("High-Level Similarity")
        self.pushButton_3.clicked.connect(lambda: self.trigger_scatterplot(3))

        self.pushButton_4 = QtWidgets.QPushButton("Color Similarity")
        self.pushButton_4.clicked.connect(lambda: self.trigger_scatterplot(4))

        self.pushButton_5 = QtWidgets.QPushButton("Selected Object Similarity")
        self.pushButton_5.clicked.connect(lambda: self.trigger_scatterplot(5))

        self.horizontalScrollBar = QtWidgets.QScrollBar()
        self.horizontalScrollBar.setGeometry(QtCore.QRect(330, 370, 331, 20))
        self.horizontalScrollBar.setOrientation(QtCore.Qt.Orientation.Horizontal)

        # Create a layout for the buttons
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(10)  # Set spacing between buttons

        # Add the push buttons to the layout
        buttons_layout.addWidget(self.pushButton)
        buttons_layout.addWidget(self.pushButton_2)
        buttons_layout.addWidget(self.pushButton_3)
        buttons_layout.addWidget(self.pushButton_4)
        buttons_layout.addWidget(self.pushButton_5)

        # Set the buttons layout to the buttons widget
        buttons_widget.setLayout(buttons_layout)

        # Set the buttons widget as the widget for the scroll area
        scroll_area.setWidget(buttons_widget)

        # Add the scroll area to the main layout
        col2.addWidget(scroll_area)

        col2.addWidget(self.timeline)

        col3 = QVBoxLayout()
        self.top = QHBoxLayout()
        self.top.addWidget(self.top1)
        self.top.addWidget(self.top2)
        self.top.addWidget(self.top3)
        col3.addLayout(self.top)

        # Create a container for the sliders
        slider_container = QGroupBox("Sliders")
        slider_layout = QVBoxLayout()

        self.label1 = QLabel("Semantic Similarity")
        self.horizontalSlider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.horizontalSlider.setGeometry(QRect(50, 50, 160, 18))
        self.horizontalSlider.setObjectName("horizontalSlider")

        # Create a label to display the slider value
        self.slider_label1 = QLabel("0")
        self.slider_label1.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Connect the slider's valueChanged signal to a lambda function that updates the label text
        self.horizontalSlider.valueChanged.connect(lambda value: self.slider_label1.setText(str(value)))

        slider_layout.addWidget(self.label1)
        slider_layout.addWidget(self.horizontalSlider)
        slider_layout.addWidget(self.slider_label1)

        # Repeat the above steps for the remaining sliders
        self.label2 = QLabel("Color Similarity")
        self.horizontalSlider_2 = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.horizontalSlider_2.setGeometry(QRect(50, 100, 160, 18))
        self.horizontalSlider_2.setObjectName("horizontalSlider_2")
        self.slider_label2 = QLabel("0")
        self.slider_label2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.horizontalSlider_2.valueChanged.connect(lambda value: self.slider_label2.setText(str(value)))

        self.label3 = QLabel("Selected Object Similarity")
        self.horizontalSlider_3 = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.horizontalSlider_3.setGeometry(QRect(50, 140, 160, 18))
        self.horizontalSlider_3.setObjectName("horizontalSlider_3")
        self.slider_label3 = QLabel("0")
        self.slider_label3.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.horizontalSlider_3.valueChanged.connect(lambda value: self.slider_label3.setText(str(value)))

        self.label4 = QLabel("Another Similarity")
        self.horizontalSlider_4 = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.horizontalSlider_4.setGeometry(QRect(50, 190, 160, 18))
        self.horizontalSlider_4.setObjectName("horizontalSlider_4")
        self.slider_label4 = QLabel("0")
        self.slider_label4.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.horizontalSlider_4.valueChanged.connect(lambda value: self.slider_label4.setText(str(value)))

        # Add the sliders and labels to the slider container layout
        slider_layout.addWidget(self.label2)
        slider_layout.addWidget(self.horizontalSlider_2)
        slider_layout.addWidget(self.slider_label2)

        slider_layout.addWidget(self.label3)
        slider_layout.addWidget(self.horizontalSlider_3)
        slider_layout.addWidget(self.slider_label3)

        slider_layout.addWidget(self.label4)
        slider_layout.addWidget(self.horizontalSlider_4)
        slider_layout.addWidget(self.slider_label4)

        slider_container.setLayout(slider_layout)

        # Add the slider container to the main layout
        col3.addWidget(slider_container)


        Hbox.addLayout(col1)
        Hbox.addLayout(col2)
        Hbox.addLayout(col3)
        main_widget.setLayout(Hbox)

        self.setCentralWidget(main_widget)

    def compute_umap(self, image_features):
        return umap.UMAP(n_neighbors=6, n_components=2, metric='cosine').fit_transform(image_features)


    def upload_photo(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        file_dialog.setNameFilter("Images (*.png *.xpm *.jpg *.jpeg)")

        if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
            filenames = file_dialog.selectedFiles()
            if len(filenames) > 0:
                self.filename = filenames[0]
                self.display_photo(self.filename)

    def display_photo(self, filename):
        pixmap = QPixmap(filename)
        self.photo_label.setPixmap(
            pixmap.scaledToWidth(self.photo_label.width()))

    def display_top_photo(self, filenames):
        top_photos = [self.top1, self.top2, self.top3]
        for i, filename in enumerate(filenames):
            pixmap = QPixmap(filename)
            top_photos[i].setPixmap(pixmap.scaledToWidth(top_photos[i].width()))

    def get_QA(self):
        text = self.input_text.text()
        image = Image.open(self.filename)
        processor = ViltProcessor.from_pretrained(
            "dandelin/vilt-b32-finetuned-vqa")
        model = ViltForQuestionAnswering.from_pretrained(
            "dandelin/vilt-b32-finetuned-vqa")
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
            self.scatterplot = ScatterplotWidget(self.points, self.indices, self.img_paths, self.config)
            self.scatterplot.setGeometry(QtCore.QRect(330, 20, 331, 351))
            self.scatterplot.draw_scatterplot
        elif button_num == 2:
            print("Button 2 clicked")
            # Create a new scatterplot with different data or settings
            self.scatterplot = ScatterplotWidget(self.points, self.config)
            self.scatterplot.setGeometry(QtCore.QRect(330, 20, 331, 351))
            self.scatterplot.draw_scatterplot
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
    
    def initialize_images(self, init_point,filename):
        self.display_photo(filename)
        # Find the 3 nearest neighbors
        nearest_indices = self.scatterplot.find_nearest_neighbors(
            init_point, n=3)
        # Generate images from nearest neighbor embeddings
        nearest_images = []
        for near_idx in nearest_indices:
            nearest_images.append(self.img_paths[near_idx])
        self.display_top_photo(nearest_images)
        # Update the layout to reflect the new scatterplot
        # self.update_layout()

    def on_canvas_click(self, ev):
        pos = ev.scenePos()
        print('on canvas click:', pos)
        if ev.button() == Qt.MouseButton.LeftButton:
            for index, item in self.scatterplot.image_items:
                if item.contains(item.mapFromScene(pos)):
                    self.scatterplot.selected_point = int(pos.x()), int(pos.y())
                    self.scatterplot.selected_index = index
                    self.clicked_on_point()
                    break

    def clicked_on_point(self):
        print('point/ image clicked, load on the left')
        # Get the selected point
        self.filename= self.scatterplot.get_image_path(self.scatterplot.selected_index)
        self.initialize_images( self.scatterplot.selected_point,self.filename)


def main():
    app = QApplication(sys.argv)
    with open("theme1.css","r") as file:
        app.setStyleSheet(file.read())
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
