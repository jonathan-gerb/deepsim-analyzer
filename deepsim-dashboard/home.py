import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QGridLayout, QLineEdit
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

class MainWindow(QMainWindow):
    def compute_umap(self, image_features):
        return umap.UMAP(n_neighbors=6, n_components=2, metric='cosine').fit_transform(image_features)

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Image Viewer")
        self.resize(1041, 564)

        # Create a label to display the photo
        self.photo_label = QLabel(self)
        self.photo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.photo_label.setFixedSize(200, 200)  # Set the fixed size to 200x200

        # Create a label to display the photo
        self.top1 = QLabel(self)
        self.top1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.top1.setFixedSize(100, 100)  # Set the fixed size to 200x200

        # Create a label to display the photo
        self.top2 = QLabel(self)
        self.top2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.top2.setFixedSize(100, 100)  # Set the fixed size to 200x200

        # Create a label to display the photo
        self.top3 = QLabel(self)
        self.top3.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.top3.setFixedSize(100, 100)  # Set the fixed size to 200x200

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

        # select num_samples samples between
        if sample_selection == 'random':
            random_indices = np.random.choice(
                len(df), num_samples, replace=False)
            tags = df['tags'].iloc[random_indices].values
            points = df[['umap_x', 'umap_y']].iloc[random_indices].values
            image_features = image_features[random_indices]
            img_paths = df['image'].iloc[random_indices].values

        if sample_selection == 'first':
            tags = df['tags'].iloc[:num_samples].values
            points = df[['umap_x', 'umap_y']].iloc[:num_samples].values
            image_features = image_features[:num_samples]
            img_paths = df['image'].iloc[:num_samples].values

        # recompute the tag embedding coordinates to umap
        if bool(config['main']['recompute_embedding']):
            points = self.compute_umap(image_features)

        self.config=config
        self.points=points
        self.scatterplot = ScatterplotWidget(points, config)
        self.scatterplot.setGeometry(QtCore.QRect(330, 20, 331, 351))
        self.wordcloud = WordcloudWidget(tags, config)

        # Connect the point_clicked signal to clicked_on_point method
        self.scatterplot.point_clicked.connect(self.clicked_on_point)

        self.timeline = HistoryTimelineWidget()

        # Create a layout and add the label, button, input field, and timeline
        main_widget = QWidget()
        Hbox = QHBoxLayout()
        col1 = QVBoxLayout()
        col1.addWidget(self.photo_label)
        col1.addLayout(img_info)
        col1.addWidget(self.input_text)
        col1.addWidget(self.upload_button)
        col1.addWidget(self.answer_label)
        col1.addWidget(self.submit_button)

        col2 = QVBoxLayout()
        col2.addWidget(self.scatterplot)

        # Create push buttons and connect them to the scatter plot
        self.pushButton = QtWidgets.QPushButton("Button 1")
        self.pushButton.clicked.connect(lambda: self.trigger_scatterplot(1))

        self.pushButton_2 = QtWidgets.QPushButton("Button 2")
        self.pushButton_2.clicked.connect(lambda: self.trigger_scatterplot(2))

        self.pushButton_3 = QtWidgets.QPushButton("Button 3")
        self.pushButton_3.clicked.connect(lambda: self.trigger_scatterplot(3))

        self.pushButton_4 = QtWidgets.QPushButton("Button 4")
        self.pushButton_4.clicked.connect(lambda: self.trigger_scatterplot(4))

        self.pushButton_5 = QtWidgets.QPushButton("Button 5")
        self.pushButton_5.clicked.connect(lambda: self.trigger_scatterplot(5))

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.pushButton)
        button_layout.addWidget(self.pushButton_2)
        button_layout.addWidget(self.pushButton_3)
        button_layout.addWidget(self.pushButton_4)
        button_layout.addWidget(self.pushButton_5)

        col2.addLayout(button_layout)
        col2.addWidget(self.timeline)

        col3 = QVBoxLayout()
        self.top = QHBoxLayout()
        self.top.addWidget(self.top1)
        self.top.addWidget(self.top2)
        self.top.addWidget(self.top3)
        col3.addLayout(self.top)

        # Add the sliders to col3
        self.horizontalSlider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.horizontalSlider.setGeometry(QRect(50, 50, 160, 18))
        self.horizontalSlider.setObjectName("horizontalSlider")
        col3.addWidget(self.horizontalSlider)

        self.horizontalSlider_2 = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.horizontalSlider_2.setGeometry(QRect(50, 100, 160, 18))
        self.horizontalSlider_2.setObjectName("horizontalSlider_2")
        col3.addWidget(self.horizontalSlider_2)

        self.horizontalSlider_3 = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.horizontalSlider_3.setGeometry(QRect(50, 140, 160, 18))
        self.horizontalSlider_3.setObjectName("horizontalSlider_3")
        col3.addWidget(self.horizontalSlider_3)

        self.horizontalSlider_4 = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.horizontalSlider_4.setGeometry(QRect(50, 190, 160, 18))
        self.horizontalSlider_4.setObjectName("horizontalSlider_4")
        col3.addWidget(self.horizontalSlider_4)


        Hbox.addLayout(col1)
        Hbox.addLayout(col2)
        Hbox.addLayout(col3)
        main_widget.setLayout(Hbox)

        self.setCentralWidget(main_widget)

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

    # def display_top_photo(self, filenames):
    #     top_photos = [self.top1, self.top2, self.top3]
    #     for i, filename in enumerate(filenames):
    #         pixmap = QPixmap(filename)
    #         top_photos[i].setPixmap(pixmap.scaledToWidth(top_photos[i].width()))


    def display_top_photo(self, imgs):
        top_photos = [self.top1, self.top2, self.top3]
        for i, img in enumerate(imgs):
            pixmap = QPixmap(img)
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
            self.scatterplot = ScatterplotWidget(self.points, self.config)
            self.scatterplot.setGeometry(QtCore.QRect(330, 20, 331, 351))
        elif button_num == 2:
            print("Button 2 clicked")
            # Create a new scatterplot with different data or settings
            self.scatterplot = ScatterplotWidget(self.points, self.config)
            self.scatterplot.setGeometry(QtCore.QRect(330, 20, 331, 351))
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
    
    def clicked_on_point():
        # Get the selected point
        selected_point = self.scatterplot.get_selected_point()

        # Find the 3 nearest neighbors
        nearest_indices = self.scatterplot.find_nearest_neighbors(
            selected_point, n=3)
        
        nearest_embeddings = self.get_embeddings_from_nearest_neighbors(nearest_indices)
        
        # Assuming 'width' and 'height' represent the desired dimensions of the image
        width = 100
        height = 100
        # Generate images from nearest neighbor embeddings
        nearest_images = []
        for embedding in nearest_embeddings:
            image = self.scatterplot.vector_to_image(embedding, width, height)
            nearest_images.append(image)
        self.display_top_photo(nearest_images)

        # Get the paths of the nearest neighbor images
        # nearest_image_paths = self.scatterplot.get_image_paths(nearest_indices)
        # Display the nearest neighbor images in top1, top2, and top3
        # self.display_top_photo(nearest_image_paths)

        # Update the layout to reflect the new scatterplot
        self.update_layout()

def main():
    app = QApplication(sys.argv)

    # with open("theme1.css","r") as file:
    #     app.setStyleSheet(file.read())
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
