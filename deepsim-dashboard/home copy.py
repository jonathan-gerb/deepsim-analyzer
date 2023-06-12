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
        # self.resize(1366, 768)
        # self.resize(1566, 768)
        self.resize(1565, 775)

        # Create a label to display the photo
        left_img_size=300
        self.photo_label = QLabel(self)
        self.photo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.photo_label.setFixedSize(left_img_size, left_img_size)  # Set the fixed size to 200x200

        # Create a label to display the photo
        top_img_size=150
        self.top1 = QLabel(self)
        self.top1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.top1.setFixedSize(top_img_size, top_img_size)  # Set the fixed size to 200x200

        # Create a label to display the photo
        self.top2 = QLabel(self)
        self.top2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.top2.setFixedSize(top_img_size, top_img_size)  # Set the fixed size to 200x200

        # Create a label to display the photo
        self.top3 = QLabel(self)
        self.top3.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.top3.setFixedSize(top_img_size, top_img_size)  # Set the fixed size to 200x200

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

        # load the config file 'config.ini'
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')

        data_path = self.config['main']['pkl_path']
        images_path = self.config['main']['images_path']

        num_samples = int(self.config['main']['num_samples'])
        sample_selection = str(self.config['main']['sample_selection'])
        # dataloading
        df = pd.read_pickle(data_path)
        print('df', df.keys())

        with h5py.File(images_path, "r") as hf:
            self.image_features = hf["image_features"][:]
            print(hf.keys())

        # self.image_features=  da.io.read_feature(datafile_path, img_hash, feature_name)

        if sample_selection == 'random':
            self.indices = np.random.choice(len(df), num_samples, replace=False)
        else:
            self.indices = slice(num_samples)

        self.tags = df['tags'].iloc[self.indices].values
        self.dates = df['date'].iloc[self.indices].values
        self.styles = df['style'].iloc[self.indices].values
        self.artist_names = df['artist_name'].iloc[self.indices].values
        self.artist_nationalities = df['artist_nationality'].iloc[self.indices].values
        self.points = df[['umap_x', 'umap_y']].iloc[self.indices].values
        self.image_features = self.image_features[self.indices]
        self.img_paths = df['image'].iloc[self.indices].values

        print('points[0]', self.points[0])
        # recompute the tag embedding coordinates to umap
        if bool(self.config['main']['recompute_embedding']):
            self.points = self.compute_umap(self.image_features)
        print(self.points[0])
        

        self.scatterplot = ScatterplotWidget(self.points,self.indices, self.img_paths, self.config)
        # self.scatterplot.setGeometry(QtCore.QRect(330, 20, 331, 351))
        # self.scatterplot.point_clicked.connect(self.on_canvas_click)
        # self.scatterplot.plot_widget.scene().sigMouseClicked.connect(self.scatterplot.point_clicked.emit)
        self.scatterplot.plot_widget.scene().sigMouseClicked.connect(self.on_canvas_click)
        self.dot_plot = QPushButton("Dot Scatterplot")
        self.dot_plot.clicked.connect(self.scatterplot.draw_scatterplot_dots)
        self.images_button = QPushButton("Images Scatterplot")
        self.images_button.clicked.connect(self.scatterplot.draw_scatterplot)

        

        # Create labels for the timeline
        self.date_label = QLabel("", self)
        self.artist_label = QLabel("", self)
        self.style_label = QLabel("", self)
        self.tag_label = QLabel("", self)
        # Set word wrap for the labels
        self.date_label.setWordWrap(True)
        self.artist_label.setWordWrap(True)
        self.style_label.setWordWrap(True)
        self.tag_label.setWordWrap(True)

        # Create a layout for the timeline
        img_info_layout = QGridLayout()
        img_info_layout.addWidget(self.date_label, 0, 0)
        img_info_layout.addWidget(self.artist_label, 0, 1)
        img_info_layout.addWidget(self.style_label, 0, 2)
        img_info_layout.addWidget(self.tag_label, 1, 0, 1, 3)  # Span tag_label across 3 columns
        img_info = QWidget() 
        img_info.setLayout(img_info_layout)
        img_info.setFixedWidth(300)  # Set the fixed width of the img_info widget

        self.init_id =np.random.randint(len(self.points))
        print(self.points[self.init_id], self.img_paths[self.init_id])
        self.initialize_images(self.points[self.init_id],self.img_paths[self.init_id], self.init_id)

        # Create a layout and add the label, button, input field, and timeline
        main_widget = QWidget()
        Hbox = QHBoxLayout()
        col1 = QVBoxLayout()
        col1.addWidget(self.photo_label)
        col1.addWidget(img_info)
        col1.addWidget(self.upload_button)
        col1.addWidget(self.input_text)
        col1.addWidget(self.answer_label)
        col1.addWidget(self.submit_button)

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

        # Create push buttons and connect them to the scatter plot
        button_texts = ["Semantic Similarity", "Low-Level Similarity", "High-Level Similarity",
                        "Color Similarity", "Selected Object Similarity"]
        button_layout = QHBoxLayout()
        # button_layout.setSpacing(10)  # Set spacing between buttons

        for text in button_texts:
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(lambda checked, t=text: self.trigger_scatterplot(button_texts.index(t) + 1))
            button_layout.addWidget(button)

        # Set the button layout to the buttons widget
        buttons_widget.setLayout(button_layout)

        # Set the buttons widget as the widget for the scroll area
        scroll_area.setWidget(buttons_widget)

        # Adjust the size of the scroll container to fit the buttons
        scroll_area.setMinimumHeight(buttons_widget.sizeHint().height()+18)
        scroll_area.setMaximumHeight(buttons_widget.sizeHint().height()+18)

        # Add the scroll area to the main layout
        col2.addWidget(scroll_area)


        self.timeline = HistoryTimelineWidget()
        self.timeline.populate_tree(["Item 1", "Item 2", "Item 3"])
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

        sliders = [
            {"label": "Semantic Similarity", "slider": QtWidgets.QSlider(Qt.Orientation.Horizontal)},
            {"label": "Color Similarity", "slider": QtWidgets.QSlider(Qt.Orientation.Horizontal)},
            {"label": "Selected Object Similarity", "slider": QtWidgets.QSlider(Qt.Orientation.Horizontal)},
            {"label": "Another Similarity", "slider": QtWidgets.QSlider(Qt.Orientation.Horizontal)}
        ]

        for slider in sliders:
            label = QLabel(slider["label"])
            slider_widget = slider["slider"]
            slider_widget.setMinimumWidth(40)  # Adjust the minimum width as needed
            # Create a label to display the slider value
            slider_label = QLabel("0")
            slider_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

            # Connect the slider's valueChanged signal to a lambda function that updates the label text
            slider_widget.valueChanged.connect(lambda value, label=slider_label: label.setText(str(value)))

            slider_layout.addWidget(label)
            slider_layout.addWidget(slider_widget)
            slider_layout.addWidget(slider_label)

        slider_button = QtWidgets.QPushButton('Submit similarity importance')
        slider_button.clicked.connect(lambda: self.trigger_scatterplot(10))
        slider_layout.addWidget(slider_button)
        slider_container.setLayout(slider_layout)
        col3.addWidget(slider_container)


        Hbox.addLayout(col1)
        Hbox.addLayout(col2)
        Hbox.addLayout(col3)
        main_widget.setLayout(Hbox)

        # self.setCentralWidget(main_widget)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(main_widget)
        self.setCentralWidget(scroll_area)

    def compute_umap(self, image_features):
        return umap.UMAP(n_neighbors=6, n_components=2, metric='cosine').fit_transform(image_features)

    def initialize_images(self, init_point, filename, init_id=None):
        self.display_photo(filename)
        nearest_indices = self.scatterplot.find_nearest_neighbors(
            init_point, n=3)
        print('nearest_indices', nearest_indices)
        
        nearest_images = []
        for near_idx in nearest_indices:
            nearest_images.append(self.img_paths[near_idx])
        self.display_top_photo(nearest_images)

        if init_id is None:
            print('hier when upload image')
            self.update_image_info('unk', 'unk', 'unk', 'unk')        
        else:
            print('hier when clicked in plot')
            self.update_image_info(self.dates[init_id], self.artist_names[init_id], self.styles[init_id], self.tags[init_id])

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
                new_point= self.get_point_new_img(self.filename)
                self.initialize_images(new_point,self.filename)

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
            self.scatterplot.draw_scatterplot()
        elif button_num == 2:
            print("Button 2 clicked")
            # If you create new instance, remove the existing scatterplot widget from the layout

            # Create a new scatterplot with different data or settings
            new_scatterplot = ScatterplotWidget(self.points,self.indices, self.img_paths, self.config)
            new_scatterplot.setGeometry(QtCore.QRect(330, 20, 331, 351))
            new_scatterplot.plot_widget.scene().sigMouseClicked.connect(self.on_canvas_click)

            # Replace the old scatterplot with the new one in the layout
            self.layout().replaceWidget(self.scatterplot, new_scatterplot)

            self.scatterplot.deleteLater()
            # Add the new scatterplot widget to the layout
            self.scatterplot = new_scatterplot

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
        print('on canvas click:', pos)
        if ev.button() == Qt.MouseButton.LeftButton:
            print('self.scatterplot.image_items', self.scatterplot.image_items)
            for idx, index, item in self.scatterplot.image_items:
                print('item.mapFromScene(pos)',item,  item.mapFromScene(pos))
                if item.contains(item.mapFromScene(pos)):
                    self.scatterplot.selected_point = int(pos.x()), int(pos.y())
                    self.scatterplot.selected_index = index
                    self.scatterplot.plot_index = idx
                    self.clicked_on_point()
                    break

    def clicked_on_point(self):
        print('point/ image clicked, load on the left')
        # Get the selected point
        self.filename= self.scatterplot.get_image_path(self.scatterplot.selected_index)
        self.initialize_images( self.scatterplot.selected_point,self.filename, self.scatterplot.plot_index)


def main():
    app = QApplication(sys.argv)
    with open("theme1.css","r") as file:
        app.setStyleSheet(file.read())
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
