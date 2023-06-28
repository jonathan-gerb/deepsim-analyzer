import sys
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QApplication,QWidget, QGraphicsScene, QGraphicsView, QGraphicsTextItem,QGraphicsPixmapItem, QLabel,QVBoxLayout
from pathlib import Path

import pandas as pd
import numpy as np
import math

import matplotlib.image as image
import matplotlib.pyplot as plt

class TimelineView(QGraphicsView):
    def __init__(self, scene, line_height, line_size, line_length, line_color, line_background):
        super().__init__(scene)
        self.line_height = line_height
        self.line_size = line_size
        self.line_length = line_length
        self.line_color = line_color
        self.line_background = line_background
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
    
    def drawBackground(self, painter: QPainter, rect):
        """
        Set up the layout of the widget. 
        """
        painter.fillRect(rect, self.line_background)
        pen = QPen(self.line_color)
        pen.setWidth(self.line_size)
        painter.setPen(pen)
        painter.drawLine(0, self.line_height, self.line_length, self.line_height)


class TimelineWindow(QWidget):
    def __init__(self, main_photo_key, metadata):
        super().__init__()

        self.line_size = 5
        self.line_height = 70
        self.line_length = 800 # TODO make modular with l_left_box from UI.
        self.line_color = Qt.GlobalColor.gray
        self.line_background = Qt.GlobalColor.white
        self.image_height_padding = -15 # Lower is higher in UI.
        self.image_scale = 28
        self.date_height_padding = 8 # Lower is higher in UI.
        self.date_width_padding = -6 # Lower is left in UI.
        self.font = QFont("Arial", 10)

        self.metadata = metadata
        self.scene = QGraphicsScene(self)
        view = TimelineView(self.scene, 
                            self.line_height, 
                            self.line_size, 
                            self.line_length,
                            self.line_color,
                            self.line_background)

        self.df = self.get_data()

        self.draw_timeline(main_photo_key)
        view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(view)



    def get_data(self):
        """
        Return dataframe with metadata of all images.
        """
        basepath = Path(__file__)
        data_info_path = basepath.parent.parent.parent.parent / 'data/artistic_visual_storytelling.csv'  # TODO make this pretty
        return pd.read_csv(str(data_info_path))


    def draw_timeline(self, main_photo_key):
        """
        Draw images on timeline.
        """
        self.scene.clear()
        image_names, image_years = self.get_images_paths_years(main_photo_key)
        n = len(image_years)
        x_pos = np.linspace(1, n, n) * 100

        for path, year, x in zip(image_names, image_years, x_pos):
            self.create_timeline_point(str(int(year)), x)
            item = self.create_timeline_item(path, x)
    

    def get_images_paths_years(self, main_photo_key):
        """
        Returns list of filenames and list of years of all images related to the main_photo.
        """
        attributes_before = ['prior_10_inside_style',
                             'prior_20_inside_style',
                             'prior_50_inside_style',
                             'prior_100_inside_style'
                             ]
        
        attributes_after = ['subsequent_10_inside_style',
                            'subsequent_20_inside_style',
                            'subsequent_50_inside_style',
                            'subsequent_100_inside_style'
                            ]

        main_photo_attributes = self.metadata[main_photo_key]
    
        img_paths = []
        img_years = []
        for att in attributes_before+attributes_after:
            if not pd.isna(main_photo_attributes[att]): 
                neighbour_idx = int(main_photo_attributes[att])
                img_paths.append(self.df["image"][neighbour_idx][7:]) # Slicing due to inconvienent naming convention of df.
                img_years.append(self.df["date"][neighbour_idx])
        return img_paths, img_years        
        

    def create_timeline_point(self, date, x_pos):
        """
        Plot date onto timeline.
        """
        text_item = QGraphicsTextItem(date)
        text_item.setFont(self.font)
        text_item.setPos(x_pos + self.date_width_padding, self.line_height + self.date_height_padding)
        self.scene.addItem(text_item)


    def create_timeline_item(self, image_path, xpos):
        """
        Plot image onto timeline.
        """
        basepath = Path(__file__)
        imgs_filepath = basepath.parent.parent.parent.parent / 'data/local_images/images'
        pixmap=QPixmap(str(imgs_filepath/ image_path)).scaled(self.image_scale, self.image_scale)
        item = QGraphicsPixmapItem(pixmap)
        item.setFlag(QGraphicsPixmapItem.GraphicsItemFlag.ItemIsMovable)
        item.setFlag(QGraphicsPixmapItem.GraphicsItemFlag.ItemSendsGeometryChanges)

        item.setPos(xpos, self.line_height + self.image_height_padding)
        self.scene.addItem(item)

