import configparser

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PIL import Image
from PIL.ImageQt import ImageQt
from tqdm import tqdm

from PyQt6.QtCore import QPoint, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QImage, QPixmap, QTransform,QPen,QColor
from PyQt6.QtWidgets import (QApplication, QDialog, QGraphicsEllipseItem,
                               QGraphicsPixmapItem, QMainWindow, QPushButton,
                               QVBoxLayout, QWidget,QGraphicsRectItem,QGraphicsScene,QGraphicsView)

import matplotlib.pyplot as plt


class ScatterplotWidget(QWidget):
    # signal that emits the index of the selected points
    selected_idx = pyqtSignal(int)

    # Define a custom signal to emit when a point is clicked
    point_clicked = pyqtSignal(tuple)
    sigMouseMoved=pyqtSignal(object)
    sigMouseReleased=pyqtSignal(object)

    def __init__(self, points, indices, img_paths, config, plot_widget):
        super().__init__()

        self.plot_widget = plot_widget
        self.setMouseTracking(True)
        self.initialize(points, indices,img_paths, config)

        self.points_size = float(config['scatterplot']['point_size'])
        self.points_color = config['scatterplot']['points_color']
        self.selection_color = config['scatterplot']['selection_color']
        self.selection_points_size = float(config['scatterplot']['selection_point_size'])
        self.rectangle_color = config['scatterplot']['rectangle_color']
        self.rectangle_opacity = float(config['scatterplot']['rectangle_opacity'])

        
        self.plot_widget.setMouseEnabled(True, True)
        # self.plot_widget.setLimits(xMin=-np.inf, xMax=np.inf, yMin=-np.inf, yMax=np.inf)
        self.plot_widget.setLimits(xMin=-1000000, xMax=1000000, yMin=-1000000, yMax=1000000)
        self.plot_widget.setAspectLocked(lock=True)

        self.plot_widget.scene().mouseReleaseEvent = self.on_scene_mouse_release
        self.plot_widget.scene().mouseMoveEvent = self.on_scene_mouse_move
        self.plot_widget.scene().mouseDoubleClickEvent = self.on_scene_mouse_double_click
        
        self.draw_scatterplot()

    def initialize(self, points, indices,img_paths, config):
        self.points = points
        self.indices=indices
        self.config=config
        self.img_paths=img_paths
        self.mean_x = np.mean(self.points[:,0])
        self.mean_y = np.mean(self.points[:,1])

        self.start_point = None
        self.end_point = None
        self.image_items = []
        self.selected_index=None
        # self.selected_point=self.points[0]
        self.selected_idx.connect(self.highlight_selected_point)
        self.plot_inex=None
        self.selected_points = []
        self.outside_points_visible = False

        # We now toggle in setup to insure images are plotted first
        self.dots_plot=False


    def reset_scatterplot(self, pos):
        # Get the range of x and y values in the scatterplot
        x_min = np.min(pos[:, 0])
        x_max = np.max(pos[:, 0])
        y_min = np.min(pos[:, 1])
        y_max = np.max(pos[:, 1])

        # Calculate the range of x and y values with a buffer
        x_buffer = (x_max - x_min) * 0.1  # 10% buffer
        y_buffer = (y_max - y_min) * 0.1  # 10% buffer
        x_range = (x_min - x_buffer, x_max + x_buffer)
        y_range = (y_min - y_buffer, y_max + y_buffer)

        self.plot_widget.setRange(xRange=x_range, yRange=y_range)


    def on_scene_mouse_double_click(self, event):
        print("mouseDBPressEvent")
        if event.button() == Qt.MouseButton.LeftButton:
            self.start_selection(event)
        
    def on_scene_mouse_release(self, event):
        print("mouseReleaseEvent")
        if event.button() == Qt.MouseButton.LeftButton and self.start_point is not None:
            self.get_selection_in_rectangle()
            if self.dots_plot:
                self.draw_scatterplot_dots()
            else:
                self.draw_scatterplot()

        self.clear_selection() # ? but will also put selected_points = [] or
        # self.start_point=None
        # self.end_point=None
        # if self.rect is not None and self.rect in self.plot_widget.scene().items():
        #     self.plot_widget.scene().removeItem(self.rect)

    def on_scene_mouse_move(self, event):
        # print('mouseMoveEvent')
        # super().mouseMoveEvent(event)
        # QGraphicsScene.mouseMoveEvent(self, event)
        
        if self.start_point is not None:
            self.end_selection(event)

    def start_selection(self, ev):
        print('start ev', ev)
        if ev.button() == Qt.MouseButton.LeftButton:
            pos = ev.scenePos()
            view_coords = self.plot_widget.mapToView(pos)
            self.start_point = (view_coords.x(), view_coords.y())

    def end_selection(self, ev):
        # if ev.button() == Qt.MouseButton.LeftButton and self.start_point is not None:
        # print("end_selection" )
        pos = ev.scenePos()
        view_coords = self.plot_widget.mapToView(pos)
        self.end_point = (view_coords.x(), view_coords.y())
        if self.start_point!= self.end_point:
            # print('start pos', self.start_point)
            # print('end pos', self.end_point)
            self.draw_selection_rectangle()


    def get_indices_nearest_neighbors(self, point):
        print('point', point)
        x, y =point
        distances = np.sqrt(np.power(self.points[:, 0] - x, 2) + np.power(self.points[:, 1] - y, 2))
        indices = np.argsort(distances)
        return indices

    def find_nearest_neighbors(self, point, n=3):
        indices = self.get_indices_nearest_neighbors(point)
        nearest_indices = indices[:n]
        return nearest_indices
    
    def clear_selection(self):
        print('clear selection')
        self.selected_points = []
        if self.rect is not None and self.rect in self.plot_widget.scene().items():
            self.plot_widget.scene().removeItem(self.rect)
        self.start_point=None
        self.end_point=None

    def draw_selection_rectangle(self):
        """Method that draws the selection rectangle on the plot"""
        # Get coordinates of the selection rectangle

        # print('draw_selection_rectangle')
        x1, y1 = self.start_point[0], self.start_point[1]
        x2, y2 = self.end_point[0], self.end_point[1]

        # Calculate the position and size of the rectangle
        x = min(x1, x2)
        y = min(y1, y2)
        w = abs(x1 - x2)
        h = abs(y1 - y2)

        # Calculate the position and size of the rectangle
        x = min(x1, x2)
        y = min(y1, y2)
        w = abs(x1 - x2)
        h = abs(y1 - y2)

        # Remove the old rectangle if it exists
        if self.rect is not None and self.rect in self.plot_widget.scene().items():
            self.plot_widget.scene().removeItem(self.rect)

        # Add the new rectangle to the plot
        self.rect = QGraphicsRectItem(x, y, w, h)
        self.rect.setPen(pg.mkPen(self.rectangle_color))
        color = pg.mkColor(self.rectangle_color)
        color.setAlphaF(self.rectangle_opacity)
        self.rect.setBrush(pg.mkBrush(color))
        self.rect.setZValue(10)
        # self.plot_widget.scene().addItem(self.rect)
        self.plot_widget.addItem(self.rect)


    def get_selection_in_rectangle(self):
        x1, y1 = self.start_point[0], self.start_point[1]
        x2, y2 = self.end_point[0], self.end_point[1]

        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])
        # Update the selected points
        # for i, p in enumerate(self.points):
        #     print( xmin , p[0], xmax,  ymin , p[1] ,ymax)
        self.selected_points = np.array([p for p in self.points if xmin <= p[0] <= xmax and ymin <= p[1] <= ymax])
        # print('self.selected_points', len(self.selected_points))
        

    def draw_scatterplot(self,reset=True) :
        print('draw_scatterplot')
        self.plot_widget.clear()
        if self.selected_points!=[]:
            points= self.selected_points
        else:
            points=self.points

        self.image_items = []
        new_pos=[]
        print(len(points))
        for i, point in enumerate(points):
            x,y = point
            # Read in image
            image_path = self.img_paths[i]
            image = plt.imread(image_path)
            # TODO: change resolution
            w, h, _ = image.shape
            # Create image item
            image_item = pg.ImageItem()
            image_item.setImage(image)
            # Adjust image
            scale = 0.3
            rotation = -90
            image_item.setScale(scale / np.sqrt(w**2 + h**2))
            image_item.setPos(x, y)
            image_item.setRotation(rotation)
            new_pos.append((x,y))

            # Add to plot
            self.plot_widget.addItem(image_item)
            self.image_items.append((i, self.indices[i], image_item)) 

        if reset:
            self.reset_scatterplot(np.array(new_pos))
        self.plot_widget.update()

    def draw_scatterplot_dots(self,reset=True):
        print('draw_scatterplot_dots')
        self.plot_widget.clear()
        if self.selected_points!=[]:
            print('draw selected points')
            points= self.selected_points
        else:
            points=self.points

        self.plot_widget.plot(points[:, 0], points[:, 1], pen=None, symbolBrush=self.points_color, symbolSize=self.points_size)
        print(len(points))
        also_show_not_selected=True
        if also_show_not_selected:
            for point in self.points:
                self.plot_widget.plot([point[0]], [point[1]], pen=None, symbolBrush=self.selection_color, symbolSize=self.selection_points_size)

        if reset:
            self.reset_scatterplot(points)
        self.plot_widget.update()

    def highlight_selected_point(self, id):
        print('draw border, id:', id)

        if self.dots_plot:
            for i, point in enumerate(self.points):
                if i == id:
                    border_color = pg.mkPen(color='r', width=4)
                    self.plot_widget.plot([point[0]], [point[1]], pen=None, symbolBrush=self.points_color, symbolSize=self.points_size, symbolPen=border_color)
                else:
                    self.plot_widget.plot([point[0]], [point[1]], pen=None, symbolBrush=self.selection_color, symbolSize=self.selection_points_size)
        else:
            for _, _, image_item in self.image_items:
                # Remove border from all image items
                image_item.setBorder(None)

            _,_,image_item=self.image_items[id]
            border_color = pg.mkPen(color='r', width=4)
            image_item.setBorder(border_color)
    
