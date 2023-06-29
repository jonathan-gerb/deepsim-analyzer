import configparser

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PIL import Image
from PIL.ImageQt import ImageQt
from tqdm import tqdm

from PyQt6.QtCore import QPoint, QSize, Qt, pyqtSignal,QEvent,QPointF
from PyQt6.QtGui import QBrush, QImage, QPixmap, QTransform,QPen,QColor,QMouseEvent
from PyQt6.QtWidgets import (QApplication, QDialog, QGraphicsEllipseItem,
                               QGraphicsPixmapItem, QMainWindow, QPushButton,
                               QVBoxLayout, QWidget,QGraphicsRectItem,QGraphicsScene,QGraphicsView,QGraphicsSceneMouseEvent)

import matplotlib.pyplot as plt

from pyqtgraph.Point import Point
from pyqtgraph import functions as fn

from pyqtgraph import PlotDataItem
from pyqtgraph.Qt import QtGui


class ScatterplotWidget(QWidget):

    # Define a custom signal to emit when a point is clicked
    point_clicked = pyqtSignal(tuple)
    sigMouseMoved=pyqtSignal(object)
    sigMouseReleased=pyqtSignal(object)
    sigMouseClicked=pyqtSignal(object)

    get_Selected_stats =pyqtSignal(int)

    def __init__(self, points, indices, img_paths, config, plot_widget, indices_to_keep):
        super().__init__()

        self.plot_widget = plot_widget
        self.setMouseTracking(True)
        self.initialize(points, indices, img_paths, config, indices_to_keep)

        self.points_size = float(config['scatterplot']['point_size'])
        self.points_color = config['scatterplot']['points_color']
        self.selection_color = config['scatterplot']['selection_color']
        self.selection_points_size = float(config['scatterplot']['selection_point_size'])
        self.rectangle_color = config['scatterplot']['rectangle_color']
        self.rectangle_opacity = float(config['scatterplot']['rectangle_opacity'])

        
        self.plot_widget.setMouseEnabled(True, True)
        self.plot_widget.setLimits(xMin=-1000000, xMax=1000000, yMin=-1000000, yMax=1000000)
        self.plot_widget.setAspectLocked(lock=True)

        self.plot_widget.scene().mouseDoubleClickEvent = self.on_scene_mouse_double_click
        self.plot_widget.scene().mouseReleaseEvent = self.on_scene_mouse_release
        self.plot_widget.scene().mouseMoveEvent = self.on_scene_mouse_move

        self.plot_widget.setAcceptHoverEvents(True)
        self.hover_img=pg.ImageItem()
        self.draw_scatterplot_dots()


    def initialize(self, points, indices, img_paths, config, indices_to_keep):
        self.points = points
        self.indices = indices
        self.indices_to_keep = indices_to_keep
        self.config = config
        self.img_paths = img_paths
        self.mean_x = np.mean(self.points[:,0])
        self.mean_y = np.mean(self.points[:,1])
        self.start_point = None
        self.end_point = None
        self.image_items = []
        self.dot_items = []
        self.selected_index = 0
        self.selected_point = self.points[0]
        self.selected_indices = []
        self.dots_plot = True # need a seperate var for tacking trak of current plot type
        self.dots_btn_toggled = False # and the dot toggle (if toggled you want images)


    def reset_scatterplot(self, pos):
        # print('reset_scatterplot', pos)
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
        print("*mouseReleaseEvent")
        if event.button() == Qt.MouseButton.LeftButton and self.start_point is not None and self.end_point is not None:
            self.get_selection_in_rectangle()
            if self.selected_indices != []:
                print('draw dots or imgs check', len(self.selected_indices), self.dots_btn_toggled)
                if self.dots_btn_toggled or 0 < len(self.selected_indices) < 100 or len(self.indices_to_keep) < 100:
                    # images
                    self.dots_plot = False
                    self.draw_scatterplot()
                else:
                    # print('self.dots_plot', self.dots_plot ,'->', True)
                    self.dots_plot = True
                    self.draw_scatterplot_dots()

        # self.clear_selection() # ? but will also put selected_indices = [] or
        self.start_point = None
        self.end_point = None
        if self.rect is not None and self.rect in self.plot_widget.scene().items():
            self.plot_widget.scene().removeItem(self.rect)
    
    def on_scene_mouse_move(self, event):
        # print('mouseMoveEvent')
        if self.start_point is not None:
            # print("end_selection", event)
            pos = event.scenePos()
            self.end_selection(pos)

        # for hovering
        if self.dots_plot:
            pos = event.scenePos()
            points = []
            range_radius = 0.1  # Adjust the range radius as needed

            # Find the points within the range around the mouse position
            for _, _, dot_item in self.dot_items:
                item_pos = dot_item.mapFromScene(pos)
                x_data, y_data = dot_item.getData()
                for x, y in zip(x_data, y_data):
                    if abs(x - item_pos.x()) <= range_radius and abs(y - item_pos.y()) <= range_radius:
                        points.append((x, y))

            if points:
                # print('show hover and points = ', points)
                point = points[0]  # Assuming you want to handle the first point
                x, y = point

                i = np.where((self.points[:, 0] == x) & (self.points[:, 1] == y))[0][0]
                image_path = self.img_paths[i]
                image = plt.imread(image_path)
                w, h, _ = image.shape
                self.hover_img.setImage(image)
                scale = 5
                rotation = -90
                self.hover_img.setScale(scale / np.sqrt(w**2 + h**2))
                self.hover_img.setPos(x, y)
                self.hover_img.setRotation(rotation)
                
                self.hover_img.setVisible(True)
                self.plot_widget.addItem(self.hover_img)
            else:
                self.hover_img.setVisible(False)


    def start_selection(self, ev):
        print('start ev')
        if ev.button() == Qt.MouseButton.LeftButton:
            pos = ev.scenePos()
            view_coords = self.plot_widget.mapToView(pos)
            self.start_point = (view_coords.x(), view_coords.y())

    def end_selection(self, pos):
        view_coords = self.plot_widget.mapToView(pos)
        self.end_point = (view_coords.x(), view_coords.y())
        if self.start_point!= self.end_point and self.end_point is not None and self.start_point is not None:
            self.draw_selection_rectangle()

    def get_indices_nearest_neighbors(self, point):
        x, y =point
        distances = np.sqrt(np.power(self.points[:, 0] - x, 2) + np.power(self.points[:, 1] - y, 2))
        indices = np.argsort(distances)
        return indices

    def find_nearest_neighbors(self, point, n=3):
        indices = self.get_indices_nearest_neighbors(point)
        nearest_indices = indices[:n]
        return nearest_indices
    
    def clear_selection(self):
        print('clear selection, is it a switch between tabs?')
        # for switch between tabs, to get all points again
        self.selected_indices = []
        if self.rect is not None and self.rect in self.plot_widget.scene().items():
            self.plot_widget.scene().removeItem(self.rect)
        self.start_point = None
        self.end_point = None

    def draw_selection_rectangle(self):
        """Method that draws the selection rectangle on the plot"""
        x1, y1 = self.start_point[0], self.start_point[1]
        x2, y2 = self.end_point[0], self.end_point[1]

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
        self.plot_widget.addItem(self.rect)


    def get_selection_in_rectangle(self):
        x1, y1 = self.start_point[0], self.start_point[1]
        x2, y2 = self.end_point[0], self.end_point[1]

        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])
        # check is all point are in rect
        # for i, p in enumerate(self.points):
        #     print( xmin , p[0], xmax,  ymin , p[1] ,ymax)
        self.selected_indices = [i for i, p in enumerate(self.points) if xmin <= p[0] <= xmax and ymin <= p[1] <= ymax]

        if self.selected_indices != []:
            self.get_Selected_stats.emit(0) 


    def draw_scatterplot(self, reset=True):
        # from deepsim_analyzer.ds_dashboard.mainwindow import MainWindow
        print('draw_scatterplot')
        self.plot_widget.clear()

        if len(self.selected_indices) != 0:
            points = self.points[self.selected_indices]
            indices = self.selected_indices
        else:
            points = self.points
            indices = self.indices

        self.image_items = []
        new_pos = []
        print('num of points in plot', len(points))
        for i, point in tqdm(enumerate(points), desc="adding points to canvas", total=len(points)):
            x,y = point
            ith_idx= indices[i]
            if ith_idx not in self.indices_to_keep:
                continue
            
            image_path = self.img_paths[ith_idx]
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
            self.image_items.append((i, ith_idx, image_item)) 

        if reset:
            self.reset_scatterplot(np.array(new_pos))
        self.highlight_selected_point(self.selected_index)
        self.plot_widget.update()

    # TODO: this could be better, remove those out of scene instead of redraw selection
    def draw_scatterplot_dots(self,reset=True):
        print('draw_scatterplot_dots')
        self.plot_widget.clear()
        # self.plot_widget.scene().clear()
        # for idx, index, item in self.dot_items:
        #     self.plot_widget.scene().removeItem(item)

        if len(self.selected_indices) != 0:
            points = self.points[self.selected_indices]
            indices = self.selected_indices
        else:
            points = self.points
            indices = self.indices

        print('num of points in plot', len(points))
        self.dot_items = []
        for i, point in tqdm(enumerate(points), desc="adding points to canvas", total=len(points)):
            ith_idx = indices[i]
            dot_item = self.plot_widget.plot([point[0]], [point[1]], pen=None, symbolBrush=self.selection_color, symbolSize=self.selection_points_size, hover=True)
            self.dot_items.append((i, ith_idx, dot_item))

        if reset:
            self.reset_scatterplot(points)
        self.highlight_selected_point(self.selected_index)
        self.plot_widget.update()


    def highlight_selected_point(self, id):
        print('draw border, id:', id, 'len(self.image_items)', len(self.image_items))
        # if current selected point not in selected_points, there is no border but left img stays
        selected_points = self.points[self.selected_indices]
        if self.points[id] in selected_points and len(selected_points)!=0:
            print('current selected point is in selected_points')
        else:
            print('current selected point not in selected_points!')

        border_color = pg.mkPen(color='r', width=4)
        if self.dots_plot:
            for _,idx, plot_item in self.dot_items:
                if id==idx:
                    print('border drawn')
                    plot_item.setSymbolPen(border_color)  
        else:
            for _, idx, image_item in self.image_items:
                if id==idx:
                    print('border drawn')
                    image_item.setBorder(border_color)


    def remove_highlight_selected_point(self, id):
        print('rmv last border, id:', id)
        # TODO: check why and fix
        # if current selected point not in selected_points, there is no border but left img stays
        selected_points = self.points[self.selected_indices]
        if self.points[id] in selected_points and len(selected_points)!=0:
            print('current selected point is in selected_points')
        else:
            print('current selected point not in selected_points!')

        if self.dots_plot:
            for _,idx, plot_item in self.dot_items:
                if id==idx:
                    plot_item.setSymbolPen(None) 
        else:
            for _, idx, image_item in self.image_items:
                if id==idx:
                    image_item.setBorder(None)

  