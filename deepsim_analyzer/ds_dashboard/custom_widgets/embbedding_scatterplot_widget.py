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
                               QVBoxLayout, QWidget,QGraphicsRectItem,QGraphicsScene)

import matplotlib.pyplot as plt

class SelectablePixmapItem(QGraphicsPixmapItem):
    def __init__(self, pixmap):
        super().__init__(pixmap)
        self.isSelected = False

    def mousePressEvent(self, event):
        print('mousePressEvent selectable pixmap')
        if event.button() == Qt.LeftButton:
            self.isSelected = True
            for item in self.scene().items():
                print('item', item)
                if isinstance(item, SelectablePixmapItem) and item != self:
                    item.isSelected = False
        else:
            self.isSelected = False

class ScatterplotWidget(QWidget):
    # signal that emits the index of the selected points
    selected_idx = pyqtSignal(int)
    #signal that emits the current mouse position
    label = pyqtSignal(str)

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
        self.selected_point=None
        self.selected_index=None
        self.plot_inex=None
        self.selected_points = []
        self.outside_points_visible = False


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
        print("mousePressEvent")
        if event.button() == Qt.MouseButton.LeftButton:
            self.start_selection(event)
        
    def on_scene_mouse_release(self, event):
        print("mouseReleaseEvent")
        if event.button() == Qt.MouseButton.LeftButton:
            self.end_selection(event)

    def on_scene_mouse_move(self, event):
        # print('mouseMoveEvent')
        # super().mouseMoveEvent(event)
        if event.buttons() == Qt.MouseButton.RightButton:
            pos = event.scenePos()
            delta = pos - event.lastPos()
            self.plot_widget.getViewBox().translateBy(delta.x(), delta.y())

        # pos = self.plot_widget.mapToScene(QPoint(int(event.x()), int(event.y())))
        # self.label.emit(f"Current Mouse Position: {pos.x():.2f}, {pos.y():.2f}")

    def start_selection(self, ev):
        print('start ev', ev)
        if ev.button() == Qt.MouseButton.LeftButton:
            pos = ev.scenePos()

            # if self.dots_plot:
            if True:
                view_coords = self.plot_widget.mapToView(pos)
                self.start_point = (view_coords.x(), view_coords.y())
            else:
                self.start_point = (pos.x(), pos.y())

    def end_selection(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton and self.start_point is not None:
            print("end_selection" )
            pos = ev.scenePos()

            # if self.dots_plot:
            if True:
                view_coords = self.plot_widget.mapToView(pos)
                self.end_point = (view_coords.x(), view_coords.y())
            else:
                self.end_point = (pos.x(), pos.y())

            if self.start_point!= self.end_point:
                print('start pos', self.start_point)
                print('end pos', self.end_point)
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
        self.plot_widget.scene().addItem(self.rect)

        # Emit a signal with the index of the selected points
        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])

        # Get the indices of the selected points
        # indices = [i for i, p in enumerate(self.points) if xmin <= p[0] <= xmax and ymin <= p[1] <= ymax]
        
        # Update the selected points
        for i, p in enumerate(self.points):
            print( xmin , p[0], xmax,  ymin , p[1] ,ymax)
        self.selected_points = np.array([p for p in self.points if xmin <= p[0] <= xmax and ymin <= p[1] <= ymax])
        print('self.selected_points', len(self.selected_points))
        
        # Redraw the scatterplot
        if self.dots_plot:
            self.draw_scatterplot_dots()
        else:
            self.draw_scatterplot()


    def draw_scatterplot_dots(self):
        print('draw_scatterplot_dots')
        self.plot_widget.clear()

        if self.selected_points!=[]:
            print('draw selected points')
            points= self.selected_points
        else:
            points=self.points

        # self.plot_widget.plot(points[:, 0], points[:, 1], pen=None, symbolBrush=self.points_color, symbolSize=self.points_size)
        print(len(points))
        for point in points:
            if len(points)<5:
                print(' only pos ', point[0], point[1])
            self.plot_widget.plot([point[0]], [point[1]], pen=None, symbolBrush=self.selection_color, symbolSize=self.selection_points_size)

        self.reset_scatterplot(points)
        self.plot_widget.update()
        

    # def draw_scatterplot(self):
    #     print('draw_scatterplot')
    #     self.plot_widget.clear()

    #     if self.selected_points!=[]:
    #         points= self.selected_points
    #     else:
    #         points=self.points

    #     self.image_items = []
    #     new_pos = []
    #     image_size = 24
    #     # min, max = np.min(self.points), np.max(self.points)
    #     scale = image_size * 8
    #     for i, point in tqdm(enumerate(points), desc="placing images on plot", total=len(points)):
    #         x, y = point
    #         x, y = x * scale, y * scale
    #         image_path = self.img_paths[i]
    #         pixmap = QPixmap(image_path)

    #         # Flip the pixmap vertically
    #         pixmap = pixmap.transformed(QTransform().scale(1, -1))

    #         # Resize the pixmap to a smaller size
    #         scaled_pixmap = pixmap.scaled(QSize(image_size, image_size), Qt.AspectRatioMode.KeepAspectRatio)

    #         # pixmap_item = QGraphicsPixmapItem(scaled_pixmap)
    #         pixmap_item = SelectablePixmapItem(scaled_pixmap)
    #         x_ = x - (pixmap.width() / 2)
    #         y_ = y + (pixmap.height() / 2)

    #         # print('img pos', (x_, y_))
    #         pixmap_item.setPos(x_, y_)
    #         new_pos.append((x_, y_))
            
    #         # pixmap_item.setZValue(10)
    #         # highlight selected point
    #         pixmap_item.mousePressEvent=self.handle_selection_changed

    #         self.plot_widget.addItem(pixmap_item)
    #         self.image_items.append((i, self.indices[i], pixmap_item)) 

    #     self.reset_scatterplot(np.array(new_pos))
    #     self.plot_widget.update()

    def draw_scatterplot(self):
        print('draw_scatterplot')
        self.plot_widget.clear()

        if self.selected_points!=[]:
            points= self.selected_points
        else:
            points=self.points

        self.image_items = []
        new_pos=[]
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
            
            # Make it clickable.
            self.image_items.append((i, self.indices[i], image_item)) 


        # self.reset_scatterplot(np.array(new_pos))
        self.plot_widget.update()

 
    

    def handle_selection_changed(self, idx):
        print('handle_selection_changed')
        for _,_,pixmap_item in self.image_items:
            print('pixmap_item.isSelected', pixmap_item.isSelected)
            if pixmap_item.isSelected:
                pen = QPen(QColor(255, 0, 0))  
                pen.setWidth(2) 
                # pixmap_item.setPen(pen)
                pixmap_item.setOpacity(0.5)
            else:
                pixmap_item.setOpacity(1)

