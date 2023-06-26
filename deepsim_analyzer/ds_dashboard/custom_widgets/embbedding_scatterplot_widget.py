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
# from pyqtgraph.graphicsItems.ScatterPlotItem import HoverEvent



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

        self.plot_widget.scene().mouseDoubleClickEvent = self.on_scene_mouse_double_click
        self.plot_widget.scene().mouseReleaseEvent = self.on_scene_mouse_release
        # self.plot_widget.scene().mouseMoveEvent = self.on_scene_mouse_move
        
        # self.plot_widget.scene().sigMouseMoved.connect(lambda event: self.on_scene_mouse_move(QMouseEvent(event)))
        # self.plot_widget.scene().sigMouseMoved.connect(lambda event: self.on_scene_mouse_move(event))
        # self.plot_widget.scene().sigMouseMoved.connect(lambda pos: self.on_scene_mouse_move(QGraphicsSceneMouseEvent(QEvent.Type.GraphicsSceneMouseMove, pos)))
        # self.plot_widget.scene().sigMouseMoved.connect(lambda pos: self.on_scene_mouse_move(QMouseEvent(QMouseEvent.Type.MouseMove, pos, QPointF(), QPointF(), Qt.MouseButton.NoButton, Qt.MouseButton(), Qt.KeyboardModifier(), None)))
        # self.plot_widget.scene().sigMouseMoved.connect(lambda event: self.on_scene_mouse_move(QMouseEvent(QMouseEvent.Type.MouseMove, event.pos(), QPointF(), Qt.MouseButton.NoButton, Qt.MouseButtons(), Qt.KeyboardModifiers())))

        self.plot_widget.scene().sigMouseMoved.connect(self.on_scene_mouse_move_with_QPointF)
        # self.plot_widget.scene().sigMouseClicked.connect(lambda event: self.on_canvas_click(event))
         
        self.plot_widget.setAcceptHoverEvents(True)
        self.hover_img=pg.ImageItem()
        self.plot_widget.addItem(self.hover_img)

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
        self.selected_index=0
        self.selected_point=self.points[0]
        self.selected_idx.connect(self.highlight_selected_point)
        self.plot_inex=None
        self.selected_points = []
        self.selected_indices=[]
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
        print("*mouseReleaseEvent")
        if event.button() == Qt.MouseButton.LeftButton and self.start_point is not None and self.end_point is not None:
            self.get_selection_in_rectangle()
            if self.dots_plot:
                self.draw_scatterplot_dots()
            else:
                self.draw_scatterplot()

        # self.selected_idx.emit(self.selected_index)
        # self.clear_selection() # ? but will also put selected_points = [] or
        self.start_point=None
        self.end_point=None
        if self.rect is not None and self.rect in self.plot_widget.scene().items():
            self.plot_widget.scene().removeItem(self.rect)

        QGraphicsScene.mouseReleaseEvent(self.plot_widget.scene(), event)
        view = self.plot_widget.getViewBox()
        view.mouseReleaseEvent(event)

    def on_scene_mouse_move_with_QPointF(self, event):
        # print('mouseMoveEvent')
        if self.start_point is not None:
            pos =event
            self.end_selection(pos)
        # QGraphicsScene.mouseMoveEvent(self.plot_widget.scene(), event)

        if self.dots_plot:
            pos = event
            points = []
            range_radius = 0.1  # Adjust the range radius as needed

            # Find the points within the range around the mouse position
            for i, plot_data_item in self.plot_data_items:
                item_pos = plot_data_item.mapFromScene(pos)
                # print('pos', pos, 'item_pos', item_pos)
                x_data, y_data = plot_data_item.getData()
                # print(plot_data_item, x_data, y_data)
                for x, y in zip(x_data, y_data):
                    if abs(x - item_pos.x()) <= range_radius and abs(y - item_pos.y()) <= range_radius:
                        points.append((x, y))
            
            if points:
                print('show hover and points = ', points)
                # Handle the found points
                point = points[0]  # Assuming you want to handle the first point
                x, y = point
                
                print(self.points[:5], point)
                i =np.where((self.points[:, 0] == x) & (self.points[:, 1] == y))[0][0]
                image_path = self.img_paths[i]
                image = plt.imread(image_path)
                w, h, _ = image.shape
                self.hover_img.setImage(image)
                # Adjust image
                scale = 0.3
                rotation = -90
                self.hover_img.setScale(scale / np.sqrt(w**2 + h**2))
                self.hover_img.setPos(x, y)
                self.hover_img.setRotation(rotation)
                self.hover_img.show()
            else:
                self.hover_img.hide()

    
    def on_scene_mouse_move(self, event):
        # print('mouseMoveEvent')
        if self.start_point is not None:
            print("end_selection", event)
            pos = event.scenePos()
            self.end_selection(pos)
        # QGraphicsScene.mouseMoveEvent(self.plot_widget.scene(), event)
        # QGraphicsView.mouseMoveEvent(self.plot_widget.view(), event)

        # view = self.plot_widget.getViewBox()
        # view.mouseMoveEvent(event)

        if self.dots_plot:
            pos = event
            points = []
            range_radius = 0.1  # Adjust the range radius as needed

            # Find the points within the range around the mouse position
            for i, plot_data_item in self.plot_data_items:
                print(plot_data_item)
                x_data, y_data = plot_data_item.getData()
                for x, y in zip(x_data, y_data):
                    if abs(x - pos.x()) <= range_radius and abs(y - pos.y()) <= range_radius:
                        points.append((x, y))

            if points:
                print('show hover')
                # Handle the found points
                point = points[0]  # Assuming you want to handle the first point
                x, y = point
                
                # print(self.points[:5], point)
                i =np.where((self.points[:, 0] == x) & (self.points[:, 1] == y))[0][0]
                image_path = self.img_paths[i]
                image = plt.imread(image_path)
                w, h, _ = image.shape
                self.hover_img.setImage(image)
                # Adjust image
                scale = 0.3
                rotation = -90
                self.hover_img.setScale(scale / np.sqrt(w**2 + h**2))
                self.hover_img.setPos(x, y)
                self.hover_img.setRotation(rotation)
                self.hover_img.show()
            else:
                self.hover_img.hide()


    def start_selection(self, ev):
        print('start ev', ev)
        if ev.button() == Qt.MouseButton.LeftButton:
            pos = ev.scenePos()
            view_coords = self.plot_widget.mapToView(pos)
            self.start_point = (view_coords.x(), view_coords.y())

    # def end_selection(self, ev):
    def end_selection(self, pos):
        # if ev.button() == Qt.MouseButton.LeftButton and self.start_point is not None:
        # print("end_selection", ev)
        # pos = ev.scenePos()
        view_coords = self.plot_widget.mapToView(pos)
        self.end_point = (view_coords.x(), view_coords.y())
        if self.start_point!= self.end_point and self.end_point is not None and self.start_point is not None:
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
        self.selected_indices=[]
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
        self.selected_indices = np.array([i for i, p in enumerate(self.points) if xmin <= p[0] <= xmax and ymin <= p[1] <= ymax])
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
        # self.plot_widget.scene().clear()
        # for idx, index, item in self.image_items:
        #     self.plot_widget.scene().removeItem(item)

        # self.plot_data_items= self.plot_widget.plot(points[:, 0], points[:, 1], pen=None, symbolBrush=self.points_color, symbolSize=self.points_size)

        also_show_not_selected=False
        if also_show_not_selected:
            self.plot_data_items = []
            self.plot_data_items_not_selected = []
            for i, point in self.points:
                if point in self.selected_indices:
                    print('point in selection', point)
                    plot_data_item= self.plot_widget.plot([point[0]], [point[1]], pen=None, symbolBrush=self.selection_color, symbolSize=self.selection_points_size)
                    # plot_data_item.setPos(point[0], point[1])
                    self.plot_data_items.append((i,plot_data_item))
                else:
                    plot_data_item= self.plot_widget.plot([point[0]], [point[1]], pen=None, symbolBrush=self.points_color, symbolSize=self.points_size)
                    # plot_data_item.setPos(point[0], point[1])
                    self.plot_data_items_not_selected.append((i,plot_data_item))
        else:
            if self.selected_points!=[]:
                print('draw selected points')
                # print(self.selected_points)
                points= self.selected_points
            else:
                points=self.points

            print(len(points),points[0] )
            self.plot_data_items = []
            for i, point in enumerate(points):
                plot_data_item= self.plot_widget.plot([point[0]], [point[1]], pen=None, symbolBrush=self.selection_color, symbolSize=self.selection_points_size)
                # plot_data_item.setPos(point[0], point[1])
                self.plot_data_items.append((i, plot_data_item))

        if reset:
            self.reset_scatterplot(points)
        self.plot_widget.update()



    def highlight_selected_point(self, id):
        print('draw border, id:', id)

        # if current selected point not in self.selected_points, there is no border but left img stays
        if self.points[id] in self.selected_points:
            print('current selected point not in self.selected_points')

        if self.dots_plot:
            for i, point in enumerate(self.points):
                if i == id:
                    border_color = pg.mkPen(color='r', width=4)
                    self.plot_widget.plot([point[0]], [point[1]], pen=None, symbolBrush=self.points_color, symbolSize=self.points_size, symbolPen=border_color)
                # else:
                #     # Check if the point has a red border and remove it
                # #     if self.plot_widget.items(i).opts['symbolPen'] is not None and self.plot_widget.items(i).opts['symbolPen'].color().name() == 'r':
                #     if self.plot_widget.itemAtIndex(i).opts['symbolPen'] is not None and self.plot_widget.itemAtIndex(i).opts['symbolPen'].color().name() == 'r':
                #         self.plot_widget.items(i).setSymbolPen(None)
                    
        else:
            print('image_items', len(self.image_items))
            for _, _, image_item in self.image_items:
                # Remove border from all image items
                image_item.setBorder(None)
            
            _,_,image_item=self.image_items[id]
            border_color = pg.mkPen(color='r', width=4)
            image_item.setBorder(border_color)

        

        
    # default panning fn
    # def mouseMoveEvent(self, ev):
    #     lpos = ev.position() if hasattr(ev, 'position') else ev.localPos()
    #     if self.lastMousePos is None:
    #         self.lastMousePos = lpos
    #     delta = Point(lpos - self.lastMousePos)
    #     self.lastMousePos = lpos

    #     super().mouseMoveEvent(ev)
    #     if not self.mouseEnabled:
    #         return
    #     self.sigSceneMouseMoved.emit(self.mapToScene(lpos))
            
    #     if self.clickAccepted:  ## Ignore event if an item in the scene has already claimed it.
    #         return
        
    #     if ev.buttons() == Qt.MouseButton.RightButton:
    #         delta = Point(fn.clip_scalar(delta[0], -50, 50), fn.clip_scalar(-delta[1], -50, 50))
    #         scale = 1.01 ** delta
    #         self.scale(scale[0], scale[1], center=self.mapToScene(self.mousePressPos))
    #         self.sigDeviceRangeChanged.emit(self, self.range)

    #     elif ev.buttons() in [Qt.MouseButton.MiddleButton, Qt.MouseButton.LeftButton]:  ## Allow panning by left or mid button.
    #         px = self.pixelSize()
    #         tr = -delta * px
            
    #         self.translate(tr[0], tr[1])
    #         self.sigDeviceRangeChanged.emit(self, self.range)