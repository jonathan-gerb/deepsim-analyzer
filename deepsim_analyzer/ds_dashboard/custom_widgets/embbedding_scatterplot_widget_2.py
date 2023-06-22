from PyQt6.QtWidgets import QVBoxLayout, QWidget
from PyQt6.QtCore import pyqtSignal, QPoint
import PyQt6.QtGui as QtGui

import numpy as np
import pyqtgraph as pg

import matplotlib.pyplot as plt

# from PIL import Image
# from PIL.ImageQt import ImageQt

# from PyQt6.QtGui import QPixmap, QImage, QBrush,QTransform
# from PyQt6.QtWidgets import QGraphicsPixmapItem, QGraphicsEllipseItem
# import pandas as pd
# import configparser

# from ..home import DeepSimDashboard

# import deepsim_analyzer as da

class ScatterplotWidget(QWidget):
    selected_idx = pyqtSignal(list)
    label = pyqtSignal(str)

    # Define a custom signal to emit when a point is clicked
    point_clicked = pyqtSignal(tuple)

    def __init__(self, points, indices,img_paths, config):
        super().__init__()

        # self.window = MainWindow()

        self.points = points
        self.indices=indices
        self.config=config
        self.img_paths=img_paths
        self.mean_x = np.mean(self.points[:,0])
        self.mean_y = np.mean(self.points[:,1])

        self.points_size = float(config['scatterplot']['point_size'])
        self.points_color = config['scatterplot']['points_color']
        self.selection_color = config['scatterplot']['selection_color']
        self.selection_points_size = float(config['scatterplot']['selection_point_size'])
        self.rectangle_color = config['scatterplot']['rectangle_color']
        self.rectangle_opacity = float(config['scatterplot']['rectangle_opacity'])

        self.start_point = None
        self.end_point = None

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setMouseEnabled(True, True)
        self.plot_widget.setLimits(xMin=-np.inf, xMax=np.inf, yMin=-np.inf, yMax=np.inf)
        self.plot_widget.setAspectLocked(lock=True)

        # self.clear_button = QPushButton("Clear Selection")
        # self.clear_button.clicked.connect(self.clear_selection)

        self.image_items = []
        
        layout = QVBoxLayout(self)
        layout.addWidget(self.plot_widget)
        # layout.addWidget(self.clear_button)

        self.selected_point=None
        self.selected_index=None
        self.plot_index=None


        # self.selected_points = []
        self.outside_points_visible = False

        # self.plot_widget.scene().sigMouseClicked.connect(self.on_canvas_click)
        self.plot_widget.scene().sigMouseMoved.connect(self.on_mouse_move)

        self.draw_scatterplot_images()
        # self.resize_widget()

    def resize_widget(self):
        size = min(self.width(), self.height())
        self.setFixedSize(size, size)
        self.plot_widget.setFixedSize(size, size)


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


    def on_mouse_move(self, ev):
        # pos = self.plot_widget.mapToScene(QPoint(ev.x(), ev.y()))
        pos = self.plot_widget.mapToScene(QPoint(int(ev.x()), int(ev.y())))
        self.label.emit(f"Current Mouse Position: {pos.x():.2f}, {pos.y():.2f}")
    

    def clear_selection(self):
        self.selected_idx.emit([])
        self.selected_points = []
        self.plot_widget.clear()
        self.plot_widget.plot(self.points[:, 0], self.points[:, 1], pen=None, symbolBrush=self.points_color, symbolSize=self.points_size)

    def draw_selection_rectangle(self):
        x1, y1 = self.start_point[0], self.start_point[1]
        x2, y2 = self.end_point[0], self.end_point[1]
        x = min(x1, x2)
        y = min(y1, y2)
        w = abs(x1 - x2)
        h = abs(y1 - y2)

        self.plot_widget.scene().removeItem(self.rect)
        self.rect = pg.QtGui.QGraphicsRectItem(x, y, w, h)
        self.rect.setPen(pg.mkPen(self.rectangle_color))
        self.rect.setBrush(pg.mkColor(self.rectangle_color + '%f' % self.rectangle_opacity))
        self.plot_widget.scene().addItem(self.rect)

        self.plot_widget.update()

        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])

        indices = [i for i, p in enumerate(self.points) if xmin <= p[0] <= xmax and ymin <= p[1] <= ymax]

        self.selected_points = indices
        self.draw_scatterplot_images()


    def draw_scatterplot_dots(self):
        self.plot_widget.clear()
        self.plot_widget.plot(self.points[:, 0], self.points[:, 1], pen=None, symbolBrush=self.points_color, symbolSize=self.points_size)

        for point in self.points:
            if len(self.points)<5:
                print(' only pos ', point[0], point[1])
            self.plot_widget.plot([point[0]], [point[1]], pen=None, symbolBrush=self.selection_color, symbolSize=self.selection_points_size)

        self.reset_scatterplot(self.points)
        self.plot_widget.update()
        

    def draw_scatterplot_images(self):
        self.plot_widget.clear()
        for idx, index, item in self.image_items:
            self.plot_widget.scene().removeItem(item)

        self.image_items = []
        new_pos=[]

        for i, point in enumerate(self.points):
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
            # new_pos.append((x,y))

            # Add to plot
            self.plot_widget.addItem(image_item)
            
            # Make it clickable.
            self.image_items.append((i, self.indices[i], image_item)) 


        # self.reset_scatterplot(np.array(new_pos))
        self.plot_widget.update()


    # --------------- TO BE REMOVED ---------------------
    def get_indices_nearest_neighbors(self, point):
        # print('point', point)
        x, y =point
        distances = np.sqrt(np.power(self.points[:, 0] - x, 2) + np.power(self.points[:, 1] - y, 2))
        indices = np.argsort(distances)
        return indices

    def find_nearest_neighbors(self, point, n=3):
        indices = self.get_indices_nearest_neighbors(point)
        nearest_indices = indices[:n]
        return nearest_indices
    

