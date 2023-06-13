from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QWidget
from PyQt6.QtCore import Qt, pyqtSignal, QPoint, QSize
import numpy as np
import pyqtgraph as pg
from PIL import Image
from PIL.ImageQt import ImageQt

from PyQt6.QtGui import QPixmap, QImage, QBrush,QTransform
from PyQt6.QtWidgets import QGraphicsPixmapItem, QGraphicsEllipseItem
import pandas as pd
import configparser

# from ..home import MainWindow

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
        # self.selected_points = []
        self.outside_points_visible = False

        # self.plot_widget.scene().sigMouseClicked.connect(self.on_canvas_click)
        self.plot_widget.scene().sigMouseMoved.connect(self.on_mouse_move)

        self.draw_scatterplot()
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

   
    # def on_canvas_click(self, ev):
    #     pos = ev.scenePos()
    #     print('on canvas click:', pos)
    #     if ev.button() == Qt.MouseButton.LeftButton:
    #         for idx, index, item in self.image_items:
    #             if item.contains(item.mapFromScene(pos)):
    #                 self.selected_point = int(pos.x()), int(pos.y())
    #                 self.selected_index = index
    #                 self.clicked_on_point()
    #                 break

    # def clicked_on_point(self):
    #     print('point/ image clicked, load on the left')
    #     # Get the selected point
    #     filename= self.get_image_path(self.selected_index)
    #     self.window.initialize_images(self.selected_point,filename)


    def get_image_path(self, index):
        # picks a random image from dataset as initial display
        # load the config file 'config.ini'
        config = configparser.ConfigParser()
        config.read('config.ini')

        data_path = config['main']['pkl_path']
        # dataloading
        df = pd.read_pickle(data_path)

        img_paths = df['image'].iloc[index]
        return img_paths
    
    # def get_image_path(self, index):
    #     # picks a random image from dataset as initial display
    #     img_array2 = da.io.load_image("../data/raw_immutable/nighthawks.png")
    #     return 'image_{}.jpg'.format(index)


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
    
    def get_embeddings_from_nearest_neighbors(self, nearest_indices):
        nearest_embeddings = []
        for index in nearest_indices:
            # Generate a random embedding for each point
            embedding = np.random.rand(128)  # Replace 128 with the desired embedding size
            nearest_embeddings.append(embedding)
        return nearest_embeddings

    
    def get_image_paths(self, indices):
        # Here, you need to provide a list of image paths corresponding to the indices
        # For demonstration purposes, let's assume the images are named as 'image_0.jpg', 'image_1.jpg', etc.
        image_paths = ['image_{}.jpg'.format(idx) for idx in indices]
        return image_paths

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
        self.draw_scatterplot()


    def draw_scatterplot_dots(self):
        self.plot_widget.clear()
        self.plot_widget.plot(self.points[:, 0], self.points[:, 1], pen=None, symbolBrush=self.points_color, symbolSize=self.points_size)

        for point in self.points:
            if len(self.points)<5:
                print(' only pos ', point[0], point[1])
            self.plot_widget.plot([point[0]], [point[1]], pen=None, symbolBrush=self.selection_color, symbolSize=self.selection_points_size)

        self.reset_scatterplot(self.points)
        self.plot_widget.update()
        


    # def draw_scatterplot(self):
    #     self.plot_widget.clear()

    #     for idx, index, item in self.image_items:
    #         self.plot_widget.scene().removeItem(item)

    #     self.image_items = []
    #     new_pos = []
    #     # scale_factor = 0.5  # Initial scale factor for the images
    #     scale_factor = 10

    #     for i, point in enumerate(self.points):
    #         x, y = point
    #         image_path = self.img_paths[i]
    #         pixmap = QPixmap(image_path)

    #         # # Calculate the image size based on the current zoom level
    #         # zoom_level = self.plot_widget.getScale()  # Replace 'getScale()' with the appropriate method for your plotting library
    #         # image_size = max(1, scale_factor / zoom_level)  # Ensure a minimum size of 1 to avoid zero division



    #         # Resize the pixmap based on the scale factor
    #         scaled_width = int(pixmap.width() * scale_factor)
    #         scaled_height = int(pixmap.height() * scale_factor)
    #         scaled_pixmap = pixmap.scaled(QSize(scaled_width, scaled_height), Qt.AspectRatioMode.KeepAspectRatio)

    #         pixmap_item = QGraphicsPixmapItem(scaled_pixmap)

    #         # Adjust the position based on the scaled image size
    #         x_pos = x - scaled_width / 2
    #         y_pos = y - scaled_height / 2

    #         # Check for overlap and adjust the position if needed
    #         for pos in new_pos:
    #             if abs(x_pos - pos[0]) < scaled_width and abs(y_pos - pos[1]) < scaled_height:
    #                 x_pos += scaled_width
    #                 y_pos += scaled_height

    #         new_pos.append((x_pos, y_pos))
    #         pixmap_item.setPos(new_pos[i][0], new_pos[i][1])

    #         self.plot_widget.addItem(pixmap_item)
    #         self.image_items.append((self.indices[i], pixmap_item))

    #         # Update the scale factor based on the current zoom level
    #         # current_scale = self.plot_widget.getViewBox().getState()['scale']
    #         current_scale =  self.plot_widget.scale()
    #         scale_factor = 0.5 / current_scale

    #     self.reset_scatterplot(np.array(new_pos))
    #     self.plot_widget.update()



    def draw_scatterplot(self):
        print('draw_scatterplot')
        self.plot_widget.clear()

        for idx, index, item in self.image_items:
            self.plot_widget.scene().removeItem(item)

        self.image_items = []
        new_pos=[]
        for i, point in enumerate(self.points):
            x,y =point
            image_path = self.img_paths[i]
            pixmap = QPixmap(image_path)

            # Resize the pixmap to a smaller size
            scaled_pixmap = pixmap.scaled(QSize(50, 50), Qt.AspectRatioMode.KeepAspectRatio)

            pixmap_item = QGraphicsPixmapItem(scaled_pixmap)
            scale2= 5
            x_=x - pixmap.width() * scale2 / 2
            y_=y + pixmap.height() * scale2 / 2
            # x_=x * scale2 
            # y_=y * scale2 
            if len(self.points)<5:
                print(x_, y_)
                    
            new_pos.append((x_, y_))
            pixmap_item.setPos(new_pos[i][0], new_pos[i][1])
            
            self.plot_widget.addItem(pixmap_item)
            self.image_items.append((i, self.indices[i], pixmap_item)) 

        self.reset_scatterplot(np.array(new_pos))
        self.plot_widget.update()


    def is_point_in_rectangle(self, point):
        if self.start_point[0] < self.end_point[0]:
            x1, x2 = self.start_point[0], self.end_point[0]
        else:
            x1, x2 = self.end_point[0], self.start_point[0]
        if self.start_point[1] < self.end_point[1]:
            y1, y2 = self.start_point[1], self.end_point[1]
        else:
            y1, y2 = self.end_point[1], self.start_point[1]

        x, y = point[0], point[1]

        if x1 <= x <= x2 and y1 <= y <= y2:
            return True
        else:
            return False

    def set_outside_points_visible(self, visible):
        self.outside_points_visible = visible
        self.draw_scatterplot()
