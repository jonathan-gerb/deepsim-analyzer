import configparser

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PIL import Image
from PIL.ImageQt import ImageQt
from tqdm import tqdm

from PyQt6.QtCore import QPoint, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QImage, QPixmap, QTransform
from PyQt6.QtWidgets import (QApplication, QDialog, QGraphicsEllipseItem,
                               QGraphicsPixmapItem, QMainWindow, QPushButton,
                               QVBoxLayout, QWidget)

# from ..home import DeepSimDashboard

# import deepsim_analyzer as da

class ScatterplotWidget(QWidget):
    selected_idx = pyqtSignal(list)
    label = pyqtSignal(str)

    # Define a custom signal to emit when a point is clicked
    point_clicked = pyqtSignal(tuple)

    def __init__(self, points, indices, img_paths, config, plot_widget):
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

        self.plot_widget = plot_widget
        self.plot_widget.setMouseEnabled(True, True)
        self.plot_widget.setLimits(xMin=-np.inf, xMax=np.inf, yMin=-np.inf, yMax=np.inf)
        self.plot_widget.setAspectLocked(lock=True)

        # self.clear_button = QPushButton("Clear Selection")
        # self.clear_button.clicked.connect(self.clear_selection)

        self.image_items = []

        self.selected_point=None
        self.selected_index=None
        self.plot_index=None


        # self.selected_points = []
        self.outside_points_visible = False

        self.plot_widget.scene().sigMouseMoved.connect(self.on_mouse_move)

        self.draw_scatterplot_dots()
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
    #                 self.window.clicked_on_point()
    #                 break

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
        self.draw_scatterplot_dots()


    def draw_scatterplot_dots(self):
        self.plot_widget.clear()
        self.plot_widget.plot(self.points[:, 0], self.points[:, 1], pen=None, symbolBrush=self.points_color, symbolSize=self.points_size)

        for point in self.points:
            if len(self.points)<5:
                print(' only pos ', point[0], point[1])
            self.plot_widget.plot([point[0]], [point[1]], pen=None, symbolBrush=self.selection_color, symbolSize=self.selection_points_size)

        self.reset_scatterplot(self.points)
        self.plot_widget.update()
        


    def draw_scatterplot(self):
        print('draw_scatterplot')
        self.plot_widget.clear()
        for idx, index, item in self.image_items:
            self.plot_widget.scene().removeItem(item)

        self.image_items = []
        new_pos = []
        image_size = 24
        # min, max = np.min(self.points), np.max(self.points)
        scale = image_size * 8
        for i, point in tqdm(enumerate(self.points), desc="placing images on plot", total=len(self.points)):
            x, y = point
            x, y = x * scale, y * scale
            image_path = self.img_paths[i]
            pixmap = QPixmap(image_path)

            # Flip the pixmap vertically
            pixmap = pixmap.transformed(QTransform().scale(1, -1))

            # Resize the pixmap to a smaller size
            scaled_pixmap = pixmap.scaled(QSize(image_size, image_size), Qt.AspectRatioMode.KeepAspectRatio)

            pixmap_item = QGraphicsPixmapItem(scaled_pixmap)
            x_ = x - (pixmap.width() / 2)
            y_ = y + (pixmap.height() / 2)
            
            # if len(self.points) < 5:
            #     print(x_, y_)
                    
            pixmap_item.setPos(x_, y_)
            new_pos.append((x_, y_))
            
            self.plot_widget.addItem(pixmap_item)
            self.image_items.append((i, self.indices[i], pixmap_item)) 

        self.reset_scatterplot(np.array(new_pos))
        self.plot_widget.update()


    # def is_point_in_rectangle(self, point):
    #     if self.start_point[0] < self.end_point[0]:
    #         x1, x2 = self.start_point[0], self.end_point[0]
    #     else:
    #         x1, x2 = self.end_point[0], self.start_point[0]
    #     if self.start_point[1] < self.end_point[1]:
    #         y1, y2 = self.start_point[1], self.end_point[1]
    #     else:
    #         y1, y2 = self.end_point[1], self.start_point[1]

    #     x, y = point[0], point[1]

    #     if x1 <= x <= x2 and y1 <= y <= y2:
    #         return True
    #     else:
    #         return False

    # def set_outside_points_visible(self, visible):
    #     self.outside_points_visible = visible
    #     self.draw_scatterplot()





########################## other attempts #######################################


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
    #         # pixmap_item= pixmap_item.setTransform(1,-1)

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
    #         self.image_items.append((i, self.indices[i], pixmap_item))

    #         # Update the scale factor based on the current zoom level
    #         # current_scale = self.plot_widget.getViewBox().getState()['scale']
    #         # current_scale =  self.plot_widget.scale()
    #         # current_scale = self.plot_widget.scale().x(), self.plot_widget.scale().y()
    #         current_scale = self.plot_widget.transform().m11()

    #         scale_factor = 0.5 / current_scale

    #     self.reset_scatterplot(np.array(new_pos))
    #     self.plot_widget.update()








# from pathlib import Path
#         # basepath = Path(__file__)
#         # imgs_filepath = basepath.parent.parent.parent.parent / 'data/raw_immutable/test_images'
#         # print(str(imgs_filepath)+'/ad-reinhardt_abstract-painting-red-1952.jpg')
#         # self.images = [
#         #     str(imgs_filepath)+'/ad-reinhardt_abstract-painting-red-1952.jpg',
#         #     str(imgs_filepath)+'/balthus_sleeping-girl-1943.jpg',
#         #     str(imgs_filepath)+'/frederic-bazille_after-the-bath.jpg'
#         # ]

# import sys
# import pyqtgraph.opengl as gl
# from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel
# from PyQt6.QtGui import QPixmap, QColor
# from PyQt6.QtCore import Qt

# class ScatterPlotWidget(QMainWindow):
#     def __init__(self):
#         super().__init__()

#         self.scatter_plot = gl.GLViewWidget()
#         self.setCentralWidget(self.scatter_plot)

#         basepath = Path(__file__)
#         imgs_filepath = basepath.parent.parent.parent.parent / 'data/raw_immutable/test_images'
#         print(str(imgs_filepath)+'/ad-reinhardt_abstract-painting-red-1952.jpg')
#         self.images = [
#             str(imgs_filepath)+'/ad-reinhardt_abstract-painting-red-1952.jpg',
#             str(imgs_filepath)+'/balthus_sleeping-girl-1943.jpg',
#             str(imgs_filepath)+'/frederic-bazille_after-the-bath.jpg'
#         ]

#         self.image_labels = []

#         # Generate random data for demonstration
#         num_points = 100
#         x = [i for i in range(num_points)]
#         y = [i for i in range(num_points)]
#         z = [i for i in range(num_points)]

#         self.scatter_plot.opts['distance'] = 20
#         self.scatter_plot.setBackgroundColor(QColor(255, 255, 255))
#         self.scatter_plot.setCameraPosition(distance=40, elevation=20, azimuth=135)
#         # self.scatter_plot.setZoomMethod(self.scatter_plot.ZoomMethod.NoZoom)
#         self.scatter_plot.setWindowTitle('3D Scatter Plot')

#         scatter_item = gl.GLScatterPlotItem(pos=np.vstack((x, y, z)), size=5)
#         self.scatter_plot.addItem(scatter_item)

#     def wheelEvent(self, event):
#         zoom_in_factor = 1.1
#         zoom_out_factor = 0.9

#         if event.angleDelta().y() > 0:
#             self.scatter_plot.opts['distance'] /= zoom_in_factor
#         else:
#             self.scatter_plot.opts['distance'] *= zoom_out_factor

#         self.update_image_labels()
#         event.accept()

#     def update_image_labels(self):
#         current_distance = self.scatter_plot.opts['distance']

#         if current_distance < 15:
#             if not self.image_labels:
#                 # Create image labels at the initial zoom level
#                 self.create_image_labels()

#             # Calculate the new label size based on the distance
#             label_size = int(40 * (15 - current_distance) / 15)

#             # Update the size of the image labels
#             for label in self.image_labels:
#                 label.setFont(label.font().pointSize() + label_size)

#         else:
#             # Remove the image labels if the distance is high
#             self.remove_image_labels()

#     def create_image_labels(self):
#         for image_path in self.images:
#             pixmap = QPixmap(image_path)
#             label = QLabel(self.scatter_plot)

#             # Set the pixmap on the label
#             label.setPixmap(pixmap)

#             # Set the label position
#             label.setPos(0, 0, 0)

#             # Add the label to the scatter plot
#             self.scatter_plot.addItem(label)

#             self.image_labels.append(label)

#     def remove_image_labels(self):
#         for label in self.image_labels:
#             self.scatter_plot.removeItem(label)
#         self.image_labels = []


# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     window = ScatterPlotWidget()
#     window.show()
#     sys.exit(app.exec())







# import sys
# import numpy as np
# from PyQt6.QtCore import Qt, QSize
# from PyQt6.QtGui import QColor, QImage, QPixmap, QTransform
# from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout
# import pyqtgraph.opengl as gl

# class ScatterPlotWidget(QMainWindow):
#     def __init__(self):
#         super().__init__()

#         self.scatter_plot = gl.GLViewWidget(self)
#         self.setCentralWidget(self.scatter_plot)

#         # Generate random data for demonstration
#         num_points = 100
#         pos = np.random.normal(size=(num_points, 3))

#         self.scatter_plot.setBackgroundColor(QColor(255, 255, 255))
#         self.scatter_plot.setCameraPosition(distance=50, elevation=30, azimuth=135)  # Set camera position

#         # Add images at each point in the scatter plot
#         self.add_images(pos)

#         self.scatter_plot.setWindowTitle('3D Scatter Plot')

#     def add_images(self, pos):
#         image_paths = ["path_to_image_1.jpg", "path_to_image_2.jpg", "path_to_image_3.jpg"]  # Replace with actual image paths

#         for i in range(len(pos)):
#             x, y, z = pos[i]

#             image_path = image_paths[i % len(image_paths)]
#             image = QImage(image_path)

#             # Flip the image vertically
#             image = image.mirrored(False, True)

#             # Convert QImage to QPixmap
#             pixmap = QPixmap.fromImage(image)

#             # Create a texture from the QPixmap
#             texture = gl.Texture(pixmap)

#             # Create a GLScatterPlotItem and add it to the scatter plot
#             item = gl.GLScatterPlotItem(pos=np.array([[x, y, z]]), size=10)
#             item.setGLOptions('translucent')
#             item.setTexture(texture)
#             self.scatter_plot.addItem(item)

# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     window = ScatterPlotWidget()
#     window.show()
#     sys.exit(app.exec())






# from PyQt6.QtCore import Qt, QRectF
# from PyQt6.QtGui import QPixmap, QPainter
# from PyQt6.QtWidgets import (
#     QApplication,
#     QMainWindow,
#     QGraphicsView,
#     QGraphicsScene,
#     QGraphicsObject,
#     QGraphicsPixmapItem,
#     QLabel,
#     QVBoxLayout,
#     QWidget
# )
# from pathlib import Path
# import sys


# class ImageDotItem(QGraphicsPixmapItem):
#     def __init__(self, x, y, image_path):
#         super().__init__()

#         self.setFlag(QGraphicsPixmapItem.GraphicsItemFlag.ItemIgnoresTransformations, True)

#         self.setPixmap(QPixmap(image_path).scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio))
#         self.setOffset(-100, -100)  # Adjust the position of the image

#         self.setPos(x, y)


# class Scatterplot(QGraphicsView):
#     def __init__(self, points, images, parent=None):
#         super().__init__(parent)
#         self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
#         self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

#         self.scene = QGraphicsScene(self)
#         self.setScene(self.scene)

#         self.points = points
#         self.images = images
#         self.image_items = []

#         self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)  # Enable panning of the entire scene

#         self.draw_scatterplot()

#     def draw_scatterplot(self):
#         self.scene.clear()
#         self.image_items.clear()

#         for point, image_path in zip(self.points, self.images):
#             x, y = point
#             dot_item = ImageDotItem(x, y, image_path)
#             self.scene.addItem(dot_item)
#             self.image_items.append(dot_item)

#     def wheelEvent(self, event):
#         zoom_in_factor = 1.15
#         zoom_out_factor = 1 / zoom_in_factor

#         if event.angleDelta().y() > 0:
#             zoom_factor = zoom_in_factor
#         else:
#             zoom_factor = zoom_out_factor

#         self.scale(zoom_factor, zoom_factor)


# app = QApplication(sys.argv)

# points = [(100, 100), (200, 200), (300, 300)]
# basepath = Path(__file__)
# imgs_filepath = basepath.parent.parent.parent.parent / 'data/raw_immutable/test_images'
# print(str(imgs_filepath)+'/ad-reinhardt_abstract-painting-red-1952.jpg')
# images = [
#     str(imgs_filepath)+'/ad-reinhardt_abstract-painting-red-1952.jpg',
#     str(imgs_filepath)+'/balthus_sleeping-girl-1943.jpg',
#     str(imgs_filepath)+'/frederic-bazille_after-the-bath.jpg'
# ]

# window = QMainWindow()
# scatterplot = Scatterplot(points, images)
# window.setCentralWidget(scatterplot)
# window.setGeometry(100, 100, 800, 600)
# window.show()

# sys.exit(app.exec())

























# from PyQt6.QtCore import Qt, QRectF
# from PyQt6.QtGui import QPixmap, QPainter
# from PyQt6.QtWidgets import (
#     QApplication,
#     QMainWindow,
#     QGraphicsView,
#     QGraphicsScene,
#     QGraphicsObject,
#     QGraphicsProxyWidget,
#     QLabel,
#     QVBoxLayout,
#     QWidget
# )
# from pathlib import Path
# import sys


# class ImageDotItem(QGraphicsObject):
#     def __init__(self, x, y, image_path):
#         super().__init__()

#         self.setFlag(QGraphicsObject.GraphicsItemFlag.ItemIgnoresTransformations, True)

#         self.image_path = image_path
#         self.label_widget = None

#         self.updateLabelWidget()
#         self.setPos(x, y)

#     def updateLabelWidget(self):
#         if self.label_widget is not None:
#             self.scene().removeItem(self.label_widget)

#         self.label_widget = QGraphicsProxyWidget(self)

#         label_layout = QVBoxLayout()
#         label_widget = QWidget()
#         label_widget.setLayout(label_layout)

#         image_label = QLabel()
#         pixmap = QPixmap(self.image_path)
#         image_label.setPixmap(pixmap.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio))
#         label_layout.addWidget(image_label)

#         self.label_widget.setWidget(label_widget)
#         self.label_widget.setPos(-100, -250)  # Adjust the position of the label widget

#     def boundingRect(self):
#         zoom_factor = self.scene().views()[0].transform().m11()
#         size = max(self.pixmap().width(), self.pixmap().height()) * zoom_factor
#         return QRectF(-size / 2, -size / 2, size, size)

#     def paint(self, painter, option, widget):
#         pass

#     def pixmap(self):
#         return QPixmap(self.image_path)

#     def boundingRect(self):
#         return QRectF(-self.pixmap().width() / 2, -self.pixmap().height() / 2,
#                       self.pixmap().width(), self.pixmap().height())


# class ZoomableScatterplot(QGraphicsView):
#     def __init__(self, points, images, parent=None):
#         super().__init__(parent)
#         self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
#         self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

#         self.scene = QGraphicsScene(self)
#         self.setScene(self.scene)

#         self.points = points
#         self.images = images
#         self.image_items = []

#         self.draw_scatterplot()

#     def draw_scatterplot(self):
#         self.scene.clear()
#         self.image_items.clear()

#         for point, image_path in zip(self.points, self.images):
#             x, y = point
#             dot_item = ImageDotItem(x, y, image_path)
#             self.scene.addItem(dot_item)
#             self.image_items.append(dot_item)

#     def wheelEvent(self, event):
#         zoom_in_factor = 1.15
#         zoom_out_factor = 1 / zoom_in_factor

#         if event.angleDelta().y() > 0:
#             zoom_factor = zoom_in_factor
#         else:
#             zoom_factor = zoom_out_factor

#         self.scale(zoom_factor, zoom_factor)


# app = QApplication(sys.argv)

# points = [(100, 100), (200, 200), (300, 300)]
# basepath = Path(__file__)
# imgs_filepath = basepath.parent.parent.parent.parent / 'data/raw_immutable/test_images'
# print(str(imgs_filepath)+'/ad-reinhardt_abstract-painting-red-1952.jpg')
# images = [
#     str(imgs_filepath)+'/ad-reinhardt_abstract-painting-red-1952.jpg',
#     str(imgs_filepath)+'/balthus_sleeping-girl-1943.jpg',
#     str(imgs_filepath)+'/frederic-bazille_after-the-bath.jpg'
# ]

# window = QMainWindow()
# scatterplot = ZoomableScatterplot(points, images)
# window.setCentralWidget(scatterplot)
# window.setGeometry(100, 100, 800, 600)
# window.show()

# sys.exit(app.exec())
