import sys
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QApplication,QWidget, QGraphicsScene, QGraphicsView, QGraphicsTextItem,QGraphicsPixmapItem, QLabel,QVBoxLayout
from pathlib import Path

class TimelineView(QGraphicsView):
    def __init__(self, scene):
        super().__init__(scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
    
    ## define backgrounds including the timeline to be drawn
    def drawBackground(self, painter: QPainter, rect):
        painter.fillRect(rect, Qt.GlobalColor.white)

        pen = QPen(Qt.GlobalColor.darkYellow)
        pen.setWidth(3)
        painter.setPen(pen)

        timeline_y = 70
        timeline_length = int(self.sceneRect().width())

        painter.drawLine(0, timeline_y, timeline_length, timeline_y)


# class TimelineWindow(QWidget):
#     def __init__(self, main_photo):
#         super().__init__()

#         # self.setWindowTitle("Timeline Visualization")
#         # self.setGeometry(100, 100, 800, 600)

#         images = [{"image_path":'./widgets/mona-lisa.jpg', 'year':2000}, 
#               {"image_path":'./widgets/mona-lisa.jpg', 'year':2010},
#               {"image_path":'./widgets/mona-lisa.jpg', 'year':1990},
#               {"image_path":'./widgets/mona-lisa.jpg', 'year':2020}
#              ]
        
#         images= self.get_timeline_photos(main_photo)

#         self.scene = QGraphicsScene(self)
#         self.images = images

#         # Define the years to be ploted
#         years = sorted([img['year'] for img in self.images])
#         for year in years:
#             x_pos = (year-min(years)) * 20
#             self.create_timeline_point(str(year), x_pos)

#         for image_data in self.images:
#             print(image_data["image_path"])
#             item = self.create_timeline_item(image_data["image_path"], image_data["year"])
#             self.scene.addItem(item)

#         view = TimelineView(self.scene)
#         view.setAlignment(Qt.AlignmentFlag.AlignTop)

#         # self.setCentralWidget(view)


class TimelineWindow(QWidget):
    def __init__(self, main_photo):
        super().__init__()

        self.scene = QGraphicsScene(self)
        
        self.draw_timeline(main_photo)

        view = TimelineView(self.scene)
        view.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(view)


    # separate so we can call it when a new left image appears
    def draw_timeline(self, main_photo):
        images= self.get_timeline_photos(main_photo)
        # Define the years to be ploted
        years = sorted([img['year'] for img in images])
        for year in years:
            x_pos = (year-min(years)) * 20
            self.create_timeline_point(str(year), x_pos)

        for image_data in images:
            print(image_data["image_path"])
            item = self.create_timeline_item(images, image_data["image_path"], image_data["year"])
            self.scene.addItem(item)
        

    def get_timeline_photos(self, main_photo):
        basepath = Path(__file__)
        # imgs_filepath = basepath.parent.parent / 'images'
        imgs_filepath = basepath.parent.parent.parent.parent / 'data/raw_immutable/test_images'
        imag_name="vincent-van-gogh_still-life-with-a-basket-of-apples-and-two-pumpkins-1885.jpg"
        images = [
            {"image_path": str(imgs_filepath / imag_name), 'year': 2000},
            {"image_path": str(imgs_filepath / imag_name), 'year': 2010},
            {"image_path": str(imgs_filepath / imag_name), 'year': 1990},
            {"image_path": str(imgs_filepath / imag_name), 'year': 2020}
        ]
        return images
    
    ## add year on the timeline
    def create_timeline_point(self, letter, x_pos):
        text_item = QGraphicsTextItem(letter)
        font = QFont("Arial", 14)
        text_item.setFont(font)

        text_item.setPos(x_pos, 80)
        self.scene.addItem(text_item)


    ## add images to the timeline
    def create_timeline_item(self,images, image_path, year):
        pixmap=QPixmap(image_path).scaled(30,30)
        item = QGraphicsPixmapItem(pixmap)
        item.setFlag(QGraphicsPixmapItem.GraphicsItemFlag.ItemIsMovable)
        item.setFlag(QGraphicsPixmapItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        
        # Calculate position on the timeline based on year
        x_pos = (year-min([img["year"] for img in images])) * 20
        item.setPos(x_pos, 50)
        
        return item


# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     ## define images to be ploted
#     window = TimelineWindow('')
#     window.show()
#     sys.exit(app.exec())
