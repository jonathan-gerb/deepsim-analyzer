
import sys
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsView, QGraphicsTextItem,QGraphicsPixmapItem

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


class TimelineWindow(QMainWindow):
    def __init__(self, images:dict):
        super().__init__()

        self.setWindowTitle("Timeline Visualization")
        self.setGeometry(100, 100, 800, 600)

        self.scene = QGraphicsScene(self)
        self.images = images

        # Define the years to be ploted
        years = sorted([img['year'] for img in self.images])
        for year in years:
            x_pos = (year-min(years)) * 20
            self.create_timeline_point(str(year), x_pos)

        for image_data in self.images:
            item = self.create_timeline_item(image_data["image_path"], image_data["year"])
            self.scene.addItem(item)

        view = TimelineView(self.scene)
        view.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.setCentralWidget(view)


    ## add year on the timeline
    def create_timeline_point(self, letter, x_pos):
        text_item = QGraphicsTextItem(letter)
        font = QFont("Arial", 14)
        text_item.setFont(font)

        text_item.setPos(x_pos, 80)
        self.scene.addItem(text_item)


    ## add images to the timeline
    def create_timeline_item(self, image_path, year):
        item = QGraphicsPixmapItem(QPixmap(image_path).scaled(30,30))
        item.setFlag(QGraphicsPixmapItem.GraphicsItemFlag.ItemIsMovable)
        item.setFlag(QGraphicsPixmapItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        
        # Calculate position on the timeline based on year
        x_pos = (year-min([img["year"] for img in self.images])) * 20
        item.setPos(x_pos, 50)
        
        return item


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ## define images to be ploted
    images = [{"image_path":'/Users/wangyangwu/Downloads/131107152744-mona-lisa.jpg', 'year':2000}, 
              {"image_path":'/Users/wangyangwu/Downloads/131107152744-mona-lisa.jpg', 'year':2010},
              {"image_path":'/Users/wangyangwu/Downloads/131107152744-mona-lisa.jpg', 'year':1990},
              {"image_path":'/Users/wangyangwu/Downloads/131107152744-mona-lisa.jpg', 'year':2020}
             ]
    window = TimelineWindow(images)
    window.show()
    sys.exit(app.exec())