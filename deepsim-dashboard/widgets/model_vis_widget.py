
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtGui import QPixmap

class ModelVis(QWidget):
    def __init__(self):
        super().__init__()

        title = QLabel("Model attention heads visualization", self)

        left_img_size=300
        filename="C:/Users/AAA/Documents/1.studie/MultiMedia_Analytics/deepsim-analyzer/deepsim-dashboard/attention_images/dino_attention.png"
        photo_label = QLabel(self)
        photo_label.setFixedSize(left_img_size, left_img_size)
        pixmap = QPixmap(filename)
        photo_label.setPixmap(
            pixmap.scaledToWidth(photo_label.width()))
        
        layout = QVBoxLayout()
        layout.addWidget(title)
        layout.addWidget(photo_label)
        self.setLayout(layout)






