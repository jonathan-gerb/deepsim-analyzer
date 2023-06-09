import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QGraphicsView, QGraphicsScene
from PyQt6.QtCore import Qt 

class HistoryTimelineWidget(QWidget):
    def __init__(self):
        super().__init__()
        
        # Create a layout for the timeline widget
        layout = QVBoxLayout(self)
        
        # Create a label for the timeline title
        title_label = QLabel("History Timeline")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Create a QGraphicsView to display the timeline events
        scene = QGraphicsScene()
        graphics_view = QGraphicsView(scene)
        layout.addWidget(graphics_view)
        
        # Add timeline events to the scene
        event1 = QLabel("Event 1")
        event1.setStyleSheet("background-color: #FF0000; color: #FFFFFF;")
        scene.addWidget(event1)
        
        event2 = QLabel("Event 2")
        event2.setStyleSheet("background-color: #00FF00; color: #000000;")
        scene.addWidget(event2)
        
        event3 = QLabel("Event 3")
        event3.setStyleSheet("background-color: #0000FF; color: #FFFFFF;")
        scene.addWidget(event3)
        

