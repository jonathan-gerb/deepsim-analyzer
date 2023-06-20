
from PyQt6.QtWidgets import QTreeWidget, QTreeWidgetItem, QWidget, QVBoxLayout
from PyQt6.QtGui import QIcon

class HistoryTimelineWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.treeWidget = QTreeWidget()
        self.treeWidget.setColumnCount(2)  # Set the number of columns in the tree widget

        layout = QVBoxLayout()
        layout.addWidget(self.treeWidget)
        self.setLayout(layout)

    def populate_tree(self, timeline_data):
        # Clear existing items in the tree widget
        self.treeWidget.clear()

        # Create and add items based on the provided timeline data
        rootItem = QTreeWidgetItem(self.treeWidget, ["Root Item"])
        for item_data in timeline_data:
            item = QTreeWidgetItem(rootItem, [item_data])
            # Set icons for items if needed
            item.setIcon(0, QIcon.fromTheme("document"))

        # Expand the root item by default
        self.treeWidget.expandItem(rootItem)





