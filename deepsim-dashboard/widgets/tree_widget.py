# from PyQt6.QtWidgets import  QTreeWidget, QTreeWidgetItem,QWidget,QVBoxLayout
# from PyQt6.QtGui import QIcon

# class HistoryTimelineWidget(QWidget):
#     def __init__(self):
#         super().__init__()
        
#         self.treeWidget = QTreeWidget()
#         self.treeWidget.setColumnCount(2)  # Set the number of columns in the tree widget

#         # Create and add items to the tree widget
#         rootItem = QTreeWidgetItem(self.treeWidget, ["Root Item"])  # The first argument is the parent item (None for the root item)
#         childItem1 = QTreeWidgetItem(rootItem, ["Child Item 1"])
#         childItem2 = QTreeWidgetItem(rootItem, ["Child Item 2"])

#         # Set icons for items
#         rootItem.setIcon(0, QIcon.fromTheme("folder"))
#         childItem1.setIcon(0, QIcon.fromTheme("document"))
#         childItem2.setIcon(0, QIcon.fromTheme("document"))

#         # Expand the root item by default
#         self.treeWidget.expandItem(rootItem)

#         layout = QVBoxLayout()
#         layout.addWidget(self.treeWidget)
#         self.setLayout(layout)

#     def populate_tree(self, data):
#         print('populate tree')
#         # Clear existing items in the tree widget
#         self.treeWidget.clear()
        
#         # Create and add items based on the provided data
#         rootItem = QTreeWidgetItem(self.treeWidget, ["Root Item"])
#         for item_data in data:
#             item = QTreeWidgetItem(rootItem, [item_data])
#             # Set icons for items if needed
#             item.setIcon(0, QIcon.fromTheme("document"))
        
#         # Expand the root item by default
#         self.treeWidget.expandItem(rootItem)        


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





