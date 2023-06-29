from PyQt6 import QtCharts
from PyQt6.QtCore import QRect
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import (QGridLayout, QHBoxLayout, QLabel, QLineEdit,
                             QRadioButton, QSlider, QToolTip, QVBoxLayout,
                             QWidget)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import  QPainter,QColor,QPen,QFont
import numpy as np

class BarChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.series = QtCharts.QBarSeries()
        self.chart = QtCharts.QChart()
        self.chart.addSeries(self.series)
        self.axisX = QtCharts.QBarCategoryAxis()
        self.axisY = QtCharts.QValueAxis()
        self.chart.addAxis(self.axisX, Qt.AlignmentFlag.AlignBottom)
        self.chart.addAxis(self.axisY, Qt.AlignmentFlag.AlignLeft)
        self.chart.legend().setVisible(True)
        self.chartView = QtCharts.QChartView(self.chart)
        self.chartView.setRenderHint(QPainter.RenderHint.Antialiasing)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0) 
        layout.addWidget(self.chartView)

        # Enable animations for the chart
        self.chart.setAnimationOptions(QtCharts.QChart.AnimationOption.SeriesAnimations)
        self.chart.setTheme(QtCharts.QChart.ChartTheme.ChartThemeDark)

    def setBarData(self, unique_styles, style_counts, style_count_selection):
        self.series.clear()
        print('unique_styles',unique_styles)
        categories = [str(style) for style in unique_styles]
        self.axisX.clear()
        self.axisX.append(categories)
        self.axisX.setLabelsAngle(-90)  
        font = QFont()
        font.setPointSize(6)
        self.axisX.setLabelsFont(font)

        print('categories',categories)

        bar_set = QtCharts.QBarSet("Images")
        for count in style_counts:
            bar_set.append(count)
        self.series.append(bar_set)

        selected_bar_set = QtCharts.QBarSet("Selected Images")
        for count in style_count_selection:
            selected_bar_set.append(count)
        self.series.append(selected_bar_set)

        # Set tooltips for individual bars
        # bar_count = len(categories)
        # for i in range(bar_count):
        #     bar_item = self.series.barSets()[0].barAt(i)
        #     bar_item.setToolTip(categories[i])

        self.chart.removeSeries(self.series)
        self.chart.addSeries(self.series)
        self.series.attachAxis(self.axisX)
        self.series.attachAxis(self.axisY)

    def fill_in_barplot(self, unique_styles, style_counts, style_count_selection):
        print('fill_in_barplot')
        
        self.setBarData(unique_styles, style_counts, style_count_selection)
        self.repaint()
