from PyQt6 import QtCharts
from PyQt6.QtCore import QRect, pyqtSignal, QPointF
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import (QGridLayout, QHBoxLayout, QLabel, QLineEdit,
                             QRadioButton, QSlider, QToolTip, QVBoxLayout,
                             QWidget)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import  QPainter,QColor,QPen,QFont, QMouseEvent, QCursor
import numpy as np

import pyqtgraph as pg

class BarChart(QWidget):
    def __init__(self, category, parent=None):
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

        self.series.hovered.connect(self.handle_bar_hovered)
        self.category = category
        self.categories = None


    def handle_bar_hovered(self, status, bar_set, bar_index):
        """
        Take in hover signal from barpot.
        """
        mouse_pos = QCursor.pos()
        if status:
            self.show_tooltip(bar_set, mouse_pos)
        else:
            QToolTip.hideText()

    def show_tooltip(self, bar_value, mouse_pos):
        """
        Plot tooltip onto UI.
        """
        tooltip_text = f"{self.category}: {self.categories[bar_value]}"
        QToolTip.showText(mouse_pos, tooltip_text, self)


    def setBarData(self, unique_styles, style_counts, style_count_selection):
        """
        Collect categories and add to barplot.
        """
        self.series.clear()
        self.categories = [str(style).replace("_", " ") for style in unique_styles]
        self.axisX.clear()
        self.axisX.append(self.categories)
        font = QFont()
        font.setPointSize(8)
        self.axisX.setLabelsFont(font)

        bar_set = QtCharts.QBarSet("Images")
        for count in style_counts:
            bar_set.append(count)
        self.series.append(bar_set)

        selected_bar_set = QtCharts.QBarSet("Selected Images")
        for count in style_count_selection:
            selected_bar_set.append(count)
        self.series.append(selected_bar_set)

        self.chart.removeSeries(self.series)
        self.chart.addSeries(self.series)
        self.series.attachAxis(self.axisX)
        self.series.attachAxis(self.axisY)


    def fill_in_barplot(self, unique_styles, style_counts, style_count_selection):
        self.setBarData(unique_styles, style_counts, style_count_selection)
        self.repaint()
