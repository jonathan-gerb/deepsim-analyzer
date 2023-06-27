from PyQt6 import QtCharts
from PyQt6.QtCore import QRect
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import (QGridLayout, QHBoxLayout, QLabel, QLineEdit,
                             QRadioButton, QSlider, QToolTip, QVBoxLayout,
                             QWidget)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import  QPainter

class RangeSlider(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.domain = [0, 100]
        self.values = [0, 100]
        self.update = [0, 100]
        self.inputValues = [0, 100]
        self.color = QColor(0, 0, 255)  # Example color
        self.typeNumber = "int"  # Example type
        self.step = 1  # Example step
        self.hover_index = 0  # Example hover index
        self.isToggleOn = False
        self.initUI()

    def initUI(self):
        layout = QGridLayout(self)
        # BarChart Widget
        self.bar_chart = BarChart(self)  # Replace BarChart with your own widget
        # self.bar_chart.setFixedHeight(40)
        # self.bar_chart.setFixedWidth(70)
        # self.bar_chart.autoFillBackground(True)
        # bar_chart.setColor(self.color)
        # Add other necessary configuration for the BarChart widget
        layout.addWidget(self.bar_chart, 0, 0, 1, 3)
       
        # Double Range Slider Widget
        range_slider = QSlider()
        range_slider.setOrientation(Qt.Orientation.Horizontal)
        # range_slider.setRange(self.domain[0], self.domain[1])
        # range_slider.setValues(self.values[0], self.values[1])
        range_slider.setMinimum(self.domain[0])
        range_slider.setMaximum(self.domain[1])
        range_slider.setValue(self.values[0])
        range_slider.setTickPosition(QSlider.TickPosition.TicksBothSides)
        range_slider.setTickInterval(1)
        range_slider.setSingleStep(1)
        range_slider.sliderMoved.connect(self.changeSlider)
        layout.addWidget(range_slider, 1, 0, 1, 3)

        # Set size policies for the widgets
        # size_policy_chart = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # size_policy_slider = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        # self.bar_chart.setSizePolicy(size_policy_chart)
        # range_slider.setSizePolicy(size_policy_slider)

        # Set stretch factors for the widgets
        # layout.setColumnStretch(0, 1)
        # layout.setColumnStretch(1, 1)
        # layout.setColumnStretch(2, 1)

        self.setLayout(layout)

        # Additional styling if required
        # self.setStyleSheet("...")

    def changeSlider(self, values):
        # Function to handle slider value changes
        pass


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
        self.chart.legend().setVisible(False)
        self.chartView = QtCharts.QChartView(self.chart)
        self.chartView.setRenderHint(QPainter.RenderHint.Antialiasing)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0) 
        layout.addWidget(self.chartView)

        # self.setMinimumSize(200, 200)
        # Enable animations for the chart
        # self.chart.setAnimationOptions(QtCharts.QChart.SeriesAnimations)
        self.chart.setAnimationOptions(QtCharts.QChart.AnimationOption.AllAnimations)
        self.chart.setTheme(QtCharts.QChart.ChartTheme.ChartThemeDark)




    def setColor(self, color):
        # Set color for the bar chart
        palette = self.chartView.palette()
        palette.setColor(QPalette.ColorRole.Window, color)
        self.chartView.setPalette(palette)

    def setBarData(self, unique_styles, style_counts, style_count_selection):
        self.series.clear()
        categories = [str(style) for style in unique_styles]
        self.axisX.clear()
        self.axisX.append(categories)

        bar_set = QtCharts.QBarSet("Bar")
        for count in style_counts:
            bar_set.append(count)
        # bar_set.setColor(QColor(0, 0, 255)) 
        self.series.append(bar_set)

        selected_bar_set = QtCharts.QBarSet("Selected Bar")
        for count in style_count_selection:
            selected_bar_set.append(count)
        # selected_bar_set.setColor(QColor(255, 0, 0))
        self.series.append(selected_bar_set)

        self.chart.removeSeries(self.series)
        self.chart.addSeries(self.series)
        self.series.attachAxis(self.axisX)
        self.series.attachAxis(self.axisY)

    def fill_in_barplot(self, unique_styles, style_counts, style_count_selection):
        print('fill_in_barplot')
        self.setBarData(unique_styles, style_counts, style_count_selection)
        self.repaint()
