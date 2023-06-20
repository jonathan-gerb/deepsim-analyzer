# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'form.ui'
##
## Created by: Qt User Interface Compiler version 6.5.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PyQt6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PyQt6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PyQt6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QMainWindow,
    QPushButton, QRadioButton, QSizePolicy, QTabWidget,
    QWidget)

from pyqtgraph import PlotWidget

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1534, 766)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.b_upload = QPushButton(self.centralwidget)
        self.b_upload.setObjectName(u"b_upload")
        self.b_upload.setGeometry(QRect(10, 330, 491, 25))
        sizePolicy = QSizePolicy()
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.b_upload.sizePolicy().hasHeightForWidth())
        self.b_upload.setSizePolicy(sizePolicy)
        self.box_metric_tabs = QTabWidget(self.centralwidget)
        self.box_metric_tabs.setObjectName(u"box_metric_tabs")
        self.box_metric_tabs.setGeometry(QRect(510, 460, 511, 261))
        self.dino_tab = QWidget()
        self.dino_tab.setObjectName(u"dino_tab")
        self.dino_opt1 = QRadioButton(self.dino_tab)
        self.dino_opt1.setObjectName(u"dino_opt1")
        self.dino_opt1.setGeometry(QRect(390, 10, 112, 23))
        self.dino_opt2 = QRadioButton(self.dino_tab)
        self.dino_opt2.setObjectName(u"dino_opt2")
        self.dino_opt2.setGeometry(QRect(390, 40, 112, 23))
        self.dino_opt3 = QRadioButton(self.dino_tab)
        self.dino_opt3.setObjectName(u"dino_opt3")
        self.dino_opt3.setGeometry(QRect(390, 70, 112, 23))
        self.box_metric_tabs.addTab(self.dino_tab, "")
        self.texture_tab = QWidget()
        self.texture_tab.setObjectName(u"texture_tab")
        self.box_metric_tabs.addTab(self.texture_tab, "")
        self.emotion_tab = QWidget()
        self.emotion_tab.setObjectName(u"emotion_tab")
        self.box_metric_tabs.addTab(self.emotion_tab, "")
        self.dummy_tab = QWidget()
        self.dummy_tab.setObjectName(u"dummy_tab")
        self.box_metric_tabs.addTab(self.dummy_tab, "")
        self.box_left_low = QWidget(self.centralwidget)
        self.box_left_low.setObjectName(u"box_left_low")
        self.box_left_low.setGeometry(QRect(10, 460, 491, 261))
        self.box_right_low = QWidget(self.centralwidget)
        self.box_right_low.setObjectName(u"box_right_low")
        self.box_right_low.setGeometry(QRect(1030, 460, 491, 261))
        self.box_metainfo = QWidget(self.centralwidget)
        self.box_metainfo.setObjectName(u"box_metainfo")
        self.box_metainfo.setGeometry(QRect(10, 370, 491, 80))
        self.t_artist = QLabel(self.box_metainfo)
        self.t_artist.setObjectName(u"t_artist")
        self.t_artist.setGeometry(QRect(10, 10, 231, 17))
        self.t_date = QLabel(self.box_metainfo)
        self.t_date.setObjectName(u"t_date")
        self.t_date.setGeometry(QRect(10, 30, 231, 17))
        self.t_style = QLabel(self.box_metainfo)
        self.t_style.setObjectName(u"t_style")
        self.t_style.setGeometry(QRect(10, 50, 231, 17))
        self.t_tags = QLabel(self.box_metainfo)
        self.t_tags.setObjectName(u"t_tags")
        self.t_tags.setGeometry(QRect(250, 10, 231, 17))
        self.box_left_img = QLabel(self.centralwidget)
        self.box_left_img.setObjectName(u"box_left_img")
        self.box_left_img.setGeometry(QRect(10, 10, 491, 311))
        self.box_right_img = QLabel(self.centralwidget)
        self.box_right_img.setObjectName(u"box_right_img")
        self.box_right_img.setGeometry(QRect(1030, 10, 491, 311))
        self.horizontalLayoutWidget = QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setObjectName(u"horizontalLayoutWidget")
        self.horizontalLayoutWidget.setGeometry(QRect(510, 420, 511, 31))
        self.scatterplot_options = QHBoxLayout(self.horizontalLayoutWidget)
        self.scatterplot_options.setObjectName(u"scatterplot_options")
        self.scatterplot_options.setContentsMargins(0, 0, 0, 0)
        self.b_recalculate = QPushButton(self.horizontalLayoutWidget)
        self.b_recalculate.setObjectName(u"b_recalculate")

        self.scatterplot_options.addWidget(self.b_recalculate)

        self.b_combined = QPushButton(self.horizontalLayoutWidget)
        self.b_combined.setObjectName(u"b_combined")

        self.scatterplot_options.addWidget(self.b_combined)

        self.b_metric = QPushButton(self.horizontalLayoutWidget)
        self.b_metric.setObjectName(u"b_metric")

        self.scatterplot_options.addWidget(self.b_metric)

        self.r_image_points = QRadioButton(self.horizontalLayoutWidget)
        self.r_image_points.setObjectName(u"r_image_points")

        self.scatterplot_options.addWidget(self.r_image_points)

        self.scatterplot_frame = PlotWidget(self.centralwidget)
        self.scatterplot_frame.setObjectName(u"scatterplot_frame")
        self.scatterplot_frame.setGeometry(QRect(510, 10, 511, 401))
        self.n1 = QLabel(self.centralwidget)
        self.n1.setObjectName(u"n1")
        self.n1.setGeometry(QRect(1030, 330, 161, 121))
        self.n2 = QLabel(self.centralwidget)
        self.n2.setObjectName(u"n2")
        self.n2.setGeometry(QRect(1190, 330, 171, 121))
        self.n3 = QLabel(self.centralwidget)
        self.n3.setObjectName(u"n3")
        self.n3.setGeometry(QRect(1360, 330, 161, 121))
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        self.box_metric_tabs.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Deep Similarity Analyzer", None))
        self.b_upload.setText(QCoreApplication.translate("MainWindow", u"Upload Image", None))
        self.dino_opt1.setText(QCoreApplication.translate("MainWindow", u"Option 1", None))
        self.dino_opt2.setText(QCoreApplication.translate("MainWindow", u"Option 2", None))
        self.dino_opt3.setText(QCoreApplication.translate("MainWindow", u"Option 3", None))
        self.box_metric_tabs.setTabText(self.box_metric_tabs.indexOf(self.dino_tab), QCoreApplication.translate("MainWindow", u"Dino", None))
        self.box_metric_tabs.setTabText(self.box_metric_tabs.indexOf(self.texture_tab), QCoreApplication.translate("MainWindow", u"Texture", None))
        self.box_metric_tabs.setTabText(self.box_metric_tabs.indexOf(self.emotion_tab), QCoreApplication.translate("MainWindow", u"Emotion", None))
        self.box_metric_tabs.setTabText(self.box_metric_tabs.indexOf(self.dummy_tab), QCoreApplication.translate("MainWindow", u"Dummy", None))
        self.t_artist.setText(QCoreApplication.translate("MainWindow", u"Artist: ", None))
        self.t_date.setText(QCoreApplication.translate("MainWindow", u"Date:", None))
        self.t_style.setText(QCoreApplication.translate("MainWindow", u"Style:", None))
        self.t_tags.setText(QCoreApplication.translate("MainWindow", u"Tags:", None))
        self.box_left_img.setText("")
        self.box_right_img.setText("")
        self.b_recalculate.setText(QCoreApplication.translate("MainWindow", u"metric specific", None))
        self.b_combined.setText(QCoreApplication.translate("MainWindow", u"combined", None))
        self.b_metric.setText(QCoreApplication.translate("MainWindow", u"recalculate", None))
        self.r_image_points.setText(QCoreApplication.translate("MainWindow", u"points as images", None))
        self.n1.setText("")
        self.n2.setText("")
        self.n3.setText("")
    # retranslateUi

