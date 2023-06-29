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
from PyQt6.QtWidgets import (QApplication, QComboBox, QFormLayout, QFrame,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QPlainTextEdit, QPushButton, QRadioButton,
    QSizePolicy, QSlider, QTabWidget, QVBoxLayout,
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
        self.b_upload.setGeometry(QRect(10, 330, 451, 31))
        sizePolicy = QSizePolicy()
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.b_upload.sizePolicy().hasHeightForWidth())
        self.b_upload.setSizePolicy(sizePolicy)
        self.box_left_low = QWidget(self.centralwidget)
        self.box_left_low.setObjectName(u"box_left_low")
        self.box_left_low.setGeometry(QRect(10, 460, 461, 301))
        self.l_timeline = QLabel(self.box_left_low)
        self.l_timeline.setObjectName(u"l_timeline")
        self.l_timeline.setGeometry(QRect(10, 170, 439, 21))
        self.verticalLayoutWidget_5 = QWidget(self.box_left_low)
        self.verticalLayoutWidget_5.setObjectName(u"verticalLayoutWidget_5")
        self.verticalLayoutWidget_5.setGeometry(QRect(9, 199, 441, 101))
        self.box_timeline_layout = QVBoxLayout(self.verticalLayoutWidget_5)
        self.box_timeline_layout.setObjectName(u"box_timeline_layout")
        self.box_timeline_layout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayoutWidget_6 = QWidget(self.box_left_low)
        self.verticalLayoutWidget_6.setObjectName(u"verticalLayoutWidget_6")
        self.verticalLayoutWidget_6.setGeometry(QRect(10, 10, 441, 141))
        self.verticalLayout_2 = QVBoxLayout(self.verticalLayoutWidget_6)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.label_5 = QLabel(self.verticalLayoutWidget_6)
        self.label_5.setObjectName(u"label_5")

        self.horizontalLayout_3.addWidget(self.label_5)

        self.dataset_filtering_nationality_cbox = QComboBox(self.verticalLayoutWidget_6)
        self.dataset_filtering_nationality_cbox.setObjectName(u"dataset_filtering_nationality_cbox")

        self.horizontalLayout_3.addWidget(self.dataset_filtering_nationality_cbox)


        self.verticalLayout_2.addLayout(self.horizontalLayout_3)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.label_8 = QLabel(self.verticalLayoutWidget_6)
        self.label_8.setObjectName(u"label_8")

        self.horizontalLayout_4.addWidget(self.label_8)

        self.dataset_filtering_media_cbox = QComboBox(self.verticalLayoutWidget_6)
        self.dataset_filtering_media_cbox.setObjectName(u"dataset_filtering_media_cbox")

        self.horizontalLayout_4.addWidget(self.dataset_filtering_media_cbox)


        self.verticalLayout_2.addLayout(self.horizontalLayout_4)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.label_3 = QLabel(self.verticalLayoutWidget_6)
        self.label_3.setObjectName(u"label_3")

        self.horizontalLayout_2.addWidget(self.label_3)

        self.dataset_filtering_from_date = QLineEdit(self.verticalLayoutWidget_6)
        self.dataset_filtering_from_date.setObjectName(u"dataset_filtering_from_date")

        self.horizontalLayout_2.addWidget(self.dataset_filtering_from_date)

        self.label_4 = QLabel(self.verticalLayoutWidget_6)
        self.label_4.setObjectName(u"label_4")

        self.horizontalLayout_2.addWidget(self.label_4)

        self.dataset_filtering_to_date = QLineEdit(self.verticalLayoutWidget_6)
        self.dataset_filtering_to_date.setObjectName(u"dataset_filtering_to_date")

        self.horizontalLayout_2.addWidget(self.dataset_filtering_to_date)


        self.verticalLayout_2.addLayout(self.horizontalLayout_2)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.apply_filters = QPushButton(self.verticalLayoutWidget_6)
        self.apply_filters.setObjectName(u"apply_filters")

        self.horizontalLayout.addWidget(self.apply_filters)

        self.reset_dataset_filters = QPushButton(self.verticalLayoutWidget_6)
        self.reset_dataset_filters.setObjectName(u"reset_dataset_filters")

        self.horizontalLayout.addWidget(self.reset_dataset_filters)

        self.reload_everything = QPushButton(self.verticalLayoutWidget_6)
        self.reload_everything.setObjectName(u"reload_everything")

        self.horizontalLayout.addWidget(self.reload_everything)

        self.label_10 = QLabel(self.verticalLayoutWidget_6)
        self.label_10.setObjectName(u"label_10")

        self.horizontalLayout.addWidget(self.label_10)

        self.filtered_dataset_size = QLabel(self.verticalLayoutWidget_6)
        self.filtered_dataset_size.setObjectName(u"filtered_dataset_size")

        self.horizontalLayout.addWidget(self.filtered_dataset_size)


        self.verticalLayout_2.addLayout(self.horizontalLayout)

        self.box_right_low = QWidget(self.centralwidget)
        self.box_right_low.setObjectName(u"box_right_low")
        self.box_right_low.setEnabled(True)
        self.box_right_low.setGeometry(QRect(1060, 420, 471, 341))
        self.statistics_tabs = QTabWidget(self.box_right_low)
        self.statistics_tabs.setObjectName(u"statistics_tabs")
        self.statistics_tabs.setGeometry(QRect(0, 0, 461, 341))
        self.statistics_tabs.setAutoFillBackground(True)
        self.statistics_tabs.setUsesScrollButtons(False)
        self.statistics_tabs.setTabsClosable(False)
        self.statistics_tabs.setTabBarAutoHide(False)
        self.style_stats = QWidget()
        self.style_stats.setObjectName(u"style_stats")
        self.verticalLayoutWidget_4 = QWidget(self.style_stats)
        self.verticalLayoutWidget_4.setObjectName(u"verticalLayoutWidget_4")
        self.verticalLayoutWidget_4.setGeometry(QRect(0, 0, 461, 311))
        self.style_stats_layout = QVBoxLayout(self.verticalLayoutWidget_4)
        self.style_stats_layout.setObjectName(u"style_stats_layout")
        self.style_stats_layout.setContentsMargins(0, 0, 0, 0)
        self.statistics_tabs.addTab(self.style_stats, "")
        self.date_stats = QWidget()
        self.date_stats.setObjectName(u"date_stats")
        self.verticalLayoutWidget_8 = QWidget(self.date_stats)
        self.verticalLayoutWidget_8.setObjectName(u"verticalLayoutWidget_8")
        self.verticalLayoutWidget_8.setGeometry(QRect(0, 0, 461, 311))
        self.date_stats_layout = QVBoxLayout(self.verticalLayoutWidget_8)
        self.date_stats_layout.setObjectName(u"date_stats_layout")
        self.date_stats_layout.setContentsMargins(0, 0, 0, 0)
        self.statistics_tabs.addTab(self.date_stats, "")
        self.nationality_stats = QWidget()
        self.nationality_stats.setObjectName(u"nationality_stats")
        self.verticalLayoutWidget_7 = QWidget(self.nationality_stats)
        self.verticalLayoutWidget_7.setObjectName(u"verticalLayoutWidget_7")
        self.verticalLayoutWidget_7.setGeometry(QRect(-1, 0, 461, 311))
        self.nationality_stats_layout = QVBoxLayout(self.verticalLayoutWidget_7)
        self.nationality_stats_layout.setObjectName(u"nationality_stats_layout")
        self.nationality_stats_layout.setContentsMargins(0, 0, 0, 0)
        self.statistics_tabs.addTab(self.nationality_stats, "")
        self.box_metainfo = QWidget(self.centralwidget)
        self.box_metainfo.setObjectName(u"box_metainfo")
        self.box_metainfo.setGeometry(QRect(10, 370, 461, 80))
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
        self.box_left_img.setGeometry(QRect(10, 10, 461, 311))
        self.box_left_img.setAutoFillBackground(True)
        self.horizontalLayoutWidget = QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setObjectName(u"horizontalLayoutWidget")
        self.horizontalLayoutWidget.setGeometry(QRect(480, 420, 573, 31))
        self.scatterplot_options = QHBoxLayout(self.horizontalLayoutWidget)
        self.scatterplot_options.setObjectName(u"scatterplot_options")
        self.scatterplot_options.setContentsMargins(0, 0, 0, 0)
        self.subset_projection_btn = QPushButton(self.horizontalLayoutWidget)
        self.subset_projection_btn.setObjectName(u"subset_projection_btn")

        self.scatterplot_options.addWidget(self.subset_projection_btn)

        self.r_image_points = QRadioButton(self.horizontalLayoutWidget)
        self.r_image_points.setObjectName(u"r_image_points")

        self.scatterplot_options.addWidget(self.r_image_points)

        self.scatterplot_frame = PlotWidget(self.centralwidget)
        self.scatterplot_frame.setObjectName(u"scatterplot_frame")
        self.scatterplot_frame.setGeometry(QRect(480, 10, 571, 401))
        self.box_metric_tabs = QTabWidget(self.centralwidget)
        self.box_metric_tabs.setObjectName(u"box_metric_tabs")
        self.box_metric_tabs.setGeometry(QRect(480, 460, 571, 301))
        self.box_metric_tabs.setAutoFillBackground(True)
        self.dino_tab = QWidget()
        self.dino_tab.setObjectName(u"dino_tab")
        self.line_2 = QFrame(self.dino_tab)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setGeometry(QRect(10, 60, 501, 20))
        self.line_2.setFrameShape(QFrame.Shape.HLine)
        self.line_2.setFrameShadow(QFrame.Shadow.Sunken)
        self.line_3 = QFrame(self.dino_tab)
        self.line_3.setObjectName(u"line_3")
        self.line_3.setGeometry(QRect(10, 120, 501, 16))
        self.line_3.setFrameShape(QFrame.Shape.HLine)
        self.line_3.setFrameShadow(QFrame.Shadow.Sunken)
        self.dino_opts1 = QGroupBox(self.dino_tab)
        self.dino_opts1.setObjectName(u"dino_opts1")
        self.dino_opts1.setGeometry(QRect(10, 10, 501, 51))
        self.dinolabl2 = QLabel(self.dino_opts1)
        self.dinolabl2.setObjectName(u"dinolabl2")
        self.dinolabl2.setGeometry(QRect(370, 10, 41, 31))
        self.dino_opt_layervis_cbox = QComboBox(self.dino_opts1)
        self.dino_opt_layervis_cbox.setObjectName(u"dino_opt_layervis_cbox")
        self.dino_opt_layervis_cbox.setGeometry(QRect(290, 10, 61, 31))
        self.dinolabel = QLabel(self.dino_opts1)
        self.dinolabel.setObjectName(u"dinolabel")
        self.dinolabel.setGeometry(QRect(230, 10, 51, 31))
        self.dino_opt_headvis_cbox = QComboBox(self.dino_opts1)
        self.dino_opt_headvis_cbox.setObjectName(u"dino_opt_headvis_cbox")
        self.dino_opt_headvis_cbox.setGeometry(QRect(420, 10, 61, 31))
        self.dino_opt_showcamap = QRadioButton(self.dino_opts1)
        self.dino_opt_showcamap.setObjectName(u"dino_opt_showcamap")
        self.dino_opt_showcamap.setGeometry(QRect(10, 10, 211, 31))
        self.dino_opts2 = QGroupBox(self.dino_tab)
        self.dino_opts2.setObjectName(u"dino_opts2")
        self.dino_opts2.setGeometry(QRect(10, 80, 501, 41))
        self.dino_opt_eucdist = QRadioButton(self.dino_opts2)
        self.dino_opt_eucdist.setObjectName(u"dino_opt_eucdist")
        self.dino_opt_eucdist.setGeometry(QRect(290, 10, 161, 31))
        self.dino_opt_cosdist = QRadioButton(self.dino_opts2)
        self.dino_opt_cosdist.setObjectName(u"dino_opt_cosdist")
        self.dino_opt_cosdist.setGeometry(QRect(150, 10, 131, 31))
        self.label = QLabel(self.dino_opts2)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(30, 10, 111, 31))
        self.label.setWordWrap(False)
        self.dino_opts3 = QGroupBox(self.dino_tab)
        self.dino_opts3.setObjectName(u"dino_opts3")
        self.dino_opts3.setGeometry(QRect(10, 130, 501, 121))
        self.dino_opt_headsim = QRadioButton(self.dino_opts3)
        self.dino_opt_headsim.setObjectName(u"dino_opt_headsim")
        self.dino_opt_headsim.setGeometry(QRect(10, 80, 231, 31))
        self.dino_opt_headsim_cbox = QComboBox(self.dino_opts3)
        self.dino_opt_headsim_cbox.setObjectName(u"dino_opt_headsim_cbox")
        self.dino_opt_headsim_cbox.setGeometry(QRect(290, 80, 61, 31))
        self.dino_opt_fullsim = QRadioButton(self.dino_opts3)
        self.dino_opt_fullsim.setObjectName(u"dino_opt_fullsim")
        self.dino_opt_fullsim.setGeometry(QRect(10, 30, 251, 31))
        self.dino_opt_2dsim = QRadioButton(self.dino_opts3)
        self.dino_opt_2dsim.setObjectName(u"dino_opt_2dsim")
        self.dino_opt_2dsim.setGeometry(QRect(290, 30, 200, 31))
        self.dinolabl2_2 = QLabel(self.dino_opts3)
        self.dinolabl2_2.setObjectName(u"dinolabl2_2")
        self.dinolabl2_2.setGeometry(QRect(360, 80, 41, 31))
        self.dino_opt_layersim_cbox = QComboBox(self.dino_opts3)
        self.dino_opt_layersim_cbox.setObjectName(u"dino_opt_layersim_cbox")
        self.dino_opt_layersim_cbox.setGeometry(QRect(420, 80, 61, 31))
        self.dinolabel_2 = QLabel(self.dino_opts3)
        self.dinolabel_2.setObjectName(u"dinolabel_2")
        self.dinolabel_2.setGeometry(QRect(240, 80, 51, 31))
        self.box_metric_tabs.addTab(self.dino_tab, "")
        self.texture_tab = QWidget()
        self.texture_tab.setObjectName(u"texture_tab")
        self.texture_opts_1 = QGroupBox(self.texture_tab)
        self.texture_opts_1.setObjectName(u"texture_opts_1")
        self.texture_opts_1.setGeometry(QRect(10, 10, 501, 51))
        self.texture_opt_filtervis = QComboBox(self.texture_opts_1)
        self.texture_opt_filtervis.setObjectName(u"texture_opt_filtervis")
        self.texture_opt_filtervis.setGeometry(QRect(290, 10, 191, 31))
        self.texture_label1 = QLabel(self.texture_opts_1)
        self.texture_label1.setObjectName(u"texture_label1")
        self.texture_label1.setGeometry(QRect(200, 10, 81, 31))
        self.texture_opt_show_fm = QRadioButton(self.texture_opts_1)
        self.texture_opt_show_fm.setObjectName(u"texture_opt_show_fm")
        self.texture_opt_show_fm.setGeometry(QRect(10, 10, 211, 31))
        self.texture_opts_2 = QGroupBox(self.texture_tab)
        self.texture_opts_2.setObjectName(u"texture_opts_2")
        self.texture_opts_2.setGeometry(QRect(10, 80, 501, 41))
        self.texture_opt_eucdist = QRadioButton(self.texture_opts_2)
        self.texture_opt_eucdist.setObjectName(u"texture_opt_eucdist")
        self.texture_opt_eucdist.setGeometry(QRect(290, 10, 161, 31))
        self.texture_opt_cosdist = QRadioButton(self.texture_opts_2)
        self.texture_opt_cosdist.setObjectName(u"texture_opt_cosdist")
        self.texture_opt_cosdist.setGeometry(QRect(150, 10, 131, 31))
        self.label_6 = QLabel(self.texture_opts_2)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setGeometry(QRect(30, 10, 111, 31))
        self.label_6.setWordWrap(False)
        self.line_4 = QFrame(self.texture_tab)
        self.line_4.setObjectName(u"line_4")
        self.line_4.setGeometry(QRect(10, 60, 501, 20))
        self.line_4.setFrameShape(QFrame.Shape.HLine)
        self.line_4.setFrameShadow(QFrame.Shadow.Sunken)
        self.groupBox_5 = QGroupBox(self.texture_tab)
        self.groupBox_5.setObjectName(u"groupBox_5")
        self.groupBox_5.setGeometry(QRect(10, 150, 501, 71))
        self.texture_opt_fullsim = QRadioButton(self.groupBox_5)
        self.texture_opt_fullsim.setObjectName(u"texture_opt_fullsim")
        self.texture_opt_fullsim.setGeometry(QRect(10, 30, 251, 31))
        self.texture_opt_2dsim = QRadioButton(self.groupBox_5)
        self.texture_opt_2dsim.setObjectName(u"texture_opt_2dsim")
        self.texture_opt_2dsim.setGeometry(QRect(240, 30, 200, 31))
        self.box_metric_tabs.addTab(self.texture_tab, "")
        self.emotion_tab = QWidget()
        self.emotion_tab.setObjectName(u"emotion_tab")
        self.emotion_opts1 = QGroupBox(self.emotion_tab)
        self.emotion_opts1.setObjectName(u"emotion_opts1")
        self.emotion_opts1.setGeometry(QRect(10, 10, 501, 51))
        self.emotion_opts_showfm = QRadioButton(self.emotion_opts1)
        self.emotion_opts_showfm.setObjectName(u"emotion_opts_showfm")
        self.emotion_opts_showfm.setGeometry(QRect(10, 10, 161, 31))
        self.emotion_opts2 = QGroupBox(self.emotion_tab)
        self.emotion_opts2.setObjectName(u"emotion_opts2")
        self.emotion_opts2.setGeometry(QRect(10, 80, 501, 41))
        self.label_7 = QLabel(self.emotion_opts2)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setGeometry(QRect(30, 10, 111, 31))
        self.label_7.setWordWrap(False)
        self.emotion_opt_cosdist = QRadioButton(self.emotion_opts2)
        self.emotion_opt_cosdist.setObjectName(u"emotion_opt_cosdist")
        self.emotion_opt_cosdist.setGeometry(QRect(150, 10, 131, 31))
        self.emotion_opt_eucdist = QRadioButton(self.emotion_opts2)
        self.emotion_opt_eucdist.setObjectName(u"emotion_opt_eucdist")
        self.emotion_opt_eucdist.setGeometry(QRect(300, 10, 161, 31))
        self.groupBox_2 = QGroupBox(self.emotion_tab)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.groupBox_2.setGeometry(QRect(10, 150, 501, 71))
        self.emotion_opt_fullsim = QRadioButton(self.groupBox_2)
        self.emotion_opt_fullsim.setObjectName(u"emotion_opt_fullsim")
        self.emotion_opt_fullsim.setGeometry(QRect(10, 30, 251, 31))
        self.emotion_opt_2dsim = QRadioButton(self.groupBox_2)
        self.emotion_opt_2dsim.setObjectName(u"emotion_opt_2dsim")
        self.emotion_opt_2dsim.setGeometry(QRect(240, 30, 200, 31))
        self.line_5 = QFrame(self.emotion_tab)
        self.line_5.setObjectName(u"line_5")
        self.line_5.setGeometry(QRect(10, 60, 501, 20))
        self.line_5.setFrameShape(QFrame.Shape.HLine)
        self.line_5.setFrameShadow(QFrame.Shadow.Sunken)
        self.box_metric_tabs.addTab(self.emotion_tab, "")
        self.semantic_tab = QWidget()
        self.semantic_tab.setObjectName(u"semantic_tab")
        self.emotion_opts2_2 = QGroupBox(self.semantic_tab)
        self.emotion_opts2_2.setObjectName(u"emotion_opts2_2")
        self.emotion_opts2_2.setGeometry(QRect(10, 80, 501, 41))
        self.label_9 = QLabel(self.emotion_opts2_2)
        self.label_9.setObjectName(u"label_9")
        self.label_9.setGeometry(QRect(30, 10, 111, 31))
        self.label_9.setWordWrap(False)
        self.semantic_opt_cosdist = QRadioButton(self.emotion_opts2_2)
        self.semantic_opt_cosdist.setObjectName(u"semantic_opt_cosdist")
        self.semantic_opt_cosdist.setGeometry(QRect(150, 10, 131, 31))
        self.semantic_opt_eucdist = QRadioButton(self.emotion_opts2_2)
        self.semantic_opt_eucdist.setObjectName(u"semantic_opt_eucdist")
        self.semantic_opt_eucdist.setGeometry(QRect(300, 10, 161, 31))
        self.line_6 = QFrame(self.semantic_tab)
        self.line_6.setObjectName(u"line_6")
        self.line_6.setGeometry(QRect(10, 60, 501, 20))
        self.line_6.setFrameShape(QFrame.Shape.HLine)
        self.line_6.setFrameShadow(QFrame.Shadow.Sunken)
        self.groupBox_3 = QGroupBox(self.semantic_tab)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.groupBox_3.setGeometry(QRect(10, 150, 501, 71))
        self.semantic_opt_fullsim = QRadioButton(self.groupBox_3)
        self.semantic_opt_fullsim.setObjectName(u"semantic_opt_fullsim")
        self.semantic_opt_fullsim.setGeometry(QRect(10, 30, 251, 31))
        self.semantic_opt_2dsim = QRadioButton(self.groupBox_3)
        self.semantic_opt_2dsim.setObjectName(u"semantic_opt_2dsim")
        self.semantic_opt_2dsim.setGeometry(QRect(240, 30, 200, 31))
        self.box_metric_tabs.addTab(self.semantic_tab, "")
        self.clip_tab = QWidget()
        self.clip_tab.setObjectName(u"clip_tab")
        self.verticalLayoutWidget = QWidget(self.clip_tab)
        self.verticalLayoutWidget.setObjectName(u"verticalLayoutWidget")
        self.verticalLayoutWidget.setGeometry(QRect(20, 80, 501, 111))
        self.Clip_input_layout = QVBoxLayout(self.verticalLayoutWidget)
        self.Clip_input_layout.setObjectName(u"Clip_input_layout")
        self.Clip_input_layout.setContentsMargins(0, 0, 0, 0)
        self.label_2 = QLabel(self.verticalLayoutWidget)
        self.label_2.setObjectName(u"label_2")

        self.Clip_input_layout.addWidget(self.label_2)

        self.tb_clip_input = QPlainTextEdit(self.verticalLayoutWidget)
        self.tb_clip_input.setObjectName(u"tb_clip_input")
        self.tb_clip_input.setCursorWidth(1)

        self.Clip_input_layout.addWidget(self.tb_clip_input)

        self.horizontalLayoutWidget_2 = QWidget(self.clip_tab)
        self.horizontalLayoutWidget_2.setObjectName(u"horizontalLayoutWidget_2")
        self.horizontalLayoutWidget_2.setGeometry(QRect(20, 20, 501, 51))
        self.horizontalLayout_5 = QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.clip_radio_combsim = QRadioButton(self.horizontalLayoutWidget_2)
        self.clip_radio_combsim.setObjectName(u"clip_radio_combsim")

        self.horizontalLayout_5.addWidget(self.clip_radio_combsim)

        self.clip_radio_imgsim = QRadioButton(self.horizontalLayoutWidget_2)
        self.clip_radio_imgsim.setObjectName(u"clip_radio_imgsim")

        self.horizontalLayout_5.addWidget(self.clip_radio_imgsim)

        self.clip_radio_textsim = QRadioButton(self.horizontalLayoutWidget_2)
        self.clip_radio_textsim.setObjectName(u"clip_radio_textsim")

        self.horizontalLayout_5.addWidget(self.clip_radio_textsim)

        self.groupBox_4 = QGroupBox(self.clip_tab)
        self.groupBox_4.setObjectName(u"groupBox_4")
        self.groupBox_4.setGeometry(QRect(20, 200, 501, 51))
        self.clip_opt_fullsim = QRadioButton(self.groupBox_4)
        self.clip_opt_fullsim.setObjectName(u"clip_opt_fullsim")
        self.clip_opt_fullsim.setGeometry(QRect(10, 10, 251, 31))
        self.clip_opt_2dsim = QRadioButton(self.groupBox_4)
        self.clip_opt_2dsim.setObjectName(u"clip_opt_2dsim")
        self.clip_opt_2dsim.setGeometry(QRect(240, 10, 200, 31))
        self.box_metric_tabs.addTab(self.clip_tab, "")
        self.combined_tab = QWidget()
        self.combined_tab.setObjectName(u"combined_tab")
        self.formLayoutWidget = QWidget(self.combined_tab)
        self.formLayoutWidget.setObjectName(u"formLayoutWidget")
        self.formLayoutWidget.setGeometry(QRect(50, 50, 241, 151))
        self.formLayout = QFormLayout(self.formLayoutWidget)
        self.formLayout.setObjectName(u"formLayout")
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.label_dino = QLabel(self.formLayoutWidget)
        self.label_dino.setObjectName(u"label_dino")

        self.formLayout.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_dino)

        self.label_texture = QLabel(self.formLayoutWidget)
        self.label_texture.setObjectName(u"label_texture")

        self.formLayout.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_texture)

        self.combo_texture_slider = QSlider(self.formLayoutWidget)
        self.combo_texture_slider.setObjectName(u"combo_texture_slider")
        self.combo_texture_slider.setMaximum(100)
        self.combo_texture_slider.setSliderPosition(100)
        self.combo_texture_slider.setOrientation(Qt.Orientation.Horizontal)

        self.formLayout.setWidget(1, QFormLayout.ItemRole.FieldRole, self.combo_texture_slider)

        self.label_dummy = QLabel(self.formLayoutWidget)
        self.label_dummy.setObjectName(u"label_dummy")

        self.formLayout.setWidget(2, QFormLayout.ItemRole.LabelRole, self.label_dummy)

        self.combo_emotion_slider = QSlider(self.formLayoutWidget)
        self.combo_emotion_slider.setObjectName(u"combo_emotion_slider")
        self.combo_emotion_slider.setMaximum(100)
        self.combo_emotion_slider.setValue(100)
        self.combo_emotion_slider.setOrientation(Qt.Orientation.Horizontal)

        self.formLayout.setWidget(2, QFormLayout.ItemRole.FieldRole, self.combo_emotion_slider)

        self.label_dummy_2 = QLabel(self.formLayoutWidget)
        self.label_dummy_2.setObjectName(u"label_dummy_2")

        self.formLayout.setWidget(3, QFormLayout.ItemRole.LabelRole, self.label_dummy_2)

        self.combo_clip_slider = QSlider(self.formLayoutWidget)
        self.combo_clip_slider.setObjectName(u"combo_clip_slider")
        self.combo_clip_slider.setMaximum(100)
        self.combo_clip_slider.setValue(100)
        self.combo_clip_slider.setOrientation(Qt.Orientation.Horizontal)

        self.formLayout.setWidget(3, QFormLayout.ItemRole.FieldRole, self.combo_clip_slider)

        self.label_dummy_3 = QLabel(self.formLayoutWidget)
        self.label_dummy_3.setObjectName(u"label_dummy_3")

        self.formLayout.setWidget(4, QFormLayout.ItemRole.LabelRole, self.label_dummy_3)

        self.combo_semantic_slider = QSlider(self.formLayoutWidget)
        self.combo_semantic_slider.setObjectName(u"combo_semantic_slider")
        self.combo_semantic_slider.setMaximum(100)
        self.combo_semantic_slider.setValue(100)
        self.combo_semantic_slider.setOrientation(Qt.Orientation.Horizontal)

        self.formLayout.setWidget(4, QFormLayout.ItemRole.FieldRole, self.combo_semantic_slider)

        self.combo_dino_slider = QSlider(self.formLayoutWidget)
        self.combo_dino_slider.setObjectName(u"combo_dino_slider")
        self.combo_dino_slider.setMaximum(100)
        self.combo_dino_slider.setSingleStep(1)
        self.combo_dino_slider.setValue(100)
        self.combo_dino_slider.setOrientation(Qt.Orientation.Horizontal)

        self.formLayout.setWidget(0, QFormLayout.ItemRole.FieldRole, self.combo_dino_slider)

        self.title = QLabel(self.combined_tab)
        self.title.setObjectName(u"title")
        self.title.setGeometry(QRect(30, 20, 269, 21))
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.recalc_similarity = QPushButton(self.combined_tab)
        self.recalc_similarity.setObjectName(u"recalc_similarity")
        self.recalc_similarity.setGeometry(QRect(320, 120, 231, 31))
        self.combined_projection_btn = QPushButton(self.combined_tab)
        self.combined_projection_btn.setObjectName(u"combined_projection_btn")
        self.combined_projection_btn.setGeometry(QRect(320, 80, 231, 31))
        self.box_metric_tabs.addTab(self.combined_tab, "")
        self.groupBox = QGroupBox(self.centralwidget)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setGeometry(QRect(1060, 10, 461, 401))
        self.box_right_img = QLabel(self.groupBox)
        self.box_right_img.setObjectName(u"box_right_img")
        self.box_right_img.setGeometry(QRect(10, 20, 441, 271))
        self.box_right_img.setAutoFillBackground(True)
        self.box_right_img.setLineWidth(1)
        self.box_right_img.setMargin(0)
        self.n2 = QLabel(self.groupBox)
        self.n2.setObjectName(u"n2")
        self.n2.setGeometry(QRect(160, 300, 141, 99))
        self.n2.setAutoFillBackground(True)
        self.n3 = QLabel(self.groupBox)
        self.n3.setObjectName(u"n3")
        self.n3.setGeometry(QRect(310, 300, 141, 99))
        self.n3.setAutoFillBackground(True)
        self.n1 = QLabel(self.groupBox)
        self.n1.setObjectName(u"n1")
        self.n1.setGeometry(QRect(10, 300, 141, 99))
        self.n1.setAutoFillBackground(True)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        self.statistics_tabs.setCurrentIndex(0)
        self.box_metric_tabs.setCurrentIndex(5)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Deep Similarity Analyzer", None))
#if QT_CONFIG(tooltip)
        self.b_upload.setToolTip(QCoreApplication.translate("MainWindow", u"Upload a new image to analyse", None))
#endif // QT_CONFIG(tooltip)
        self.b_upload.setText(QCoreApplication.translate("MainWindow", u"Upload Image", None))
        self.l_timeline.setText(QCoreApplication.translate("MainWindow", u"Timeline for above image", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"nationality", None))
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"material type", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"date range", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"to", None))
        self.apply_filters.setText(QCoreApplication.translate("MainWindow", u"Apply", None))
        self.reset_dataset_filters.setText(QCoreApplication.translate("MainWindow", u"Reset filter", None))
        self.reload_everything.setText(QCoreApplication.translate("MainWindow", u"Reload data", None))
        self.label_10.setText(QCoreApplication.translate("MainWindow", u"filtered datset size: ", None))
        self.filtered_dataset_size.setText(QCoreApplication.translate("MainWindow", u"INT", None))
        self.statistics_tabs.setTabText(self.statistics_tabs.indexOf(self.style_stats), QCoreApplication.translate("MainWindow", u"Style statistics", None))
        self.statistics_tabs.setTabText(self.statistics_tabs.indexOf(self.date_stats), QCoreApplication.translate("MainWindow", u"Date statistics", None))
        self.statistics_tabs.setTabText(self.statistics_tabs.indexOf(self.nationality_stats), QCoreApplication.translate("MainWindow", u"Nationality statistics", None))
        self.t_artist.setText(QCoreApplication.translate("MainWindow", u"Artist: ", None))
        self.t_date.setText(QCoreApplication.translate("MainWindow", u"Date:", None))
        self.t_style.setText(QCoreApplication.translate("MainWindow", u"Style:", None))
        self.t_tags.setText(QCoreApplication.translate("MainWindow", u"Tags:", None))
        self.box_left_img.setText("")
        self.subset_projection_btn.setText(QCoreApplication.translate("MainWindow", u"Reproject current", None))
        self.r_image_points.setText(QCoreApplication.translate("MainWindow", u"points as images", None))
        self.dino_opts1.setTitle("")
        self.dinolabl2.setText(QCoreApplication.translate("MainWindow", u"head:", None))
        self.dinolabel.setText(QCoreApplication.translate("MainWindow", u"layer:", None))
        self.dino_opt_showcamap.setText(QCoreApplication.translate("MainWindow", u"show crossattention map", None))
        self.dino_opts2.setTitle("")
        self.dino_opt_eucdist.setText(QCoreApplication.translate("MainWindow", u"euclidian distance", None))
        self.dino_opt_cosdist.setText(QCoreApplication.translate("MainWindow", u"cosine distance", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"distance metric:", None))
        self.dino_opts3.setTitle(QCoreApplication.translate("MainWindow", u"Adjust the recommended images using these features", None))
        self.dino_opt_headsim.setText(QCoreApplication.translate("MainWindow", u"similarity from specific head", None))
        self.dino_opt_fullsim.setText(QCoreApplication.translate("MainWindow", u"full feature vector similarity", None))
        self.dino_opt_2dsim.setText(QCoreApplication.translate("MainWindow", u"2d reprojection similarity", None))
        self.dinolabl2_2.setText(QCoreApplication.translate("MainWindow", u"head:", None))
        self.dinolabel_2.setText(QCoreApplication.translate("MainWindow", u"layer:", None))
        self.box_metric_tabs.setTabText(self.box_metric_tabs.indexOf(self.dino_tab), QCoreApplication.translate("MainWindow", u"Dino", None))
        self.texture_opts_1.setTitle("")
        self.texture_label1.setText(QCoreApplication.translate("MainWindow", u"filter index", None))
        self.texture_opt_show_fm.setText(QCoreApplication.translate("MainWindow", u"show feature map", None))
        self.texture_opts_2.setTitle("")
        self.texture_opt_eucdist.setText(QCoreApplication.translate("MainWindow", u"euclidian distance", None))
        self.texture_opt_cosdist.setText(QCoreApplication.translate("MainWindow", u"cosine distance", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"distance metric:", None))
        self.groupBox_5.setTitle("")
        self.texture_opt_fullsim.setText(QCoreApplication.translate("MainWindow", u"full feature vector similarity", None))
        self.texture_opt_2dsim.setText(QCoreApplication.translate("MainWindow", u"2d reprojection similarity", None))
        self.box_metric_tabs.setTabText(self.box_metric_tabs.indexOf(self.texture_tab), QCoreApplication.translate("MainWindow", u"Texture", None))
        self.emotion_opts1.setTitle("")
        self.emotion_opts_showfm.setText(QCoreApplication.translate("MainWindow", u"show feature map", None))
        self.emotion_opts2.setTitle("")
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"distance metric:", None))
        self.emotion_opt_cosdist.setText(QCoreApplication.translate("MainWindow", u"cosine distance", None))
        self.emotion_opt_eucdist.setText(QCoreApplication.translate("MainWindow", u"euclidian distance", None))
        self.groupBox_2.setTitle("")
        self.emotion_opt_fullsim.setText(QCoreApplication.translate("MainWindow", u"full feature vector similarity", None))
        self.emotion_opt_2dsim.setText(QCoreApplication.translate("MainWindow", u"2d reprojection similarity", None))
        self.box_metric_tabs.setTabText(self.box_metric_tabs.indexOf(self.emotion_tab), QCoreApplication.translate("MainWindow", u"Emotion", None))
        self.emotion_opts2_2.setTitle("")
        self.label_9.setText(QCoreApplication.translate("MainWindow", u"distance metric:", None))
        self.semantic_opt_cosdist.setText(QCoreApplication.translate("MainWindow", u"cosine distance", None))
        self.semantic_opt_eucdist.setText(QCoreApplication.translate("MainWindow", u"euclidian distance", None))
        self.groupBox_3.setTitle("")
        self.semantic_opt_fullsim.setText(QCoreApplication.translate("MainWindow", u"full feature vector similarity", None))
        self.semantic_opt_2dsim.setText(QCoreApplication.translate("MainWindow", u"2d reprojection similarity", None))
        self.box_metric_tabs.setTabText(self.box_metric_tabs.indexOf(self.semantic_tab), QCoreApplication.translate("MainWindow", u"Semantic", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"Describe similarity aspect to a Clip model", None))
        self.tb_clip_input.setPlaceholderText(QCoreApplication.translate("MainWindow", u"image attribute to compare with.", None))
        self.clip_radio_combsim.setText(QCoreApplication.translate("MainWindow", u"combined sim", None))
        self.clip_radio_imgsim.setText(QCoreApplication.translate("MainWindow", u"image embedding sim", None))
        self.clip_radio_textsim.setText(QCoreApplication.translate("MainWindow", u"text sim", None))
        self.groupBox_4.setTitle("")
        self.clip_opt_fullsim.setText(QCoreApplication.translate("MainWindow", u"full feature vector similarity", None))
        self.clip_opt_2dsim.setText(QCoreApplication.translate("MainWindow", u"2d reprojection similarity", None))
        self.box_metric_tabs.setTabText(self.box_metric_tabs.indexOf(self.clip_tab), QCoreApplication.translate("MainWindow", u"Clip", None))
        self.label_dino.setText(QCoreApplication.translate("MainWindow", u"Dino", None))
        self.label_texture.setText(QCoreApplication.translate("MainWindow", u"Texture", None))
        self.label_dummy.setText(QCoreApplication.translate("MainWindow", u"Emotion", None))
        self.label_dummy_2.setText(QCoreApplication.translate("MainWindow", u"Clip", None))
        self.label_dummy_3.setText(QCoreApplication.translate("MainWindow", u"Semantic", None))
#if QT_CONFIG(tooltip)
        self.combo_dino_slider.setToolTip("")
#endif // QT_CONFIG(tooltip)
        self.title.setText(QCoreApplication.translate("MainWindow", u"Metric weights", None))
        self.recalc_similarity.setText(QCoreApplication.translate("MainWindow", u"Recalculate similarity", None))
        self.combined_projection_btn.setText(QCoreApplication.translate("MainWindow", u"Calc combined projection", None))
        self.box_metric_tabs.setTabText(self.box_metric_tabs.indexOf(self.combined_tab), QCoreApplication.translate("MainWindow", u"Combined", None))
        self.groupBox.setTitle(QCoreApplication.translate("MainWindow", u"Similar Images", None))
        self.box_right_img.setText("")
        self.n2.setText("")
        self.n3.setText("")
        self.n1.setText("")
    # retranslateUi

        """
        ----------------------------------------------------------------------
                                All Tooltips text
        ----------------------------------------------------------------------
        """
        distance_metric = "Choose how the nearest neighbour points are calculated."
        reprojection_similarity_full = "Use vector from original multidimensional featurespace."
        reprojection_similarity_2d = "Use projected feature vectors to find similarities."
        
        # General UI tooltips
        self.subset_projection_btn.setToolTip("Placeholder0") # Reproject current
        self.b_upload.setToolTip("Find similarity of your own image. This may take a few minutes.") # Upload button
        self.reset_dataset_filters.setToolTip("Set all filters to default.") # Reset filter
        self.reload_everything.setToolTip("Deletes all images that are uploaded.") # Reload Data
        self.l_timeline.setToolTip("Get related works based on the Artistic Visual Story dataset.") # Timeline for above image

        # Dino tab tooltips
        self.dino_opt_showcamap.setToolTip("Overlay image with a heatmap indicating the sections the model is focussing on.") # Show crossattention map
        self.label.setToolTip(distance_metric) # Distance metric
        self.dino_opt_fullsim.setToolTip(reprojection_similarity_full) # Full feature vector similarity
        self.dino_opt_2dsim.setToolTip(reprojection_similarity_2d) # 2d reprojection similarity
        self.dino_opt_headsim.setToolTip("Select ") # Similarity from specific head

        # Texture tab tooltips
        self.label_6.setToolTip(distance_metric) # Distance metric
        self.texture_opt_show_fm.setToolTip("Placeholder5") # Show feature map
        self.texture_label1.setToolTip("Placeholder6") # Filter index
        self.texture_opt_fullsim.setToolTip(reprojection_similarity_full) # Full feature vector similarity
        self.texture_opt_2dsim.setToolTip(reprojection_similarity_2d) # 2d reprojection similarity

        # Emotion tab tooltips
        self.emotion_opts_showfm.setToolTip("Placeholder9") # Show feature map
        self.emotion_opt_fullsim.setToolTip(reprojection_similarity_full) # Full feature vector similarity
        self.emotion_opt_2dsim.setToolTip(reprojection_similarity_2d) # 2d reprojection similarity

        # Semantic tab tooltips
        self.semantic_opt_fullsim.setToolTip(reprojection_similarity_full) # Full feature vector similarity
        self.semantic_opt_2dsim.setToolTip(reprojection_similarity_2d) # 2d reprojection similarity

        # CLIP tab tooltips
        self.clip_radio_combsim.setToolTip("Placeholder14") # Combined sim
        self.clip_radio_imgsim.setToolTip("Placholder15") # Image embedding sim
        self.clip_radio_textsim.setToolTip("Placeholder16") # Text sim
        self.label_2.setToolTip("Write down your own features of interest. These will be incorpororated when calculating the similarity.") # Describe similarity aspect to a Clip model
        self.clip_opt_fullsim.setToolTip(reprojection_similarity_full) # Full feature vector similarity
        self.clip_opt_2dsim.setToolTip(reprojection_similarity_2d) # 2d reprojection similarity

        # Combined tab tooltips
        self.combined_projection_btn.setToolTip("Placeholder19") # Calc combined projection
        self.recalc_similarity.setToolTip("Placeholder20") # Recalculate similarity
        self.title.setToolTip("Adjust the weight of individual metrics. The feature embeddings will be scaled accordingly.") # Metric weights
