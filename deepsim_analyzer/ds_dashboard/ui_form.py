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
from PyQt6.QtWidgets import (QApplication, QComboBox, QFrame, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QPushButton, QRadioButton, QSizePolicy, QSlider,
    QTabWidget, QWidget)

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
        self.box_metric_tabs.setGeometry(QRect(500, 460, 511, 261))
        self.dino_tab = QWidget()
        self.dino_tab.setObjectName(u"dino_tab")
        self.line_2 = QFrame(self.dino_tab)
        self.line_2.setObjectName(u"line_2")
        self.line_2.setGeometry(QRect(10, 50, 491, 20))
        self.line_2.setFrameShape(QFrame.Shape.HLine)
        self.line_2.setFrameShadow(QFrame.Shadow.Sunken)
        self.line_3 = QFrame(self.dino_tab)
        self.line_3.setObjectName(u"line_3")
        self.line_3.setGeometry(QRect(10, 100, 491, 16))
        self.line_3.setFrameShape(QFrame.Shape.HLine)
        self.line_3.setFrameShadow(QFrame.Shadow.Sunken)
        self.dino_opts1 = QGroupBox(self.dino_tab)
        self.dino_opts1.setObjectName(u"dino_opts1")
        self.dino_opts1.setGeometry(QRect(10, 0, 491, 51))
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
        self.dino_opts2.setGeometry(QRect(10, 60, 491, 41))
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
        self.dino_opts3.setGeometry(QRect(10, 110, 491, 111))
        self.dino_opt_headsim = QRadioButton(self.dino_opts3)
        self.dino_opt_headsim.setObjectName(u"dino_opt_headsim")
        self.dino_opt_headsim.setGeometry(QRect(10, 70, 231, 31))
        self.dino_opt_headsim_cbox = QComboBox(self.dino_opts3)
        self.dino_opt_headsim_cbox.setObjectName(u"dino_opt_headsim_cbox")
        self.dino_opt_headsim_cbox.setGeometry(QRect(290, 70, 61, 31))
        self.dino_opt_fullsim = QRadioButton(self.dino_opts3)
        self.dino_opt_fullsim.setObjectName(u"dino_opt_fullsim")
        self.dino_opt_fullsim.setGeometry(QRect(10, 20, 251, 31))
        self.dino_opt_2dsim = QRadioButton(self.dino_opts3)
        self.dino_opt_2dsim.setObjectName(u"dino_opt_2dsim")
        self.dino_opt_2dsim.setGeometry(QRect(290, 20, 200, 31))
        self.dinolabl2_2 = QLabel(self.dino_opts3)
        self.dinolabl2_2.setObjectName(u"dinolabl2_2")
        self.dinolabl2_2.setGeometry(QRect(360, 70, 41, 31))
        self.dino_opt_layersim_cbox = QComboBox(self.dino_opts3)
        self.dino_opt_layersim_cbox.setObjectName(u"dino_opt_layersim_cbox")
        self.dino_opt_layersim_cbox.setGeometry(QRect(420, 70, 61, 31))
        self.dinolabel_2 = QLabel(self.dino_opts3)
        self.dinolabel_2.setObjectName(u"dinolabel_2")
        self.dinolabel_2.setGeometry(QRect(240, 70, 51, 31))
        self.box_metric_tabs.addTab(self.dino_tab, "")
        self.texture_tab = QWidget()
        self.texture_tab.setObjectName(u"texture_tab")
        self.texture_opts_1 = QGroupBox(self.texture_tab)
        self.texture_opts_1.setObjectName(u"texture_opts_1")
        self.texture_opts_1.setGeometry(QRect(10, 10, 491, 51))
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
        self.texture_opts_2.setGeometry(QRect(10, 70, 491, 41))
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
        self.box_metric_tabs.addTab(self.texture_tab, "")
        self.emotion_tab = QWidget()
        self.emotion_tab.setObjectName(u"emotion_tab")
        self.label_7 = QLabel(self.emotion_tab)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setGeometry(QRect(10, 50, 111, 31))
        self.label_7.setWordWrap(False)
        self.emotion_opt_cosdist = QRadioButton(self.emotion_tab)
        self.emotion_opt_cosdist.setObjectName(u"emotion_opt_cosdist")
        self.emotion_opt_cosdist.setGeometry(QRect(120, 50, 131, 31))
        self.emotion_opt_eucdist = QRadioButton(self.emotion_tab)
        self.emotion_opt_eucdist.setObjectName(u"emotion_opt_eucdist")
        self.emotion_opt_eucdist.setGeometry(QRect(260, 50, 161, 31))
        self.show_attention = QPushButton(self.emotion_tab)
        self.show_attention.setObjectName(u"show_attention")
        self.show_attention.setGeometry(QRect(10, 10, 111, 24))
        self.emotion_opt_2dsim_3 = QRadioButton(self.emotion_tab)
        self.emotion_opt_2dsim_3.setObjectName(u"emotion_opt_2dsim_3")
        self.emotion_opt_2dsim_3.setGeometry(QRect(290, 140, 200, 31))
        self.emotion_opt_fullsim = QRadioButton(self.emotion_tab)
        self.emotion_opt_fullsim.setObjectName(u"emotion_opt_fullsim")
        self.emotion_opt_fullsim.setGeometry(QRect(10, 140, 251, 31))
        self.box_metric_tabs.addTab(self.emotion_tab, "")
        self.dummy_tab = QWidget()
        self.dummy_tab.setObjectName(u"dummy_tab")
        self.box_metric_tabs.addTab(self.dummy_tab, "")
        self.semantic_tab = QWidget()
        self.semantic_tab.setObjectName(u"semantic_tab")
        self.box_metric_tabs.addTab(self.semantic_tab, "")
        self.box_left_low = QWidget(self.centralwidget)
        self.box_left_low.setObjectName(u"box_left_low")
        self.box_left_low.setGeometry(QRect(10, 460, 491, 261))
        self.label_8 = QLabel(self.box_left_low)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setGeometry(QRect(170, 10, 121, 17))
        self.horizontalLayoutWidget_2 = QWidget(self.box_left_low)
        self.horizontalLayoutWidget_2.setObjectName(u"horizontalLayoutWidget_2")
        self.horizontalLayoutWidget_2.setGeometry(QRect(10, 30, 461, 41))
        self.horizontalLayout = QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.style_label_text_2 = QLabel(self.horizontalLayoutWidget_2)
        self.style_label_text_2.setObjectName(u"style_label_text_2")

        self.horizontalLayout.addWidget(self.style_label_text_2)

        self.label_11 = QLabel(self.horizontalLayoutWidget_2)
        self.label_11.setObjectName(u"label_11")

        self.horizontalLayout.addWidget(self.label_11)

        self.dataset_filtering_from_date = QLineEdit(self.horizontalLayoutWidget_2)
        self.dataset_filtering_from_date.setObjectName(u"dataset_filtering_from_date")

        self.horizontalLayout.addWidget(self.dataset_filtering_from_date)

        self.label_12 = QLabel(self.horizontalLayoutWidget_2)
        self.label_12.setObjectName(u"label_12")

        self.horizontalLayout.addWidget(self.label_12)

        self.dataset_filtering_to_date = QLineEdit(self.horizontalLayoutWidget_2)
        self.dataset_filtering_to_date.setObjectName(u"dataset_filtering_to_date")

        self.horizontalLayout.addWidget(self.dataset_filtering_to_date)

        self.horizontalLayoutWidget_3 = QWidget(self.box_left_low)
        self.horizontalLayoutWidget_3.setObjectName(u"horizontalLayoutWidget_3")
        self.horizontalLayoutWidget_3.setGeometry(QRect(10, 80, 461, 41))
        self.horizontalLayout_2 = QHBoxLayout(self.horizontalLayoutWidget_3)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.style_label_text = QLabel(self.horizontalLayoutWidget_3)
        self.style_label_text.setObjectName(u"style_label_text")

        self.horizontalLayout_2.addWidget(self.style_label_text)

        self.dataset_filtering_nationality_cbox = QComboBox(self.horizontalLayoutWidget_3)
        self.dataset_filtering_nationality_cbox.setObjectName(u"dataset_filtering_nationality_cbox")

        self.horizontalLayout_2.addWidget(self.dataset_filtering_nationality_cbox)

        self.horizontalLayoutWidget_4 = QWidget(self.box_left_low)
        self.horizontalLayoutWidget_4.setObjectName(u"horizontalLayoutWidget_4")
        self.horizontalLayoutWidget_4.setGeometry(QRect(10, 130, 461, 41))
        self.horizontalLayout_4 = QHBoxLayout(self.horizontalLayoutWidget_4)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.style_label_text_4 = QLabel(self.horizontalLayoutWidget_4)
        self.style_label_text_4.setObjectName(u"style_label_text_4")

        self.horizontalLayout_4.addWidget(self.style_label_text_4)

        self.dataset_filtering_media_cbox = QComboBox(self.horizontalLayoutWidget_4)
        self.dataset_filtering_media_cbox.setObjectName(u"dataset_filtering_media_cbox")

        self.horizontalLayout_4.addWidget(self.dataset_filtering_media_cbox)

        self.horizontalLayoutWidget_5 = QWidget(self.box_left_low)
        self.horizontalLayoutWidget_5.setObjectName(u"horizontalLayoutWidget_5")
        self.horizontalLayoutWidget_5.setGeometry(QRect(10, 180, 461, 41))
        self.horizontalLayout_5 = QHBoxLayout(self.horizontalLayoutWidget_5)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.apply_filters = QPushButton(self.horizontalLayoutWidget_5)
        self.apply_filters.setObjectName(u"apply_filters")

        self.horizontalLayout_5.addWidget(self.apply_filters)

        self.label_13 = QLabel(self.horizontalLayoutWidget_5)
        self.label_13.setObjectName(u"label_13")

        self.horizontalLayout_5.addWidget(self.label_13)

        self.filtered_dataset_size = QLabel(self.horizontalLayoutWidget_5)
        self.filtered_dataset_size.setObjectName(u"filtered_dataset_size")

        self.horizontalLayout_5.addWidget(self.filtered_dataset_size)

        self.box_right_low = QWidget(self.centralwidget)
        self.box_right_low.setObjectName(u"box_right_low")
        self.box_right_low.setGeometry(QRect(1030, 460, 491, 261))
        self.combo_texture_slider = QSlider(self.box_right_low)
        self.combo_texture_slider.setObjectName(u"combo_texture_slider")
        self.combo_texture_slider.setGeometry(QRect(390, 70, 91, 20))
        self.combo_texture_slider.setMaximum(100)
        self.combo_texture_slider.setSliderPosition(100)
        self.combo_texture_slider.setOrientation(Qt.Orientation.Horizontal)
        self.combo_dummy_slider = QSlider(self.box_right_low)
        self.combo_dummy_slider.setObjectName(u"combo_dummy_slider")
        self.combo_dummy_slider.setGeometry(QRect(390, 90, 91, 20))
        self.combo_dummy_slider.setMaximum(100)
        self.combo_dummy_slider.setOrientation(Qt.Orientation.Horizontal)
        self.combo_dino_slider = QSlider(self.box_right_low)
        self.combo_dino_slider.setObjectName(u"combo_dino_slider")
        self.combo_dino_slider.setGeometry(QRect(390, 50, 91, 20))
        self.combo_dino_slider.setMaximum(100)
        self.combo_dino_slider.setOrientation(Qt.Orientation.Horizontal)
        self.label_2 = QLabel(self.box_right_low)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(350, 20, 111, 21))
        self.label_3 = QLabel(self.box_right_low)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(320, 50, 67, 17))
        self.label_4 = QLabel(self.box_right_low)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setGeometry(QRect(320, 70, 67, 17))
        self.label_5 = QLabel(self.box_right_low)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setGeometry(QRect(320, 90, 67, 17))
        self.recalc_similarity = QPushButton(self.box_right_low)
        self.recalc_similarity.setObjectName(u"recalc_similarity")
        self.recalc_similarity.setGeometry(QRect(320, 170, 161, 25))
        self.label_9 = QLabel(self.box_right_low)
        self.label_9.setObjectName(u"label_9")
        self.label_9.setGeometry(QRect(320, 110, 67, 17))
        self.combo_semantic_slider = QSlider(self.box_right_low)
        self.combo_semantic_slider.setObjectName(u"combo_semantic_slider")
        self.combo_semantic_slider.setGeometry(QRect(390, 110, 91, 20))
        self.combo_semantic_slider.setMaximum(100)
        self.combo_semantic_slider.setOrientation(Qt.Orientation.Horizontal)
        self.label_10 = QLabel(self.box_right_low)
        self.label_10.setObjectName(u"label_10")
        self.label_10.setGeometry(QRect(320, 130, 67, 17))
        self.combo_emotion_slider = QSlider(self.box_right_low)
        self.combo_emotion_slider.setObjectName(u"combo_emotion_slider")
        self.combo_emotion_slider.setGeometry(QRect(390, 130, 91, 20))
        self.combo_emotion_slider.setMaximum(100)
        self.combo_emotion_slider.setOrientation(Qt.Orientation.Horizontal)
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
        self.n1.setGeometry(QRect(1030, 360, 161, 91))
        self.n2 = QLabel(self.centralwidget)
        self.n2.setObjectName(u"n2")
        self.n2.setGeometry(QRect(1190, 360, 171, 91))
        self.n3 = QLabel(self.centralwidget)
        self.n3.setObjectName(u"n3")
        self.n3.setGeometry(QRect(1360, 360, 161, 91))
        self.pushButton = QPushButton(self.centralwidget)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setGeometry(QRect(1030, 330, 491, 25))
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        self.box_metric_tabs.setCurrentIndex(3)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Deep Similarity Analyzer", None))
#if QT_CONFIG(tooltip)
        self.b_upload.setToolTip(QCoreApplication.translate("MainWindow", u"Upload a new image to analyse", None))
#endif // QT_CONFIG(tooltip)
        self.b_upload.setText(QCoreApplication.translate("MainWindow", u"Upload Image", None))
        self.dino_opts1.setTitle("")
        self.dinolabl2.setText(QCoreApplication.translate("MainWindow", u"head:", None))
        self.dinolabel.setText(QCoreApplication.translate("MainWindow", u"layer:", None))
        self.dino_opt_showcamap.setText(QCoreApplication.translate("MainWindow", u"show crossattention map", None))
        self.dino_opts2.setTitle("")
        self.dino_opt_eucdist.setText(QCoreApplication.translate("MainWindow", u"euclidian distance", None))
        self.dino_opt_cosdist.setText(QCoreApplication.translate("MainWindow", u"cosine distance", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"distance metric:", None))
        self.dino_opts3.setTitle("")
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
        self.box_metric_tabs.setTabText(self.box_metric_tabs.indexOf(self.texture_tab), QCoreApplication.translate("MainWindow", u"Texture", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"distance metric:", None))
        self.emotion_opt_cosdist.setText(QCoreApplication.translate("MainWindow", u"cosine distance", None))
        self.emotion_opt_eucdist.setText(QCoreApplication.translate("MainWindow", u"euclidian distance", None))
        self.show_attention.setText(QCoreApplication.translate("MainWindow", u"Show attention", None))
        self.emotion_opt_2dsim_3.setText(QCoreApplication.translate("MainWindow", u"2d reprojection similarity", None))
        self.emotion_opt_fullsim.setText(QCoreApplication.translate("MainWindow", u"full feature vector similarity", None))
        self.box_metric_tabs.setTabText(self.box_metric_tabs.indexOf(self.emotion_tab), QCoreApplication.translate("MainWindow", u"Emotion", None))
        self.box_metric_tabs.setTabText(self.box_metric_tabs.indexOf(self.dummy_tab), QCoreApplication.translate("MainWindow", u"Dummy", None))
        self.box_metric_tabs.setTabText(self.box_metric_tabs.indexOf(self.semantic_tab), QCoreApplication.translate("MainWindow", u"Semantic", None))
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"Dataset Filtering", None))
        self.style_label_text_2.setText(QCoreApplication.translate("MainWindow", u"Period", None))
        self.label_11.setText(QCoreApplication.translate("MainWindow", u"from", None))
        self.dataset_filtering_from_date.setText(QCoreApplication.translate("MainWindow", u"0", None))
        self.label_12.setText(QCoreApplication.translate("MainWindow", u"To", None))
        self.dataset_filtering_to_date.setText(QCoreApplication.translate("MainWindow", u"2024", None))
        self.style_label_text.setText(QCoreApplication.translate("MainWindow", u"Artist Nationality", None))
        self.style_label_text_4.setText(QCoreApplication.translate("MainWindow", u"Painting Medium", None))
        self.apply_filters.setText(QCoreApplication.translate("MainWindow", u"Apply Filters", None))
        self.label_13.setText(QCoreApplication.translate("MainWindow", u"Current dataset size:", None))
        self.filtered_dataset_size.setText(QCoreApplication.translate("MainWindow", u"INT", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"metric weights", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"dino", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"texture", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"dummy", None))
        self.recalc_similarity.setText(QCoreApplication.translate("MainWindow", u"recalculate similarity", None))
        self.label_9.setText(QCoreApplication.translate("MainWindow", u"semantic", None))
        self.label_10.setText(QCoreApplication.translate("MainWindow", u"emotion", None))
        self.t_artist.setText(QCoreApplication.translate("MainWindow", u"Artist: ", None))
        self.t_date.setText(QCoreApplication.translate("MainWindow", u"Date:", None))
        self.t_style.setText(QCoreApplication.translate("MainWindow", u"Style:", None))
        self.t_tags.setText(QCoreApplication.translate("MainWindow", u"Tags:", None))
        self.box_left_img.setText("")
        self.box_right_img.setText("")
        self.b_recalculate.setText(QCoreApplication.translate("MainWindow", u"metric specific", None))
        self.b_combined.setText(QCoreApplication.translate("MainWindow", u"combined", None))
        self.b_metric.setText(QCoreApplication.translate("MainWindow", u"recalculate projection", None))
        self.r_image_points.setText(QCoreApplication.translate("MainWindow", u"points as images", None))
        self.n1.setText("")
        self.n2.setText("")
        self.n3.setText("")
#if QT_CONFIG(tooltip)
        self.pushButton.setToolTip(QCoreApplication.translate("MainWindow", u"Send the above image to the left for deeper analysis", None))
#endif // QT_CONFIG(tooltip)
        self.pushButton.setText(QCoreApplication.translate("MainWindow", u"inspect", None))
    # retranslateUi

