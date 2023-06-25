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
    QGroupBox, QHBoxLayout, QLabel, QMainWindow,
    QPlainTextEdit, QPushButton, QRadioButton, QSizePolicy,
    QSlider, QTabWidget, QVBoxLayout, QWidget)

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
        self.b_upload.setGeometry(QRect(10, 330, 451, 25))
        sizePolicy = QSizePolicy()
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.b_upload.sizePolicy().hasHeightForWidth())
        self.b_upload.setSizePolicy(sizePolicy)
        self.box_left_low = QWidget(self.centralwidget)
        self.box_left_low.setObjectName(u"box_left_low")
        self.box_left_low.setGeometry(QRect(10, 460, 461, 301))
        self.verticalLayoutWidget = QWidget(self.box_left_low)
        self.verticalLayoutWidget.setObjectName(u"verticalLayoutWidget")
        self.verticalLayoutWidget.setGeometry(QRect(10, 10, 441, 141))
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

        self.b_submit = QPushButton(self.verticalLayoutWidget)
        self.b_submit.setObjectName(u"b_submit")

        self.Clip_input_layout.addWidget(self.b_submit)

        self.l_timeline = QLabel(self.box_left_low)
        self.l_timeline.setObjectName(u"l_timeline")
        self.l_timeline.setGeometry(QRect(10, 170, 439, 21))
        self.box_timeline = QWidget(self.box_left_low)
        self.box_timeline.setObjectName(u"box_timeline")
        self.box_timeline.setGeometry(QRect(10, 200, 439, 81))
        self.box_timeline.setAutoFillBackground(True)
        self.box_right_low = QWidget(self.centralwidget)
        self.box_right_low.setObjectName(u"box_right_low")
        self.box_right_low.setEnabled(True)
        self.box_right_low.setGeometry(QRect(1060, 420, 471, 341))
        self.formLayoutWidget = QWidget(self.box_right_low)
        self.formLayoutWidget.setObjectName(u"formLayoutWidget")
        self.formLayoutWidget.setGeometry(QRect(10, 0, 171, 139))
        self.formLayout = QFormLayout(self.formLayoutWidget)
        self.formLayout.setObjectName(u"formLayout")
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.title = QLabel(self.formLayoutWidget)
        self.title.setObjectName(u"title")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.formLayout.setWidget(0, QFormLayout.ItemRole.SpanningRole, self.title)

        self.label_dino = QLabel(self.formLayoutWidget)
        self.label_dino.setObjectName(u"label_dino")

        self.formLayout.setWidget(1, QFormLayout.ItemRole.LabelRole, self.label_dino)

        self.combo_dino_slider = QSlider(self.formLayoutWidget)
        self.combo_dino_slider.setObjectName(u"combo_dino_slider")
        self.combo_dino_slider.setMaximum(100)
        self.combo_dino_slider.setOrientation(Qt.Orientation.Horizontal)

        self.formLayout.setWidget(1, QFormLayout.ItemRole.FieldRole, self.combo_dino_slider)

        self.label_texture = QLabel(self.formLayoutWidget)
        self.label_texture.setObjectName(u"label_texture")

        self.formLayout.setWidget(2, QFormLayout.ItemRole.LabelRole, self.label_texture)

        self.combo_texture_slider = QSlider(self.formLayoutWidget)
        self.combo_texture_slider.setObjectName(u"combo_texture_slider")
        self.combo_texture_slider.setMaximum(100)
        self.combo_texture_slider.setSliderPosition(100)
        self.combo_texture_slider.setOrientation(Qt.Orientation.Horizontal)

        self.formLayout.setWidget(2, QFormLayout.ItemRole.FieldRole, self.combo_texture_slider)

        self.label_dummy = QLabel(self.formLayoutWidget)
        self.label_dummy.setObjectName(u"label_dummy")

        self.formLayout.setWidget(3, QFormLayout.ItemRole.LabelRole, self.label_dummy)

        self.combo_dummy_slider = QSlider(self.formLayoutWidget)
        self.combo_dummy_slider.setObjectName(u"combo_dummy_slider")
        self.combo_dummy_slider.setMaximum(100)
        self.combo_dummy_slider.setOrientation(Qt.Orientation.Horizontal)

        self.formLayout.setWidget(3, QFormLayout.ItemRole.FieldRole, self.combo_dummy_slider)

        self.recalc_similarity = QPushButton(self.formLayoutWidget)
        self.recalc_similarity.setObjectName(u"recalc_similarity")

        self.formLayout.setWidget(4, QFormLayout.ItemRole.SpanningRole, self.recalc_similarity)

        self.statistics_tab = QTabWidget(self.box_right_low)
        self.statistics_tab.setObjectName(u"statistics_tab")
        self.statistics_tab.setGeometry(QRect(10, 150, 451, 181))
        self.selection_stats = QWidget()
        self.selection_stats.setObjectName(u"selection_stats")
        self.verticalLayoutWidget_3 = QWidget(self.selection_stats)
        self.verticalLayoutWidget_3.setObjectName(u"verticalLayoutWidget_3")
        self.verticalLayoutWidget_3.setGeometry(QRect(-1, 9, 451, 141))
        self.selection_stats_layout = QVBoxLayout(self.verticalLayoutWidget_3)
        self.selection_stats_layout.setObjectName(u"selection_stats_layout")
        self.selection_stats_layout.setContentsMargins(0, 0, 0, 0)
        self.statistics_tab.addTab(self.selection_stats, "")
        self.other_stats = QWidget()
        self.other_stats.setObjectName(u"other_stats")
        self.verticalLayoutWidget_4 = QWidget(self.other_stats)
        self.verticalLayoutWidget_4.setObjectName(u"verticalLayoutWidget_4")
        self.verticalLayoutWidget_4.setGeometry(QRect(-1, 9, 451, 141))
        self.other_stats_layout = QVBoxLayout(self.verticalLayoutWidget_4)
        self.other_stats_layout.setObjectName(u"other_stats_layout")
        self.other_stats_layout.setContentsMargins(0, 0, 0, 0)
        self.statistics_tab.addTab(self.other_stats, "")
        self.box_recom_img_stats = QGroupBox(self.box_right_low)
        self.box_recom_img_stats.setObjectName(u"box_recom_img_stats")
        self.box_recom_img_stats.setGeometry(QRect(200, 0, 261, 141))
        self.verticalLayoutWidget_2 = QWidget(self.box_recom_img_stats)
        self.verticalLayoutWidget_2.setObjectName(u"verticalLayoutWidget_2")
        self.verticalLayoutWidget_2.setGeometry(QRect(-1, 9, 261, 131))
        self.verticalLayout = QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.radioButton = QRadioButton(self.verticalLayoutWidget_2)
        self.radioButton.setObjectName(u"radioButton")

        self.verticalLayout.addWidget(self.radioButton)

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
        self.horizontalLayoutWidget.setGeometry(QRect(480, 420, 571, 31))
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
        self.scatterplot_frame.setGeometry(QRect(480, 10, 571, 401))
        self.box_metric_tabs = QTabWidget(self.centralwidget)
        self.box_metric_tabs.setObjectName(u"box_metric_tabs")
        self.box_metric_tabs.setGeometry(QRect(490, 460, 551, 301))
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
        self.groupBox = QGroupBox(self.centralwidget)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setGeometry(QRect(1060, 10, 471, 401))
        self.box_right_img = QLabel(self.groupBox)
        self.box_right_img.setObjectName(u"box_right_img")
        self.box_right_img.setGeometry(QRect(10, 20, 451, 261))
        self.box_right_img.setAutoFillBackground(True)
        self.box_right_img.setLineWidth(1)
        self.horizontalLayoutWidget_2 = QWidget(self.groupBox)
        self.horizontalLayoutWidget_2.setObjectName(u"horizontalLayoutWidget_2")
        self.horizontalLayoutWidget_2.setGeometry(QRect(9, 290, 451, 111))
        self.preview_layout = QHBoxLayout(self.horizontalLayoutWidget_2)
        self.preview_layout.setObjectName(u"preview_layout")
        self.preview_layout.setContentsMargins(0, 0, 0, 0)
        self.n3 = QLabel(self.horizontalLayoutWidget_2)
        self.n3.setObjectName(u"n3")
        self.n3.setAutoFillBackground(True)

        self.preview_layout.addWidget(self.n3)

        self.n2 = QLabel(self.horizontalLayoutWidget_2)
        self.n2.setObjectName(u"n2")
        self.n2.setAutoFillBackground(True)

        self.preview_layout.addWidget(self.n2)

        self.n1 = QLabel(self.horizontalLayoutWidget_2)
        self.n1.setObjectName(u"n1")
        self.n1.setAutoFillBackground(True)

        self.preview_layout.addWidget(self.n1)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        self.statistics_tab.setCurrentIndex(0)
        self.box_metric_tabs.setCurrentIndex(2)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Deep Similarity Analyzer", None))
#if QT_CONFIG(tooltip)
        self.b_upload.setToolTip(QCoreApplication.translate("MainWindow", u"Upload a new image to analyse", None))
#endif // QT_CONFIG(tooltip)
        self.b_upload.setText(QCoreApplication.translate("MainWindow", u"Upload Image", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"Describe similarity aspect to a Clip model", None))
        self.tb_clip_input.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Can you give me an image with a similar ...", None))
        self.b_submit.setText(QCoreApplication.translate("MainWindow", u"Submit Similarity Description", None))
        self.l_timeline.setText(QCoreApplication.translate("MainWindow", u"Timeline for above image", None))
        self.title.setText(QCoreApplication.translate("MainWindow", u"Metric weights", None))
        self.label_dino.setText(QCoreApplication.translate("MainWindow", u"Dino", None))
        self.label_texture.setText(QCoreApplication.translate("MainWindow", u"Texture", None))
        self.label_dummy.setText(QCoreApplication.translate("MainWindow", u"Dummy", None))
        self.recalc_similarity.setText(QCoreApplication.translate("MainWindow", u"Recalculate similarity", None))
        self.statistics_tab.setTabText(self.statistics_tab.indexOf(self.selection_stats), QCoreApplication.translate("MainWindow", u"Selection statistics", None))
        self.statistics_tab.setTabText(self.statistics_tab.indexOf(self.other_stats), QCoreApplication.translate("MainWindow", u"Other statistics", None))
        self.box_recom_img_stats.setTitle(QCoreApplication.translate("MainWindow", u"Similarity aspects stats", None))
        self.radioButton.setText(QCoreApplication.translate("MainWindow", u"temp_button", None))
        self.t_artist.setText(QCoreApplication.translate("MainWindow", u"Artist: ", None))
        self.t_date.setText(QCoreApplication.translate("MainWindow", u"Date:", None))
        self.t_style.setText(QCoreApplication.translate("MainWindow", u"Style:", None))
        self.t_tags.setText(QCoreApplication.translate("MainWindow", u"Tags:", None))
        self.box_left_img.setText("")
        self.b_recalculate.setText(QCoreApplication.translate("MainWindow", u"Metric specific", None))
        self.b_combined.setText(QCoreApplication.translate("MainWindow", u"Weighted projection", None))
        self.b_metric.setText(QCoreApplication.translate("MainWindow", u"Recalculate projection", None))
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
        self.box_metric_tabs.setTabText(self.box_metric_tabs.indexOf(self.texture_tab), QCoreApplication.translate("MainWindow", u"Texture", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"distance metric:", None))
        self.emotion_opt_cosdist.setText(QCoreApplication.translate("MainWindow", u"cosine distance", None))
        self.emotion_opt_eucdist.setText(QCoreApplication.translate("MainWindow", u"euclidian distance", None))
        self.show_attention.setText(QCoreApplication.translate("MainWindow", u"Show attention", None))
        self.emotion_opt_2dsim_3.setText(QCoreApplication.translate("MainWindow", u"2d reprojection similarity", None))
        self.emotion_opt_fullsim.setText(QCoreApplication.translate("MainWindow", u"full feature vector similarity", None))
        self.box_metric_tabs.setTabText(self.box_metric_tabs.indexOf(self.emotion_tab), QCoreApplication.translate("MainWindow", u"Emotion", None))
        self.box_metric_tabs.setTabText(self.box_metric_tabs.indexOf(self.dummy_tab), QCoreApplication.translate("MainWindow", u"Dummy", None))
        self.groupBox.setTitle(QCoreApplication.translate("MainWindow", u"Similar Images", None))
        self.box_right_img.setText("")
        self.n3.setText("")
        self.n2.setText("")
        self.n1.setText("")
    # retranslateUi

