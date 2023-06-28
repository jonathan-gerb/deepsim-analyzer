# This Python file uses the following encoding: utf-8
import configparser
import os
import sys
from copy import deepcopy
# normal imports
from pathlib import Path

import cv2
import clip
import numpy as np
import pyqtgraph as pg
import torch
from PIL import Image
import configparser
from scipy import spatial
from sklearn.preprocessing import minmax_scale
from sklearn.metrics.pairwise import cosine_distances
import os
from tqdm import tqdm
import cv2
from tensorflow import keras
import matplotlib.cm as cm
import joblib
from umap import UMAP

# qt imports
from PyQt6 import QtWidgets
from PyQt6.QtCore import QCoreApplication, QDate, QEvent, QRect, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QMouseEvent, QPainter, QPixmap, QTransform
from PyQt6.QtWidgets import (QApplication, QFileDialog, QGraphicsScene,
                             QGraphicsView, QMainWindow, QSizePolicy,
                             QTabWidget, QVBoxLayout,QLabel)
from tqdm import tqdm

# deepsim analyzer package
import deepsim_analyzer as da

# custom widgets
from .custom_widgets import ScatterplotWidget, TimelineView, TimelineWindow,BarChart
# Important:
# You need to run the following command to generate the ui_form.py file
#     pyside6-uic form.ui -o ui_form.py, or
#     pyside2-uic form.ui -o ui_form.py
# additionally run python fix_ui_script.py, which replaces the not working ui stuff.
from .ui_form import Ui_MainWindow


class MainWindow(QMainWindow):

    def __init__(self, key_dict, datafile_path, image_directory, features_to_use):
        super().__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.b_upload.clicked.connect(self.upload_image_left)

        print("setting up configs")

        # load the config file 'config.ini'
        self.config = configparser.ConfigParser()
        basepath = Path(__file__).parent
        config_filepath = str(basepath / "config.ini")
        self.config.read(config_filepath)
        self.datafile_path = datafile_path

        # set image max allocation pretty high
        os.environ['QT_IMAGEIO_MAXALLOC'] = "512"

        # save passed arguments as attributes
        self.image_key_dict = key_dict
        self.image_directory = image_directory
        self.image_keys = [key for key in key_dict.keys()] # hashes
        self.image_indices = list(range(len(self.image_keys)))
        self.non_filtered_indices = self.image_indices
        self.key_to_idx = {key:i for i, key in enumerate(self.image_keys)} # hashes
        self.image_paths = [str(Path(image_directory) / self.image_key_dict[key]) for key in self.image_keys]

        # load umap reducers
        self.available_features = features_to_use

        self.reducer_basepath = Path(__file__).parents[2] / "data" / "processed" / "reducers"
        reducer_paths = [path for path in self.reducer_basepath.glob("*.umap")]
        self.reducers = {}
        for r_path in reducer_paths:
            reducer_name = r_path.name.split(".")[0]
            self.reducers[reducer_name] = joblib.load(str(r_path))
        self.any_reprojection_applied = False

        # metric option defaults
        self.dino_distance_measure = "euclidian"
        self.texture_distance_measure = "euclidian"
        self.emotion_distance_measure = "euclidian"
        self.semantic_distance_measure = "cosine"
        self.emotion_opt_sim_vector_type = "full"
        self.dino_opt_sim_vector_type = "full"
        self.clip_opt_sim_vector_type = "full"
        self.texture_opt_sim_vector_type = "full"
        self.semantic_opt_sim_vector_type = "full"

        # set color for main ui
        self.set_color_element(self.ui.centralwidget, [71, 71, 71])
        # set color for lower text boxes
        self.set_color_element(self.ui.box_left_low, [143, 143, 143])
        self.set_color_element(self.ui.box_right_low, [143, 143, 143])
        self.set_color_element(self.ui.box_metainfo, [143, 143, 143])
        
        # set tab ui color
        self.set_color_element(self.ui.box_metric_tabs, [143, 143, 143])
        # self.set_color_element(self.ui.dummy_tab, [143, 143, 143])
        self.set_color_element(self.ui.dino_tab, [143, 143, 143])
        self.set_color_element(self.ui.texture_tab, [143, 143, 143])
        self.set_color_element(self.ui.emotion_tab, [143, 143, 143])
        self.set_color_element(self.ui.semantic_tab, [143, 143, 143])
        self.set_color_element(self.ui.clip_tab, [143, 143, 143])
        self.set_color_element(self.ui.combined_tab, [143, 143, 143])

        # set color for stats 
        self.set_color_element(self.ui.style_stats, [143, 143, 143])
        self.set_color_element(self.ui.date_stats, [143, 143, 143])
        self.set_color_element(self.ui.nationality_stats, [143, 143, 143])


        # ================ SETUP DATA ================
        print("setting up internal data")
        self.metadata = {}

        # read metadata for all image keys that were passed
        print("reading metadata")
        self.metadata = da.read_metadata_batch(datafile_path, self.image_keys)

        # sets up the data_dict
        self.load_metric_data()

        # ================ SETUP LEFT COLUMN ================
        print("-------setting up left column of dashboard")
        # ---------------- STARTING IMG ----------------
        # load an initial first image to display
        default_image_key = list(key_dict.keys())[0]
        default_image_path = key_dict[default_image_key]
        default_image_path_absolute = str(Path(image_directory) / default_image_path)
        
        self.left_img_key = default_image_key
        self.left_img_filename = default_image_path_absolute

        print('default_image_path',default_image_path)
        # load in timeline
        self.timeline= TimelineWindow(default_image_path)
        self.ui.box_timeline_layout.addWidget(self.timeline)
        self.no_timeline_label = QLabel('No data for timeline of new uploaded image')
        self.ui.box_timeline_layout.addWidget(self.no_timeline_label)

        # display the base image
        self.display_photo_left(default_image_path_absolute)
        # data for left img feature, can come  from dataset or be calculated on the fly
        self.update_leftimg_data(self.left_img_key)
        # add additional data in box_left_low

        self.setup_filters()
        self.ui.apply_filters.pressed.connect(self.apply_filters)
        self.ui.reset_dataset_filters.pressed.connect(self.reset_data_filters)
        self.ui.reload_everything.pressed.connect(self.reload_data_dict)

        # ================ SETUP MIDDLE COLUMN ================
        
        print("------setting up scatterplot")
        self.ui.box_metric_tabs.setCurrentIndex(4)
        # setup scatterplot
        # TODO: setup feature projection plot with combined and individual plots!
        self.ui.box_metric_tabs.currentChanged.connect(self.setup_scatterplot)
        # And setup up once to initialize
        self.setup_scatterplot()
    
        # functionality to recalculate projections
        self.ui.combined_projection_btn.pressed.connect(self.calc_combined_projection)

        self.ui.subset_projection_btn.pressed.connect(self.recalc_projection)

        print("------setting up middle metric options")
        # ----------------SETUP TEXTURE OPTIONS----------------
        print("setting up texture options")
        # options for what distance measure to use.
        self.ui.texture_opt_eucdist.toggle()
        self.ui.texture_opt_cosdist.toggled.connect(self.texture_opt_dist_cos)
        self.ui.texture_opt_eucdist.toggled.connect(self.texture_opt_dist_euc)

        self.ui.texture_opt_fullsim.toggle()
        self.ui.texture_opt_2dsim.toggled.connect(self.texture_opt_simtype)
        self.ui.texture_opt_fullsim.toggled.connect(self.texture_opt_simtype)

        # add options for head similarity to comboboxes
        for i in range(256):
            self.ui.texture_opt_filtervis.addItem(f"3a {i+1}")

        for i in range(480):
            self.ui.texture_opt_filtervis.addItem(f"3b {i+1}")

        self.ui.texture_opt_filtervis.currentIndexChanged.connect(self.texture_show_fm)
        self.ui.texture_opt_show_fm.toggled.connect(self.texture_show_fm)

        # ----------------SETUP CLIP OPTIONS----------------
        print("setting up clip options")
        self.device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        print("loading clip model")
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        self.clip_model = self.clip_model.to(self.device)
        self.ui.clip_radio_imgsim.toggle()

        self.ui.clip_opt_fullsim.toggle()
        self.ui.clip_opt_2dsim.toggled.connect(self.clip_opt_simtype)
        self.ui.clip_opt_fullsim.toggled.connect(self.clip_opt_simtype)

        # ----------------SETUP EMOTION OPTIONS----------------
        print("setting up emotion options")
        # options for what distance measure to use.
        self.ui.emotion_opt_eucdist.toggle()
        self.ui.emotion_opt_cosdist.toggled.connect(self.emotion_opt_dist_cos)
        self.ui.emotion_opt_eucdist.toggled.connect(self.emotion_opt_dist_euc)
        
        # options for calculating similarity based on what vector
        self.ui.emotion_opt_fullsim.toggle()
        self.ui.emotion_opt_2dsim.toggled.connect(self.emotion_opt_simtype)
        self.ui.emotion_opt_fullsim.toggled.connect(self.emotion_opt_simtype)

        self.ui.emotion_opts_showfm.toggled.connect(self.emotion_show_fm)
        # ----------------SETUP SEMANTIC OPTIONS----------------
        self.ui.semantic_opt_eucdist.toggle()
        self.ui.semantic_opt_cosdist.toggled.connect(self.semantic_opt_dist_cos)
        self.ui.semantic_opt_eucdist.toggled.connect(self.semantic_opt_dist_euc)
        
        # options for calculating similarity based on what vector
        self.ui.semantic_opt_fullsim.toggle()
        self.ui.semantic_opt_2dsim.toggled.connect(self.semantic_opt_simtype)
        self.ui.semantic_opt_fullsim.toggled.connect(self.semantic_opt_simtype)

        # ----------------SETUP DINO OPTIONS----------------
        print("setting up dino options")
        # options for what distance measure to use.
        self.ui.dino_opt_eucdist.toggle()
        self.ui.dino_opt_cosdist.toggled.connect(self.dino_opt_dist_cos)
        self.ui.dino_opt_eucdist.toggled.connect(self.dino_opt_dist_euc)
        
        # options for calculating similarity based on what vector
        self.ui.dino_opt_fullsim.toggle()
        self.ui.dino_opt_2dsim.toggled.connect(self.dino_opt_simtype)
        self.ui.dino_opt_fullsim.toggled.connect(self.dino_opt_simtype)
        self.ui.dino_opt_headsim.toggled.connect(self.dino_opt_simtype)

        # add options for head similarity to comboboxes
        for i in range(12):
            self.ui.dino_opt_headsim_cbox.addItem(f"{i+1}")
            self.ui.dino_opt_layersim_cbox.addItem(f"{i+1}")
            self.ui.dino_opt_headvis_cbox.addItem(f"{i+1}")
            self.ui.dino_opt_layervis_cbox.addItem(f"{i+1}")


        self.ui.dino_opt_headvis_cbox.currentIndexChanged.connect(self.dino_show_camap)
        self.ui.dino_opt_layervis_cbox.currentIndexChanged.connect(self.dino_show_camap)

        # dropdown options for dino head-specific similarity
        self.ui.dino_opt_headsim_cbox.editTextChanged.connect(self.dino_opt_simtype)
        self.ui.dino_opt_layersim_cbox.editTextChanged.connect(self.dino_opt_simtype)

        # option for showing crossattention map
        self.ui.dino_opt_showcamap.toggled.connect(self.dino_show_camap)

        # ================ SETUP COMBINED PROJECTION ================
        self.calc_combined_projection()

        # ================ SETUP RIGHT COLUMN ================
        print("------setting up right column, calculating nearest neighbours")
        self.recalc_similarity()
        
        print("recalculating")
        self.ui.recalc_similarity.pressed.connect(self.recalc_similarity)
        print("dashboard setup complete!")

        print('--------setting up barplots')
        self.bp = BarChart(self)
        self.bp2 = BarChart(self)
        self.bp3 = BarChart(self)
        self.scatterplot.get_Selected_stats.connect(self.get_selected_points_stats)
        self.scatterplot.get_Selected_stats.emit(0) # once for initialization, after in scatterplot.get_selection

        img_stats_container = self.ui.style_stats_layout
        img_stats_container.layout().addWidget(self.bp)

        img_stats_container2 = self.ui.date_stats_layout
        img_stats_container2.layout().addWidget(self.bp2)
        
        img_stats_container3 = self.ui.nationality_stats_layout
        img_stats_container3.layout().addWidget(self.bp3)

        # Set the size policy for the bar plot widget
        size_policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.bp.setSizePolicy(size_policy)
        self.bp2.setSizePolicy(size_policy)
        self.bp3.setSizePolicy(size_policy)

        self.ui.statistics_tabs.currentChanged.connect(self.show_animation_on_tab_switch)
        self.bar_plots = [self.bp, self.bp2, self.bp3]


    def load_metric_data(self):
        # init datadict
        self.data_dict = {}

        # key data for first key to get shapes
        test_dict_key = self.get_features_from_dataset(self.image_keys[0])

        for feature_name in self.available_features:
            # get test feature to mock shape
            test_feature = test_dict_key[feature_name]['full']
            self.data_dict[feature_name] = {}
            self.data_dict[feature_name]["projection"] = np.zeros((len(self.image_keys), 2))
            self.data_dict[feature_name]["full"] = np.zeros((len(self.image_keys), test_feature.shape[0]))

        for i, key in tqdm(enumerate(self.image_keys), desc="loading in feature data", total=len(self.image_keys)):
            feature_dict_key = self.get_features_from_dataset(key)
            for feature_name, value in feature_dict_key.items():
                self.data_dict[feature_name]["projection"][i] = value['projection']
                self.data_dict[feature_name]["full"][i] = value['full']


    def recalc_projection(self):
        self.any_reprojection_applied = True
        current_metric_type = self.ui.box_metric_tabs.tabText(self.ui.box_metric_tabs.currentIndex()).lower()
        print(f"reprojecting data for: {current_metric_type}")

        if len(self.scatterplot.selected_indices) > 0:
            indices_to_keep_scatterplot = [idx for idx in self.scatterplot.selected_indices if idx in self.non_filtered_indices]
            print('selected points:')
            print(indices_to_keep_scatterplot, len(indices_to_keep_scatterplot))
            print("indices to keep: ")
            print(self.non_filtered_indices, len(self.non_filtered_indices))
            indices_to_project = indices_to_keep_scatterplot
        else:
            indices_to_project = self.non_filtered_indices

        data_to_reproject = self.data_dict[current_metric_type]['full'][indices_to_project]

        # only use the selected indices to calculate the umap, after that use it to project all the data
        # (even if it's not actually visible) so that we don't mix projected and reprojected data.
        print(f"projecting data with shape: {data_to_reproject.shape}")
        reducer = UMAP(n_neighbors=6, n_components=2, metric='cosine').fit(data_to_reproject)
        print(f"overwriting reprojected data for metric: {current_metric_type}")
        reprojected_data = reducer.transform(self.data_dict[current_metric_type]['full'])
        print(reprojected_data.shape)
        self.data_dict[current_metric_type]['projection'] = reprojected_data

        # now recalculate projection etc
        self.recalc_similarity()
        self.setup_scatterplot(clear_selection=False)

    
    def calc_combined_projection(self):
        print("calculating combined projection")
        self.data_dict["combined"] = {}

        # first concatenate all the vectors into one big mega vector
        if "full" not in self.data_dict["combined"]:
            self.data_dict["combined"]["full"] = np.concatenate([self.data_dict[feature_name]['full'] for feature_name in self.available_features], axis=1)

        # construct reweighting matrix
        feature_weight_dict = self.get_metric_combo_weights()
        combined_feature_weight = []
        for feature_name in self.available_features:
            feature_size = self.data_dict[feature_name]['full'].shape[1]
            weights = np.zeros(feature_size) + feature_weight_dict[feature_name]
            combined_feature_weight.append(weights)

        combined_feature_weight = np.concatenate(combined_feature_weight, axis=0)
        
        # normalize the weight matrix
        combined_feature_weight = combined_feature_weight / np.linalg.norm(combined_feature_weight)
        
        # make it a diagonal matrix
        combined_feature_weight = np.diag(combined_feature_weight)

        # apply the weighting
        all_features_combined = self.data_dict["combined"]["full"] @ combined_feature_weight

        # then calculate the umap
        print("calculating Umap projection model")
        self.combined_reducer = UMAP(n_neighbors=6, n_components=2, metric='cosine', verbose=True).fit(all_features_combined)
        self.data_dict["combined"]["projection"] = self.combined_reducer.transform(all_features_combined)
        # then plot that combined umap
        self.recalc_similarity()
        self.setup_scatterplot()

    def show_animation_on_tab_switch(self,index):
        if 0 <= index < len(self.bar_plots):
            print('show_animation_on_tab_switch')
            # self.bar_plots[index].repaint()
            self.bar_plots[index].chartView.update()


    def setup_filters(self):
        nationalities = []
        artist_names = []
        media = []
        dates = []
        for value in self.metadata.values():
            ns = value['artist_nationality'].split(",")
            for n in ns:
                nationalities.append(n.strip())

            ms = value['media'].split(",")
            for m in ms:
                media.append(m.strip())

            dates.append(int(value['date']))
            artist_names.append(str(value['artist_name']))

        nationalities = list(set(nationalities))
        media = list(set(media))

        self.ui.dataset_filtering_nationality_cbox.clear()
        self.ui.dataset_filtering_media_cbox.clear()

        self.ui.dataset_filtering_nationality_cbox.addItems(nationalities)
        self.ui.dataset_filtering_media_cbox.addItems(media)

        self.ui.dataset_filtering_nationality_cbox.addItem("all")
        self.ui.dataset_filtering_media_cbox.addItem("all")

        self.ui.dataset_filtering_nationality_cbox.setCurrentText("all")
        self.ui.dataset_filtering_media_cbox.setCurrentText("all")

        self.ui.dataset_filtering_from_date.setText(str(min(dates)))
        self.ui.dataset_filtering_to_date.setText(str(max(dates)+1))

        self.ui.filtered_dataset_size.setText(f"{len(self.non_filtered_indices)}/{len(self.image_indices)}")
        # self.ui.filtered_dataset_size.setText(f"{len(self.indices_to_keep)}/{len(self.image_indices)}")
        

    def apply_filters(self):
        filter_media = self.ui.dataset_filtering_media_cbox.currentText()
        filter_nationality = self.ui.dataset_filtering_nationality_cbox.currentText()
        try:
            filter_date_from = int(self.ui.dataset_filtering_from_date.text())
        except:
            print(f"could not convert date_from to int: {self.ui.dataset_filtering_from_date.text()}, using year 0 as default")
            filter_date_from = 0
        try:
            filter_date_to = int(self.ui.dataset_filtering_to_date.text())
        except:
            print(f"could not convert date_from to int: {self.ui.dataset_filtering_to_date.text()}, using year 3000 as default")
            filter_date_to = 3000

        keys_to_keep = []
        for key, value in self.metadata.items():
            # if any of the nationalities match we consider it valid
            nationalities = value['artist_nationality'].split(",")
            nationalities = [n.strip() for n in nationalities]
            
            media = value['media'].split(",")
            media = [m.strip() for m in media]
            
            passed_nationality_filter = filter_nationality in nationalities or filter_nationality == "all"
            passed_media_filter = filter_media in media or filter_media == "all"
            passed_date_filter = int(value['date']) >= filter_date_from and int(value['date']) < filter_date_to

            if passed_nationality_filter and passed_media_filter and passed_date_filter:
                print("-" * 50)
                print(f"keeping {key}, with values: {value}")
                keys_to_keep.append(key)

        print(f"keeping {len(keys_to_keep)} indices after applying nationality: {filter_nationality} and media: {filter_media} and date: {filter_date_from}-{filter_date_to}")


        # self.filter_data_by_key(keys_to_keep)
        self.non_filtered_indices = [self.key_to_idx[img_hash] for img_hash in keys_to_keep]
        self.recalc_similarity()
        self.setup_scatterplot()

        self.ui.filtered_dataset_size.setText(f"{len(self.non_filtered_indices)}/{len(self.image_indices)}")


    def reset_data_filters(self):
        """reset filters. don't reset projections"""
        self.non_filtered_indices = self.image_indices
        self.setup_filters()
        self.apply_filters()
        
    def reload_data_dict(self):
        """reset filters and reload dataset. Also refresh scatterplot and similarity scores."""
        self.reset_data_filters()
        self.load_metric_data()
        if self.any_reprojection_applied:
            self.calc_combined_projection()
            self.any_reprojection_applied = False

        self.recalc_similarity()
        self.setup_scatterplot()

    def setup_scatterplot(self, clear_selection=True):
        current_metric_type = self.ui.box_metric_tabs.tabText(self.ui.box_metric_tabs.currentIndex()).lower()
        if current_metric_type not in self.available_features:
            print("ignoring tab change!, featuer: ")
            return
        
        print("changing 2d scatterplot to: ", current_metric_type)

        if not hasattr(self, 'scatterplot'):
            print(f'a new scatterplot is created for {current_metric_type}, passing points data of shape: {self.data_dict[current_metric_type]["projection"].shape}')
            self.scatterplot = ScatterplotWidget(
                self.data_dict[current_metric_type]["projection"], self.image_indices, self.image_paths, self.config, self.ui.scatterplot_frame, self.non_filtered_indices
            )
            self.scatterplot.plot_widget.scene().mousePressEvent=self.on_canvas_click
        else:
            print(f'redraw scatterplot for {current_metric_type}, passing points data of shape: {self.data_dict[current_metric_type]["projection"].shape}')
            self.scatterplot.points = self.data_dict[current_metric_type]["projection"]
            self.scatterplot.update_selected_points_values()
            self.scatterplot.indices_to_keep = self.non_filtered_indices
            if clear_selection:
                self.scatterplot.clear_selection()

            if 0 < len(self.scatterplot.selected_indices)<100:
                self.scatterplot.dots_plot=False
                self.scatterplot.draw_scatterplot()
            else:
                self.scatterplot.dots_plot=True
                self.scatterplot.draw_scatterplot_dots()


    def recalc_similarity(self):
        print('recalculating similarity')
        topk_dict = self.calculate_nearest_neighbours()
        self.display_nearest_neighbours(topk_dict)
        
    def texture_opt_dist_cos(self):
        self.texture_distance_measure = "cosine"
        self.recalc_similarity()
    
    def texture_opt_dist_euc(self):
        self.texture_distance_measure = "euclidian"
        self.recalc_similarity()

    def dino_opt_dist_cos(self):
        print('dino_opt_dist_cos')
        self.dino_distance_measure = "cosine"
        self.recalc_similarity()
    
    def dino_opt_dist_euc(self):
        print('dino_opt_dist_euc')
        self.dino_distance_measure = "euclidian"
        self.recalc_similarity()

    def emotion_opt_dist_cos(self):
        self.emotion_distance_measure = "cosine"
        self.recalc_similarity()
    
    def emotion_opt_dist_euc(self):
        self.emotion_distance_measure = "euclidian"
        self.recalc_similarity()

    def semantic_opt_dist_cos(self):
        self.semantic_distance_measure = "cosine"
        self.recalc_similarity()
    
    def semantic_opt_dist_euc(self):
        self.semantic_distance_measure = "euclidian"
        self.recalc_similarity()

    def clip_opt_simtype(self):
        if self.ui.clip_opt_fullsim.isChecked():
            self.clip_opt_sim_vector_type = "full"
        elif self.ui.clip_opt_2dsim.isChecked():
            self.clip_opt_sim_vector_type = "projection"
        else:
            raise ValueError("something is wrong with the clip similarity options.")
        
    def emotion_opt_simtype(self):
        if self.ui.emotion_opt_fullsim.isChecked():
            self.emotion_opt_sim_vector_type = "full"
        elif self.ui.emotion_opt_2dsim.isChecked():
            self.emotion_opt_sim_vector_type = "projection"
        else:
            raise ValueError("something is wrong with the emotion similarity options.")
        
    def texture_opt_simtype(self):
        if self.ui.texture_opt_fullsim.isChecked():
            self.texture_opt_sim_vector_type = "full"
        elif self.ui.texture_opt_2dsim.isChecked():
            self.texture_opt_sim_vector_type = "projection"
        else:
            raise ValueError("something is wrong with the texture similarity options.")
        
    def semantic_opt_simtype(self):
        if self.ui.semantic_opt_fullsim.isChecked():
            self.semantic_opt_sim_vector_type = "full"
        elif self.ui.semantic_opt_2dsim.isChecked():
            self.semantic_opt_sim_vector_type = "projection"
        else:
            raise ValueError("something is wrong with the semantic similarity options.")

    def dino_opt_simtype(self):
        if self.ui.dino_opt_fullsim.isChecked():
            self.dino_opt_sim_vector_type = "full"
        elif self.ui.dino_opt_2dsim.isChecked():
            self.dino_opt_sim_vector_type = "projection"
        elif self.ui.dino_opt_headsim.isChecked:
            l = self.ui.isChecked().currentText()
            h = self.ui.dino_opt_headsim_cbox.currentText()
            self.dino_opt_sim_vector_type = f"{l}_{h}"

        else:
            raise ValueError("something is wrong with the dino similarity options.")
        print(f"changed dino vector type to : {self.dino_opt_sim_vector_type}")


    def emotion_show_fm(self):
        if not self.ui.emotion_opts_showfm.isChecked():
            self.display_photo_left(self.left_img_filename)
            self.display_photo_right(self.right_img_filename)
            print("not checked", self.ui.emotion_opts_showfm)
        else:
            original_img_left = da.io.load_image(self.left_img_filename)
            original_img_right = da.io.load_image(self.right_img_filename)

            layer_key = 'emotion_fm'

            feature_maps_left = da.io.read_feature(
                    self.datafile_path, self.left_img_key, layer_key, read_projection=False
                ).squeeze()
            feature_maps_right = da.io.read_feature(
                    self.datafile_path, self.right_img_key, layer_key, read_projection=False
                ).squeeze()
            
            # Rescale heatmap to a range 0-255
            heatmap_left = np.uint8(255 * feature_maps_left)
            heatmap_right = np.uint8(255 * feature_maps_right)
            # Use jet colormap to colorize heatmap
            jet = cm.get_cmap("jet")

            # Use RGB values of the colormap
            jet_colors = jet(np.arange(256))[:, :3]
            jet_heatmap_left = jet_colors[heatmap_left]
            jet_heatmap_right = jet_colors[heatmap_right]

            # Create an image with RGB colorized heatmap
            jet_heatmap_left = keras.utils.array_to_img(jet_heatmap_left)
            jet_heatmap_left = jet_heatmap_left.resize((original_img_left.shape[1], original_img_left.shape[0]))
            jet_heatmap_left = keras.utils.img_to_array(jet_heatmap_left)

            jet_heatmap_right = keras.utils.array_to_img(jet_heatmap_right)
            jet_heatmap_right = jet_heatmap_right.resize((original_img_right.shape[1], original_img_right.shape[0]))
            jet_heatmap_right = keras.utils.img_to_array(jet_heatmap_right)

            # Superimpose the heatmap on original image
            alpha = 0.4
            superimposed_img_left = jet_heatmap_left * alpha + original_img_left
            superimposed_img_left = keras.utils.array_to_img(superimposed_img_left)

            superimposed_img_right = jet_heatmap_right * alpha + original_img_right
            superimposed_img_right = keras.utils.array_to_img(superimposed_img_right)

            # Save the superimposed image
            # cam_path += name + '.jpg'
            # superimposed_img.save(cam_path)

            # Display Grad CAM
            # display(Image(cam_path))
            leftname = "_tmp_overlay_left.png"
            rightname = "_tmp_overlay_right.png"
            superimposed_img_left.save(leftname)

            superimposed_img_right.save(rightname)

            self.display_photo_left(leftname)
            self.display_photo_right(rightname)

    
    def texture_show_fm(self):
        print('texture_show_fm')
        if not self.ui.texture_opt_show_fm.isChecked():
            self.display_photo_left(self.left_img_filename)
            self.display_photo_right(self.right_img_filename)
            print("not checked", self.ui.texture_opt_show_fm)
        else:
            # get selected head
            (layer, index) = self.ui.texture_opt_filtervis.currentText().split(" ")
            index = int(index) - 1
            if layer == "3a":
                layer_key = "texture_fm_3a"
            else:
                layer_key = "texture_fm_3b"
            
            feature_maps_left = da.io.read_feature(
                    self.datafile_path, self.left_img_key, layer_key, read_projection=False
                ).squeeze()[index]
            feature_maps_right = da.io.read_feature(
                    self.datafile_path, self.right_img_key, layer_key, read_projection=False
                ).squeeze()[index]
            
            # feature_maps_right = np.moveaxis(feature_maps_right[index], 0, -1)
            # feature_maps_left = np.moveaxis(feature_maps_left[index]

                
            original_img_left = da.io.load_image(self.left_img_filename)
            original_img_right = da.io.load_image(self.right_img_filename)
            l_h, l_w, l_c = original_img_left.shape
            r_h, r_w, l_c = original_img_right.shape

            print(original_img_left.shape)
            print(original_img_right.shape)
            print("-----------")
            print(feature_maps_left.shape)
            print(feature_maps_right.shape)

            feature_maps_left = cv2.resize(
                        feature_maps_left, dsize=(l_w, l_h), interpolation=cv2.INTER_NEAREST
                    )
            feature_maps_right = cv2.resize(
                        feature_maps_right, dsize=(r_w, r_h), interpolation=cv2.INTER_NEAREST
                    )
            print(feature_maps_left.shape)
            print(feature_maps_right.shape)
            print("-----------")
            
            heatmap, minmaxed = da.similarity_methods.heatmap_utils.feature_map_to_colormap(feature_maps_left)
            overlayed_left = da.similarity_methods.heatmap_utils.overlay_heatmap(original_img_left, heatmap, minmaxed)

            heatmap, minmaxed = da.similarity_methods.heatmap_utils.feature_map_to_colormap(feature_maps_right)
            overlayed_right = da.similarity_methods.heatmap_utils.overlay_heatmap(original_img_right, heatmap, minmaxed)
            
            leftname = "_tmp_overlay_left.png"
            rightname = "_tmp_overlay_right.png"
            left = Image.fromarray(overlayed_left)
            left.save(leftname)

            right = Image.fromarray(overlayed_right)
            right.save(rightname)

            self.display_photo_left(leftname)
            self.display_photo_right(rightname)

    def dino_show_camap(self):
        print('dino_show_camap')
        
        if not self.ui.dino_opt_showcamap.isChecked():
            self.display_photo_left(self.left_img_filename)
            self.display_photo_right(self.right_img_filename)
        else:
            # get selected head
            h = int(self.ui.dino_opt_headvis_cbox.currentText()) - 1
            l = int(self.ui.dino_opt_layervis_cbox.currentText()) - 1

            feature_maps_left = da.io.read_feature(
                    self.datafile_path, self.left_img_key, 'dino_fm', read_projection=False
                )[h][l]
            
            feature_maps_right = da.io.read_feature(
                    self.datafile_path, self.right_img_key, 'dino_fm', read_projection=False
                )[h][l]
            
            original_img_left = da.io.load_image(self.left_img_filename)
            original_img_right = da.io.load_image(self.right_img_filename)
            l_h, l_w, l_c = original_img_left.shape
            r_h, r_w, l_c = original_img_right.shape

            print(original_img_left.shape)
            print(original_img_right.shape)
            print("-----------")

            feature_maps_left = cv2.resize(
                        feature_maps_left, dsize=(l_w, l_h), interpolation=cv2.INTER_NEAREST
                    )
            feature_maps_right = cv2.resize(
                        feature_maps_right, dsize=(r_w, r_h), interpolation=cv2.INTER_NEAREST
                    )
            print(feature_maps_left.shape)
            print(feature_maps_right.shape)
            print("-----------")
            
            heatmap, minmaxed = da.similarity_methods.heatmap_utils.feature_map_to_colormap(feature_maps_left)
            overlayed_left = da.similarity_methods.heatmap_utils.overlay_heatmap(original_img_left, heatmap, minmaxed)

            heatmap, minmaxed = da.similarity_methods.heatmap_utils.feature_map_to_colormap(feature_maps_right)
            overlayed_right = da.similarity_methods.heatmap_utils.overlay_heatmap(original_img_right, heatmap, minmaxed)
            
            leftname = "_tmp_overlay_left.png"
            rightname = "_tmp_overlay_right.png"
            left = Image.fromarray(overlayed_left)
            left.save(leftname)

            right = Image.fromarray(overlayed_right)
            right.save(rightname)

            self.display_photo_left(leftname)
            self.display_photo_right(rightname)


    def set_color_element(self, ui_element, color):
        ui_element.setAutoFillBackground(True)
        p = ui_element.palette()
        p.setColor(ui_element.backgroundRole(), QColor(*color))
        ui_element.setPalette(p)
    
    def display_nearest_neighbours(self, topk):
        print('display_nearest_neighbours')

        # save for potential use in other parts of the program
        self.topk = topk
        try:
            distance, idx = topk['combined']["distances"][0], int(topk['combined']['ranking'][0])
                
            top_img_path = self.image_paths[idx]
            self.right_img_key = self.image_keys[idx]
            self.right_img_filename = self.image_paths[idx]
        # in case there is only 1 image and so no nearest neighbour
        except IndexError:
            top_img_path = self.left_img_filename

        self.display_photo_right(top_img_path)
        print(topk['combined']['distances'].shape)
        # if we cannot find nearest neighbours, we just display the original image again, edgecase handling
        try:
            distance_1, idx_1 = topk['combined']['distances'][1], int(topk['combined']["ranking"][1])
        except IndexError:
            idx_1 = self.key_to_idx[self.left_img_key]
            distance_1 = 0
        try:
            distance_2, idx_2 = topk['combined']['distances'][2], int(topk['combined']["ranking"][2])
        except IndexError:
            idx_2 = self.key_to_idx[self.left_img_key]
            distance_2 = 0
        try:
            distance_3, idx_3 = topk['combined']['distances'][3], int(topk['combined']["ranking"][3])
        except IndexError:
            idx_3 = self.key_to_idx[self.left_img_key]
            distance_3 = 0

        indices_nn_preview = [idx_1, idx_2, idx_3]
        print(f"{indices_nn_preview=}")
        print(f"{distance_1=}")
        print(f"{distance_2=}")
        print(f"{distance_3=}")
        fp_nn_preview = [self.image_paths[int(index)] for index in indices_nn_preview]
        print(f"{fp_nn_preview=}")

        self.display_preview_nns(fp_nn_preview)
        

    def display_preview_nns(self, filenames):
        for i, filename in enumerate(filenames[:3]):
            ui_element = getattr(self.ui, f"n{i+1}")
            ui_element.setAutoFillBackground(True)
            p = ui_element.palette()
            p.setColor(ui_element.backgroundRole(), QColor(0, 0, 0))
            ui_element.setPalette(p)

            w, h = ui_element.width(), ui_element.height()
            pixmap = QPixmap(filename)
            ui_element.setPixmap(pixmap.scaled(w,h,Qt.AspectRatioMode.KeepAspectRatio))
            ui_element.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
    def get_metric_combo_weights(self):
        # dummy = self.ui.combo_dummy_slider.value()
        dino = self.ui.combo_dino_slider.value()
        texture = self.ui.combo_texture_slider.value()
        emotion = self.ui.combo_emotion_slider.value()
        semantic = self.ui.combo_semantic_slider.value()
        clip = self.ui.combo_clip_slider.value()
        # TODO: add sliders for the other metrics
        feature_weight_dict = {
            # "dummy": dummy / 100,
            "dino": dino / 100,
            "texture": texture / 100,
            "emotion": emotion / 100,
            "semantic": semantic / 100,
            "clip": clip / 100
        }
        return feature_weight_dict
    
    def euclidian_distance(self, current_vector, feature_name, vector_type_key):
        a_min_b = current_vector.reshape(-1, 1) - self.data_dict[feature_name][vector_type_key].T
        distances = np.sqrt(np.einsum('ij,ij->j', a_min_b, a_min_b))
        return distances
    
    def cosine_distance(self, current_vector, feature_name, vector_type_key):
        distances = cosine_distances(current_vector.reshape(1, -1), self.data_dict[feature_name][vector_type_key]).squeeze()
        return distances
    
    def calculate_nearest_neighbours(self, topk=5):
        # get features for current image
        topk_results = {}
        distances_dict = {}
        
        feature_weight_dict = self.get_metric_combo_weights()
        print("weights of all metrics: ", feature_weight_dict)
        
        indices = np.arange(len(self.image_keys))

        # apply filter mask
        indices = indices[self.non_filtered_indices]

        for feature_name in self.available_features:
            
            # weither to use full vector for similarity, only a specific part or the 2d reprojection
            if feature_name == "dino":
                vector_type_key = self.dino_opt_sim_vector_type
            elif feature_name == "clip":
                vector_type_key = self.clip_opt_sim_vector_type
            elif feature_name == "texture":
                vector_type_key = self.texture_opt_sim_vector_type
            elif feature_name == "emotion":
                vector_type_key = self.emotion_opt_sim_vector_type
            elif feature_name == "semantic":
                vector_type_key = self.semantic_opt_sim_vector_type
            else:
                # TODO: implement additional options for other metrics to use projection or not
                # for metric similarity
                vector_type_key = "full"
                
            print(f"calculating {feature_name} with {vector_type_key} vectors ")
            current_vector = self.left_img_features[feature_name][vector_type_key]
            distances = np.zeros((self.data_dict[feature_name][vector_type_key].shape[0]))
            
            # calculate distances
            if feature_name == "dino":
                if self.dino_distance_measure == "cosine":
                    distances = self.cosine_distance(current_vector, feature_name, vector_type_key)
                if self.dino_distance_measure == "euclidian":
                    distances = self.euclidian_distance(current_vector, feature_name, vector_type_key)
            elif feature_name == "texture":
                if self.texture_distance_measure == "cosine":
                    distances = self.cosine_distance(current_vector, feature_name, vector_type_key)
                if self.texture_distance_measure == "euclidian":
                    distances = self.euclidian_distance(current_vector, feature_name, vector_type_key)
            elif feature_name == "semantic":
                if self.semantic_distance_measure == "cosine":
                    distances = self.cosine_distance(current_vector, feature_name, vector_type_key)
                if self.semantic_distance_measure == "euclidian":
                    distances = self.euclidian_distance(current_vector, feature_name, vector_type_key)
            elif feature_name == "emotion":
                if self.emotion_distance_measure == "cosine":
                    distances = self.cosine_distance(current_vector, feature_name, vector_type_key)
                if self.emotion_distance_measure == "euclidian":
                    distances = self.euclidian_distance(current_vector, feature_name, vector_type_key)

            elif feature_name == "clip":
                if self.ui.clip_radio_textsim.isChecked():
                    text = self.ui.tb_clip_input.toPlainText()
                    print(f"using text similarity, similar to: '{text}'")
                    text = clip.tokenize([text]).to(self.device)
                    current_vector = self.clip_model.encode_text(text).squeeze().cpu().detach().numpy()

                    # TODO actually implement this
                    if vector_type_key == "projection":
                        current_vector = self.reducers['clip'].transform(current_vector)

                    distances = self.cosine_distance(current_vector, feature_name, vector_type_key)
                elif self.ui.clip_radio_imgsim.isChecked():
                    # use the normal image based current_vector as retrieved above
                    distances = self.cosine_distance(current_vector, feature_name, vector_type_key)
                elif self.ui.clip_radio_combsim.isChecked():
                    print("using combined similarity!")
                    text = self.ui.tb_clip_input.toPlainText()
                    text = clip.tokenize([text]).to(self.device)
                    current_text_vector = self.clip_model.encode_text(text).squeeze().cpu().detach().numpy()

                    distances_text = self.cosine_distance(current_text_vector, feature_name, vector_type_key)
                    distances_img = self.cosine_distance(current_vector, feature_name, vector_type_key)
                    # we can just sum as they should be in the same scale
                    distances = distances_text + distances_img 

            else:
                # by default we use euclidian distance for a feature
                print("defaulting to euclidian distance")
                distances = self.euclidian_distance(current_vector, feature_name, vector_type_key)
            
            # rescale distances so that the distances are always within the range of 0-1
            # this way we can combine them, the element with distance 0 is the image itself if it's 
            # in dataset, otherwise it's the nearest neighbour
            distances = minmax_scale(distances, axis=0, feature_range=(0, 1), copy=False)

            distances_dict[feature_name] = distances

            distances = distances[self.non_filtered_indices]

            sorting_indices = distances.argsort()
            sorted_distances = distances[sorting_indices]
            ranking_feature = indices[sorting_indices]

            # sorted_distances = distances[distances[:, 0].argsort()]
            # sorted_distances = sorted_distances[sorted_distances[:,0] > 1e-20]
            
            # remove images with distance (almost) 0 as those are just the original image
            # but only do so if we know the current image is in the dataset
            if self.left_img_key in self.image_keys:
                sorted_distances = sorted_distances[1:]
                ranking_feature = ranking_feature[1:]


            topk_results[feature_name] = {}
            topk_results[feature_name]['distances'] = sorted_distances
            topk_results[feature_name]['ranking'] = ranking_feature
            
        all_distances = np.stack(list(distances_dict.values()), axis=-1)

        if feature_weight_dict is not None:
            weights = [feature_weight_dict[feature_name] for feature_name in self.available_features]
            weights = np.array(weights)

            all_distances_sum = np.average(all_distances, axis=1, weights=weights)
        else:
            all_distances_sum = np.average(all_distances, axis=1)

        # apply filter mask
        all_distances_sum = all_distances_sum[self.non_filtered_indices]

        sorting_indices = all_distances_sum.argsort()
        combined_ranking = indices[sorting_indices]
        combined_distances = all_distances_sum[sorting_indices]

        if self.left_img_key in self.image_keys:
            combined_distances = combined_distances[1:]
            combined_ranking = combined_ranking[1:]
        
        topk_results["combined"] = {}
        topk_results["combined"]['ranking'] = combined_ranking
        topk_results["combined"]['distances'] = combined_distances

        return topk_results

    def upload_image_left(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        file_dialog.setNameFilter("Images (*.png *.xpm *.jpg *.jpeg)")

        if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
            filenames = file_dialog.selectedFiles()
            if len(filenames) > 0:
                # current_filepath = self.image_paths[self.key_to_idx[img_hash]]
                # we take only one file/ first file
                current_filepath=[filenames[0]]
                print('current_filepath', current_filepath)
                
                # img_hash = da.get_image_hash(current_filepath)
                # print(f"hash: {img_hash}")
                # print('does this hash already exist?', img_hash in self.image_keys)
                # self.left_img_key = img_hash
                # self.left_img_filename = current_filepath

                dataset_filepath = str( Path(f"{__file__}").parents[1] / "data" / "processed" / "dataset.h5")
                for feature in self.available_features:
                    # if feature == "dummy":
                    #     da.dummy.calc_and_save_features(current_filepath, dataset_filepath)
                    if feature == "dino":
                        da.dino.calc_and_save_features(current_filepath, dataset_filepath)
                    elif feature == "texture":
                        da.texture.calc_and_save_features(current_filepath, dataset_filepath)
                    elif feature == "emotion":
                        da.emotion.calc_and_save_features(current_filepath, dataset_filepath)
                    elif feature == "semantic":
                        da.semantic.calc_and_save_features(current_filepath, dataset_filepath)

                # display the photo on the left
                self.display_photo_left(current_filepath)
                self.update_leftimg_data(self.left_img_key)

                self.recalc_similarity()


    def update_leftimg_data(self, img_hash):
        # update image metdata if available
        if img_hash in self.metadata:
            filepath = self.image_paths[self.key_to_idx[img_hash]]
            print(f"found metadata for image: {filepath}")
            self.update_image_info(
                self.metadata[img_hash]['date'],
                self.metadata[img_hash]['artist_name'],
                self.metadata[img_hash]['style'],
                self.metadata[img_hash]['tags'],
            )
            self.left_img_features = self.get_features_from_dataset(img_hash)

            # load in timeline
            base_filename = os.path.basename(filepath)
            self.timeline.draw_timeline(base_filename)
            self.no_timeline_label.hide()
            self.timeline.show()
        else:
            filepath = self.image_paths[self.key_to_idx[img_hash]]
            # get feature_vectors for new image 
            self.left_img_features = self.get_point_new_img(filepath)

            print(f"no metadata available for image: {filepath}")
            self.update_image_info("unknown date", "unknown artist", "unknown style", "unknown tags")
            self.timeline.hide()
            self.no_timeline_label.show()


    def get_features_from_dataset(self, img_hash):
        feature_dict = {}
        for feature_name in self.available_features:
            # get feature for default image
            test_feature = da.io.read_feature(
                self.datafile_path, img_hash, feature_name, read_projection=False
            )
            test_feature_p = da.io.read_feature(
                self.datafile_path, img_hash, feature_name, read_projection=True
            )

            # left image data is always saved seperately 
            feature_dict[feature_name] = {
                    "full": test_feature,
                    "projection": test_feature_p
                }
        return feature_dict

    def get_point_new_img(self, filename):
        print(f"given filename: {filename}, ignoring file for now and returning feature_vector")
        feature_dict = {}
        for feature_name in self.data_dict.keys():
            vector_size =  self.data_dict[feature_name]["projection"].shape[1]
            random_array = np.random.uniform(low=-10.0, high=10.0, size=(vector_size,))
            random_array_p = np.random.uniform(low=-10.0, high=10.0, size=(2,))

            feature_dict[feature_name] = {}
            feature_dict[feature_name]["projection"] = random_array_p
            feature_dict[feature_name]["full"] = random_array

        print('random_array', random_array)
        image_features = random_array
        new_point = image_features
        return new_point

    def update_image_info(self, date, artist, style, tags):
        # Update the label texts
        self.ui.t_date.setText(f"Date: {int(date)}")
        self.ui.t_artist.setText(f"Artist: {artist}")
        self.ui.t_style.setText(f"Style: {style}")
        self.ui.t_tags.setText(f"Tags: {tags}")
        
    def display_photo_left(self, filename):
        img_container = self.ui.box_left_img
        img_container.setAutoFillBackground(True)
        p = img_container.palette()
        p.setColor(img_container.backgroundRole(), QColor(0, 0, 0))
        img_container.setPalette(p)

        w, h = img_container.width(), img_container.height()
        print('displaying photo:', filename)
        pixmap = QPixmap(filename)
        self.ui.box_left_img.setPixmap(pixmap.scaled(w,h,Qt.AspectRatioMode.KeepAspectRatio))
        self.ui.box_left_img.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def display_photo_right(self, filename):
        img_container = self.ui.box_right_img
        img_container.setAutoFillBackground(True)
        p = img_container.palette()
        p.setColor(img_container.backgroundRole(), QColor(0, 0, 0))
        img_container.setPalette(p)

        w, h = img_container.width(), img_container.height()
        print('displaying photo on right:', os.path.basename(filename))
        self.top_pixmap = QPixmap(filename)
        self.top_filename= filename
        self.ui.box_right_img.setPixmap(self.top_pixmap.scaled(w,h,Qt.AspectRatioMode.KeepAspectRatio))
        self.ui.box_right_img.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def display_nearest_neighbours(self, topk):
        # save for potential use in other parts of the program
        self.topk = topk
        try:
            distance, idx = topk['combined']["distances"][0], int(topk['combined']['ranking'][0])
                
            top_img_path = self.image_paths[idx]
            self.right_img_key = self.image_keys[idx]
            self.right_img_filename = self.image_paths[idx]
        # in case there is only 1 image and so no nearest neighbour
        except IndexError:
            top_img_path = self.left_img_filename

        self.display_photo_right(top_img_path)
        print(topk['combined']['distances'].shape)
        # if we cannot find nearest neighbours, we just display the original image again, edgecase handling
        try:
            distance_1, idx_1 = topk['combined']['distances'][1], int(topk['combined']["ranking"][1])
        except IndexError:
            idx_1 = self.key_to_idx[self.left_img_key]
            distance_1 = 0
        try:
            distance_2, idx_2 = topk['combined']['distances'][2], int(topk['combined']["ranking"][2])
        except IndexError:
            idx_2 = self.key_to_idx[self.left_img_key]
            distance_2 = 0
        try:
            distance_3, idx_3 = topk['combined']['distances'][3], int(topk['combined']["ranking"][3])
        except IndexError:
            idx_3 = self.key_to_idx[self.left_img_key]
            distance_3 = 0

        indices_nn_preview = [idx_1, idx_2, idx_3]
        # print(f"{indices_nn_preview=}")
        # print(f"{distance_1=}")
        # print(f"{distance_2=}")
        # print(f"{distance_3=}")
        fp_nn_preview = [self.image_paths[int(index)] for index in indices_nn_preview]
        # print(f"{fp_nn_preview=}")

        self.display_preview_nns(fp_nn_preview)
        

    def display_preview_nns(self, filenames,idx=[0,1,2]):
        print('display_preview_nns')
        if len(filenames)>3:
            filenames=filenames[:3]

        for i, filename in enumerate(filenames):
            id= idx[i]
            filename= filenames[i]
            ui_element = getattr(self.ui, f"n{id+1}")
            ui_element.setMouseTracking(True)
            # ui_element.mousePressEvent=self.switch_top_and_preview(i, filename)
            ui_element.mousePressEvent = lambda event, id=id, filename=filename: self.switch_top_and_preview(id, filename)
            # print('id',id+1,'filename', os.path.basename(filename))

            ui_element.setAutoFillBackground(True)
            p = ui_element.palette()
            p.setColor(ui_element.backgroundRole(), QColor(0, 0, 0))
            ui_element.setPalette(p)

            w, h = ui_element.width(), ui_element.height()
            pixmap = QPixmap(filename)
            ui_element.setPixmap(pixmap.scaled(w,h,Qt.AspectRatioMode.KeepAspectRatio))
            ui_element.setAlignment(Qt.AlignmentFlag.AlignCenter)


    def switch_top_and_preview(self, i, filename):
        print('switch_top_and_preview')
        print('self.top_filename', os.path.basename(self.top_filename))
        self.display_preview_nns([self.top_filename], idx=[i])
        self.display_photo_right(filename)


    def on_canvas_click(self, ev):
        # super().mousePressEvent(ev)
        self.scatterplot.clear_selection()
        pos = ev.scenePos()
        print("on canvas click:", pos)
        if ev.button() == Qt.MouseButton.LeftButton:
            if self.scatterplot.dots_plot:
                range_radius = 0.1
                for i, index,dot_item in self.scatterplot.dot_items:
                    item_pos = dot_item.mapFromScene(pos)
                    x_data, y_data = dot_item.getData()
                    # print('pos:', pos, 'item_pos:', item_pos)
                    for x, y in zip(x_data, y_data):
                        if abs(x - item_pos.x()) <= range_radius and abs(y - item_pos.y()) <= range_radius:
                            self.scatterplot.selected_point = item_pos.x(), item_pos.y()
                            print('self.scatterplot.selected_point',self.scatterplot.selected_point)
                            self.scatterplot.remove_highlight_selected_point(self.scatterplot.selected_index)
                            self.scatterplot.selected_index = index
                            self.scatterplot.highlight_selected_point(index)
                            self.clicked_on_point()
                            break
            else:
                # TODO: rmv i after merge and its still not used
                for i, index, item in self.scatterplot.image_items:
                    item_pos=item.mapFromScene(pos)
                    if item.contains(item_pos):
                        print('selected_index==plot_index?',index==i)
                        self.scatterplot.selected_point = item_pos.x(), item_pos.y()
                        self.scatterplot.remove_highlight_selected_point(self.scatterplot.selected_index)
                        self.scatterplot.selected_index = index
                        self.scatterplot.highlight_selected_point(index)
                        self.clicked_on_point()
                        break
                

    def clicked_on_point(self):
        print("point/ image clicked, load on the left")
        self.left_img_filename = self.image_paths[self.scatterplot.selected_index]
        self.left_img_key = self.image_keys[self.scatterplot.selected_index]
        self.left_img_features = self.get_features_from_dataset(self.left_img_key)
        self.display_photo_left(self.left_img_filename)
        self.update_leftimg_data(self.left_img_key)
        self.recalc_similarity()

 
    # TODO: put in barplot_widget.py?
    def get_selected_points_stats(self, int):
        print('get_selected_points_stats')
        print(len(self.scatterplot.selected_indices), len(self.scatterplot.indices))
        img_hashes = [self.image_keys[index] for index in self.scatterplot.selected_indices]

        sel_unique_dates, sel_date_counts = np.unique([self.metadata[hash_]['date'] for hash_ in img_hashes], return_counts=True)
        sel_unique_tags, sel_tag_counts = np.unique([self.metadata[hash_]['tags'] for hash_ in img_hashes], return_counts=True) # could have more than one
        sel_unique_artist_names, sel_artist_name_counts = np.unique([self.metadata[hash_]['artist_name'] for hash_ in img_hashes], return_counts=True)
        sel_unique_nationalities, sel_nationalities_counts = np.unique([self.metadata[hash_]['artist_nationality'] for hash_ in img_hashes], return_counts=True)
        sel_unique_media, sel_media_counts = np.unique([self.metadata[hash_]['media'] for hash_ in img_hashes], return_counts=True) # could have more than one
        sel_unique_styles, sel_style_counts = np.unique([self.metadata[hash_]['style'] for hash_ in img_hashes], return_counts=True)

        img_hashes = [self.image_keys[index] for index in self.scatterplot.indices]

        unique_dates, date_counts = np.unique([self.metadata[hash_]['date'] for hash_ in img_hashes], return_counts=True)
        unique_tags, tag_counts = np.unique([self.metadata[hash_]['tags'] for hash_ in img_hashes], return_counts=True)
        unique_artist_names, artist_name_counts = np.unique([self.metadata[hash_]['artist_name'] for hash_ in img_hashes], return_counts=True)
        unique_nationalities, nationalities_counts = np.unique([self.metadata[hash_]['artist_nationality'] for hash_ in img_hashes], return_counts=True)
        unique_media, media_counts = np.unique([self.metadata[hash_]['media'] for hash_ in img_hashes], return_counts=True)
        unique_styles, style_counts = np.unique([self.metadata[hash_]['style'] for hash_ in img_hashes], return_counts=True)

        sel_date_bins, sel_date_bin_counts,date_bins, date_bin_counts = self.make_date_bins(sel_unique_dates,sel_date_counts,unique_dates,date_counts)
        tag_count_selection = [sel_tag_counts[np.where(sel_unique_tags == tag)[0].tolist()[0]] if np.isin(tag , sel_unique_tags) else 0 for tag in unique_tags]
        artist_count_selection = [sel_artist_name_counts[np.where(sel_unique_artist_names == artist_name)[0].tolist()[0]] if np.isin(artist_name, sel_unique_artist_names) else 0 for artist_name in unique_artist_names]
        nationalities_count_selection = [sel_nationalities_counts[np.where(sel_unique_nationalities == nationalities)[0].tolist()[0]] if np.isin(nationalities, sel_unique_nationalities) else 0 for nationalities in unique_nationalities]
        media_count_selection = [sel_media_counts[np.where(sel_unique_media == media)[0].tolist()[0]] if np.isin(media, sel_unique_media) else 0 for media in unique_media]
        style_count_selection = [sel_style_counts[np.where(sel_unique_styles == style)[0].tolist()[0]] if np.isin(style, sel_unique_styles) else 0 for style in unique_styles]
       
        self.bp.fill_in_barplot(unique_styles,style_counts,style_count_selection)
        self.bp2.fill_in_barplot(date_bins,date_bin_counts,sel_date_bin_counts)
        self.bp3.fill_in_barplot(unique_nationalities,nationalities_counts,nationalities_count_selection)
        
        
    def make_date_bins(self,sel_unique_dates,sel_date_counts,unique_dates,date_counts):
        bin_size = 20  
        #round to the nearest tens
        start_year = np.floor(unique_dates.min() / 10) * 10
        end_year = np.ceil(unique_dates.max() / 10) * 10
        print(start_year, end_year)

        date_count_selection = [sel_date_counts[np.where(sel_unique_dates == date)[0].tolist()[0]] if np.isin(date , sel_unique_dates) else 0 for date in unique_dates]

        num_bins = int((end_year - start_year) / bin_size) + 1
        print('num_bins', num_bins)

        # Create an array of bin edges
        bin_edges = np.arange(start_year, end_year + bin_size, bin_size)
        # print('bin_edges', bin_edges)

        # Assign each date to a bin using digitize
        bin_indices = np.digitize(unique_dates, bin_edges)- 1
        # print(len(bin_indices))
        # print(bin_indices)

        # Initialize an array to store the counts for each bin
        sel_bin_counts = np.zeros(num_bins)
        bin_counts = np.zeros(num_bins)

        # Count dates within each bin
        for i in range(num_bins):
            indices = np.where(bin_indices == i)[0].astype(int)
            # print('indices',indices, type(indices))
            # print('date_count_selection', type(date_count_selection))
            date_count_selection = np.array(date_count_selection)
            date_counts = np.array(date_counts)
            if len(indices) > 0:
                sel_bin_counts[i] = np.sum(date_count_selection[indices])
                bin_counts[i] = np.sum(date_counts[indices])
        # print(sel_bin_counts)
        # print(bin_counts)

        # Print the bins and their corresponding counts
        for i in range(num_bins):
            bin_start = bin_edges[i]
            bin_end = bin_edges[i + 1] - 1 if i < num_bins - 1 else end_year
            bin_count = bin_counts[i]
            # print(f"Bin {i+1}: {bin_start}-{bin_end} - Count: {bin_count}")

        return bin_edges,sel_bin_counts,bin_edges ,bin_counts


def start_dashboard(key_dict, dataset_filepath, images_filepath, features_to_use):
    app = QApplication(sys.argv)
    basepath = Path(__file__)
    css_filepath = str(basepath.parent / "theme1.css")
    with open(css_filepath, "r") as file:
        app.setStyleSheet(file.read())

    widget = MainWindow(key_dict, dataset_filepath, images_filepath, features_to_use)
    widget.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    key_dict = {
        "8a2d5cacda630af528690878ed1c6224" : "pierre-auguste-renoir_young-girl-combing-her-hair-1894.jpg"
    }

    basepath = Path(__file__)
    css_filepath = str(basepath.parent / "theme1.css")
    with open(css_filepath, "r") as file:
        app.setStyleSheet(file.read())

    datafile_path = "/home/parting/master_AI/MMA/deepsim-analyzer/data/processed/dataset.h5"
    images_filepath = "/home/parting/master_AI/MMA/deepsim-analyzer/data/raw_immutable/test_images"

    widget = MainWindow(key_dict, datafile_path, images_filepath)
    widget.show()
    sys.exit(app.exec())

