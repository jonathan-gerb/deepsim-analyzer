# This Python file uses the following encoding: utf-8
import sys

# normal imports
from pathlib import Path
import numpy as np
from PIL import Image
import configparser
from scipy import spatial
from sklearn.preprocessing import minmax_scale
from sklearn.metrics.pairwise import cosine_distances
import os
from tqdm import tqdm
import cv2

# qt imports
from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QMainWindow, QFileDialog, QApplication, QVBoxLayout, QTabWidget,QGraphicsView
from PyQt6.QtGui import QPixmap, QPainter, QColor
from PyQt6.QtCore import QRect, Qt

# custom widgets
from .custom_widgets import  ScatterplotWidget, TimelineView, TimelineWindow

# deepsim analyzer package
import deepsim_analyzer as da

# Important:
# You need to run the following command to generate the ui_form.py file
#     pyside6-uic form.ui -o ui_form.py, or
#     pyside2-uic form.ui -o ui_form.py
# additionally run python fix_ui_script.py, which replaces the not working ui stuff.
from .ui_form import Ui_MainWindow

class MainWindow(QMainWindow):
    def __init__(self, key_dict, datafile_path, images_filepath):
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
        self.image_keys = [key for key in key_dict.keys()] # hashes
        self.key_to_idx = {key:i for i, key in enumerate(key_dict.keys())} # hashes
        self.image_indices = [i for i in range(len(key_dict))]
        self.image_paths = [str(Path(images_filepath) / image_name) for image_name in key_dict.values()]

        # in time we have to get all features for all the data, we will start with
        # just the dummy feature
        self.available_features = ["dummy", "texture", "dino"]

        # metric option defaults
        self.dino_distance_measure = "euclidian"
        self.texture_distance_measure = "euclidian"
        self.dino_opt_sim_vector_type = "full"

        # set color for main ui
        self.set_color_element(self.ui.centralwidget, [71, 71, 71])
        # set color for lower text boxes
        self.set_color_element(self.ui.box_left_low, [143, 143, 143])
        self.set_color_element(self.ui.box_right_low, [143, 143, 143])
        self.set_color_element(self.ui.box_metainfo, [143, 143, 143])
        
        # set tab ui color
        self.set_color_element(self.ui.box_metric_tabs, [143, 143, 143])
        self.set_color_element(self.ui.dummy_tab, [143, 143, 143])
        self.set_color_element(self.ui.dino_tab, [143, 143, 143])
        self.set_color_element(self.ui.texture_tab, [143, 143, 143])
        self.set_color_element(self.ui.emotion_tab, [143, 143, 143])

        # ================ SETUP DATA ================
        print("setting up internal data")
        self.metadata = {}

        # read metadata for all image keys that were passed
        print("reading metadata")
        self.metadata = da.read_metadata_batch(datafile_path, self.image_keys)

        # data dict should contain all the data
        self.data_dict = {}
        # init datadict

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

        # ================ SETUP LEFT COLUMN ================
        print("setting up left column of dashboard")
        # ---------------- STARTING IMG ----------------
        # load an initial first image to display
        default_image_key = list(key_dict.keys())[0]
        default_image_path = key_dict[default_image_key]
        default_image_path_absolute = str(Path(images_filepath) / default_image_path)
        
        self.left_img_key = default_image_key
        self.left_img_filename = default_image_path_absolute

        # display the base image
        self.display_photo_left(default_image_path_absolute)
        # data for left img feature, can come  from dataset or be calculated on the fly
        self.update_leftimg_data(self.left_img_key)
        # add additional data in box_left_low

        # ================ SETUP MIDDLE COLUMN ================
        print("setting up scatterplot")
        # setup scatterplot
        # TODO: setup feature projection plot with combined and individual plots!
        self.ui.box_metric_tabs.currentChanged.connect(self.setup_scatterplot)
        # And setup up once to initialize
        self.setup_scatterplot()
    
        # toggle the the dots to images radio button
        self.ui.r_image_points.toggle()
        self.ui.r_image_points.toggled.connect(self.change_scatterplot_pointtype)


        print("setting up middle metric options")
        # SETUP TEXTURE OPTIONs
        # options for what distance measure to use.
        self.ui.texture_opt_cosdist.toggled.connect(self.texture_opt_dist_cos)
        self.ui.texture_opt_eucdist.toggled.connect(self.texture_opt_dist_euc)
        self.ui.texture_opt_eucdist.toggle()

        # add options for head similarity to comboboxes
        for i in range(256):
            self.ui.texture_opt_filtervis.addItem(f"3a {i+1}")

        for i in range(480):
            self.ui.texture_opt_filtervis.addItem(f"3b {i+1}")

        self.ui.texture_opt_filtervis.currentIndexChanged.connect(self.texture_show_fm)
        self.ui.texture_opt_show_fm.toggled.connect(self.texture_show_fm)

        # SETUP DINO OPTIONS

        # options for what distance measure to use.
        self.ui.dino_opt_cosdist.toggled.connect(self.dino_opt_dist_cos)
        self.ui.dino_opt_eucdist.toggled.connect(self.dino_opt_dist_euc)
        self.ui.dino_opt_eucdist.toggle()

        # options for calculating similarity based on what vector
        self.ui.dino_opt_2dsim.toggled.connect(self.dino_opt_simtype)
        self.ui.dino_opt_fullsim.toggled.connect(self.dino_opt_simtype)
        self.ui.dino_opt_headsim.toggled.connect(self.dino_opt_simtype)
        self.ui.dino_opt_fullsim.toggle()

        self.ui.dino_opt_headvis_cbox.currentIndexChanged.connect(self.dino_show_camap)
        self.ui.dino_opt_layervis_cbox.currentIndexChanged.connect(self.dino_show_camap)

        # dropdown options for dino head-specific similarity
        self.ui.dino_opt_headsim_cbox.editTextChanged.connect(self.dino_opt_simtype)
        self.ui.dino_opt_layersim_cbox.editTextChanged.connect(self.dino_opt_simtype)

        # option for showing crossattention map
        self.ui.dino_opt_showcamap.toggled.connect(self.dino_show_camap)

        # add options for head similarity to comboboxes
        for i in range(12):
            self.ui.dino_opt_headsim_cbox.addItem(f"{i+1}")
            self.ui.dino_opt_layersim_cbox.addItem(f"{i+1}")
            self.ui.dino_opt_headvis_cbox.addItem(f"{i+1}")
            self.ui.dino_opt_layervis_cbox.addItem(f"{i+1}")

        self.ui.box_metric_tabs.setCurrentIndex(0)

        # ================ SETUP RIGHT COLUMN ================
        print("setting up right column, calculating nearest neighbours")
        topk_dict = self.calculate_nearest_neighbours()
        self.display_nearest_neighbours(topk_dict)
        
        print("recalculating")
        self.ui.recalc_similarity.toggled.connect(self.recalc_similarity)
        print("dashboard setup complete!")
      

    def setup_scatterplot(self):
        current_metric_type = self.ui.box_metric_tabs.tabText(self.ui.box_metric_tabs.currentIndex())
        print("changing 2d scatterplot to: ", current_metric_type)
        if not hasattr(self, 'scatterplot'):
            print('a new scatterplot is created')
            self.scatterplot = ScatterplotWidget(
                self.data_dict[current_metric_type.lower()]["projection"], self.image_indices, self.image_paths, self.config, self.ui.scatterplot_frame
            )
            self.scatterplot.plot_widget.scene().mousePressEvent=self.on_canvas_click
            self.scatterplot.selected_idx.emit(0)
        else:
            print('only redraw scatterplot')
            if self.scatterplot.dots_plot:
                self.scatterplot.draw_scatterplot_dots()
            else:
                self.scatterplot.draw_scatterplot()
            self.scatterplot.selected_idx.emit(self.scatterplot.selected_index)
            

    def recalc_similarity(self):
        topk_dict = self.calculate_nearest_neighbours()
        self.display_nearest_neighbours(topk_dict)
        
    def texture_opt_dist_cos(self):
        self.texture_distance_measure = "cosine"
        topk_dict = self.calculate_nearest_neighbours()
        self.display_nearest_neighbours(topk_dict)
    
    def texture_opt_dist_euc(self):
        self.texture_distance_measure = "euclidian"
        topk_dict = self.calculate_nearest_neighbours()
        self.display_nearest_neighbours(topk_dict)

    def dino_opt_dist_cos(self):
        self.dino_distance_measure = "cosine"
        topk_dict = self.calculate_nearest_neighbours()
        self.display_nearest_neighbours(topk_dict)
    
    def dino_opt_dist_euc(self):
        self.dino_distance_measure = "euclidian"
        topk_dict = self.calculate_nearest_neighbours()
        self.display_nearest_neighbours(topk_dict)

    def dino_opt_simtype(self):
        if self.ui.dino_opt_fullsim.isChecked:
            self.dino_opt_sim_vector_type = "full"
        elif self.ui.dino_opt_2dsim.isChecked:
            self.dino_opt_sim_vector_type = "projection"
        elif self.ui.dino_opt_headsim.isChecked:
            l = self.ui.dino_opt_layersim_cbox.currentText()
            h = self.ui.dino_opt_headsim_cbox.currentText()
            self.dino_opt_sim_vector_type = f"{l}_{h}"

        else:
            raise ValueError("something is wrong with the dino similarity options.")
    

    def texture_show_fm(self):
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
        # save for potential use in other parts of the program
        self.topk = topk

        distance, idx = topk['combined']["distances"][0], int(topk['combined']['ranking'][0])
        top_img_path = self.image_paths[idx]
        self.right_img_key = self.image_keys[idx]
        self.right_img_filename = self.image_paths[idx]

        self.display_photo_right(top_img_path)
        print(topk['combined']['distances'].shape)
        distance_1, idx_1 = topk['combined']['distances'][1], int(topk['combined']["ranking"][1])
        distance_2, idx_2 = topk['combined']['distances'][2], int(topk['combined']["ranking"][2])
        distance_3, idx_3 = topk['combined']['distances'][3], int(topk['combined']["ranking"][3])

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
        dummy = self.ui.combo_dummy_slider.value()
        dino = self.ui.combo_dino_slider.value()
        texture = self.ui.combo_texture_slider.value()
        feature_weight_dict = {
            "dummy": dummy / 100,
            "dino": dino / 100,
            "texture": texture / 100
        }
        return feature_weight_dict

    def calculate_nearest_neighbours(self, topk=5):
        # get features for current image
        topk_results = {}
        distances_dict = {}
        
        feature_weight_dict = self.get_metric_combo_weights()
        print("weights of all metrics: ", feature_weight_dict)
        
        indices = np.arange(len(self.image_keys))
        for feature_name in self.available_features:
            
            # weither to use full vector for similarity, only a specific part or the 2d reprojection
            if feature_name == "dino":
                vector_type_key = self.dino_opt_sim_vector_type
            else:
                # TODO: implement additional options for other metrics to use projection or not
                # for metric similarity
                vector_type_key = "full"

            current_vector = self.left_img_features[feature_name][vector_type_key]
            distances = np.zeros((self.data_dict[feature_name][vector_type_key].shape[0]))
            
            # calculate distances
            if feature_name == "dino":
                if self.dino_distance_measure == "cosine":
                    distances = cosine_distances(current_vector.reshape(1, -1), self.data_dict[feature_name][vector_type_key]).squeeze()
                if self.dino_distance_measure == "euclidian":
                    a_min_b = current_vector.reshape(-1, 1) - self.data_dict[feature_name][vector_type_key].T
                    distances = np.sqrt(np.einsum('ij,ij->j', a_min_b, a_min_b))
            elif feature_name == "texture":
                if self.texture_distance_measure == "cosine":
                    distances = cosine_distances(current_vector.reshape(1, -1), self.data_dict[feature_name][vector_type_key]).squeeze()
                if self.texture_distance_measure == "euclidian":
                    a_min_b = current_vector.reshape(-1, 1) - self.data_dict[feature_name][vector_type_key].T
                    distances = np.sqrt(np.einsum('ij,ij->j', a_min_b, a_min_b))
            else:
                a_min_b = current_vector.reshape(-1, 1) - self.data_dict[feature_name][vector_type_key].T
                distances = np.sqrt(np.einsum('ij,ij->j', a_min_b, a_min_b))
            
            # rescale distances so that the distances are always within the range of 0-1
            # this way we can combine them, the element with distance 0 is the image itself if it's 
            # in dataset, otherwise it's the nearest neighbour
            distances = minmax_scale(distances, axis=0, feature_range=(0, 1), copy=False)

            distances_dict[feature_name] = distances

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


    def change_scatterplot_pointtype(self):
        """Use radio toggle to draw dots or images, triggered on toggle of the radio button.
        """
        # TODO: REIMPLEMENT
        print('change_scatterplot_pointtype is called')
        if self.ui.r_image_points.isChecked():
            #TODO: give user reset option/button. initially its FALSE
            self.scatterplot.dots_plot=False
            self.scatterplot.draw_scatterplot(reset=False)
            self.scatterplot.selected_idx.emit(self.scatterplot.selected_index)
        else:
            self.scatterplot.dots_plot=True
            self.scatterplot.draw_scatterplot_dots(reset=False)
            self.scatterplot.selected_idx.emit(self.scatterplot.selected_index)

    def on_canvas_click(self, ev):
        # QGraphicsScene.mousePressEvent(self.scatterplot.plot_widget.scene(), ev)
        # super().mousePressEvent(ev)

        self.scatterplot.clear_selection()
        pos = ev.scenePos()
        print("on canvas click:", pos)
        if ev.button() == Qt.MouseButton.LeftButton:
            # print("self.scatterplot.image_items", self.scatterplot.image_items)
            for idx, index, item in self.scatterplot.image_items:
                # print("item.mapFromScene(pos)", item, item.mapFromScene(pos))
                if item.contains(item.mapFromScene(pos)):
                    self.scatterplot.selected_point = int(pos.x()), int(pos.y())
                    self.scatterplot.selected_index = index
                    self.scatterplot.selected_idx.emit(index)
                    self.scatterplot.plot_index = idx
                    # TODO: rmv after all check, partial select ect
                    print('selected_index==plot_index?',index==idx)
                    self.clicked_on_point()
                    break

    # TODO: maybe change loc of this fn, or split its a little in between scatterplot and main
    def clicked_on_point(self):
        print("point/ image clicked, load on the left")
        self.left_img_filename = self.image_paths[self.scatterplot.selected_index]
        self.left_img_key = self.image_keys[self.scatterplot.selected_index]
        # set features for left 
        self.left_img_features = self.get_features_from_dataset(self.left_img_key)
        # display the image
        self.display_photo_left(self.left_img_filename)
        self.update_leftimg_data(self.left_img_key)

        topk_dict = self.calculate_nearest_neighbours()
        self.display_nearest_neighbours(topk_dict)


    def upload_image_left(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        file_dialog.setNameFilter("Images (*.png *.xpm *.jpg *.jpeg)")

        if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
            filenames = file_dialog.selectedFiles()
            if len(filenames) > 0:
                current_filepath = self.image_paths[self.key_to_idx[img_hash]]
                
                img_hash = da.get_image_hash(current_filepath)
                print(f"hash: {img_hash}")
                self.left_img_key = img_hash
                self.left_img_filename = current_filepath

                self.update_leftimg_data(self.left_img_key)
                # display the photo on the left
                self.display_photo_left(current_filepath)

                topk_dict = self.calculate_nearest_neighbours()
                self.display_nearest_neighbours(topk_dict)


    def update_leftimg_data(self, img_hash):
        # update image metdata if available
        if img_hash in self.metadata:
            filepath = self.image_paths[self.key_to_idx[img_hash]]
            print(f"found metadata for image: {filepath}")
            self.update_image_info(
                self.metadata[img_hash]['date'],
                self.metadata[img_hash]['artist_name'].decode('UTF-8'),
                self.metadata[img_hash]['style'].decode('UTF-8'),
                self.metadata[img_hash]['tags'].decode('UTF-8'),
            )
            self.left_img_features = self.get_features_from_dataset(img_hash)
        else:
            # get feature_vectors for new image 
            self.left_img_features = self.get_point_new_img(filepath)

            print(f"no metadata available for image: {filepath}")
            self.update_image_info("unknown date", "unknown artist", "unknown style", "unknown tags")



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
        print('displaying photo on right:', filename)
        pixmap = QPixmap(filename)
        self.ui.box_right_img.setPixmap(pixmap.scaled(w,h,Qt.AspectRatioMode.KeepAspectRatio))
        self.ui.box_right_img.setAlignment(Qt.AlignmentFlag.AlignCenter)

    #TODO: rewrite
    def display_preview_photos(self, filenames):
        for i, filename in enumerate(filenames):
            if i < len(self.preview_photo_labels):
                label = self.preview_photo_labels[i]
                pixmap = QPixmap(filename)
                # Scale the pixmap to fit the width while maintaining aspect ratio
                scaled_pixmap = pixmap.scaled(150, 150, Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation)
            
                # Create a square pixmap with a white background
                square_pixmap = QPixmap(150, 150)
                square_pixmap.fill(QColor(Qt.GlobalColor.white))
                
                # Calculate the position to center the scaled pixmap
                x = (square_pixmap.width() - scaled_pixmap.width()) // 2
                y = (square_pixmap.height() - scaled_pixmap.height()) // 2
            
                # Draw the scaled pixmap onto the square pixmap
                painter = QPainter(square_pixmap)
                painter.drawPixmap(x, y, scaled_pixmap)
                painter.end()
                
                # Set the square pixmap as the label's pixmap
                label.setPixmap(square_pixmap)

def start_dashboard(key_dict, dataset_filepath, images_filepath):
    app = QApplication(sys.argv)
    basepath = Path(__file__)
    css_filepath = str(basepath.parent / "theme1.css")
    with open(css_filepath, "r") as file:
        app.setStyleSheet(file.read())

    widget = MainWindow(key_dict, dataset_filepath, images_filepath)
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
