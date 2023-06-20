# This Python file uses the following encoding: utf-8
import sys

# normal imports
from pathlib import Path
import numpy as np
from PIL import Image
import configparser
from scipy import spatial
from sklearn.preprocessing import minmax_scale

# qt imports
from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QMainWindow, QFileDialog, QApplication, QVBoxLayout
from PyQt6.QtGui import QPixmap, QPainter, QColor
from PyQt6.QtCore import QRect, Qt

# custom widgets
from .custom_widgets import ButtonWidget, ScatterplotWidget, ImageWidget, ModelVis, TimelineView, TimelineWindow, HistoryTimelineWidget

# deepsim analyzer package
import deepsim_analyzer as da

# from PyQt6.QtWidgets import (
#    QApplication,
#    QMainWindow,
#    QLabel,
#    QPushButton,
#    QVBoxLayout,
#    QHBoxLayout,
#    QWidget,
#    QFileDialog,
#    QGridLayout,
#    QLineEdit,
#    QMessageBox,
#    QFrame
#)


# Important:
# You need to run the following command to generate the ui_form.py file
#     pyside6-uic form.ui -o ui_form.py, or
#     pyside2-uic form.ui -o ui_form.py
from .ui_form import Ui_MainWindow

class MainWindow(QMainWindow):
    def __init__(self, key_dict, datafile_path, images_filepath):
        super().__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.b_upload.clicked.connect(self.upload_image_left)

        # load the config file 'config.ini'
        self.config = configparser.ConfigParser()
        basepath = Path(__file__).parent
        config_filepath = str(basepath / "config.ini")
        self.config.read(config_filepath)
        self.datafile_path = datafile_path

        # save passed arguments as attributes
        self.image_key_dict = key_dict
        self.image_keys = [key for key in key_dict.keys()] # hashes
        self.key_to_idx = {key:i for i, key in enumerate(key_dict.keys())} # hashes
        self.image_indices = [i for i in range(len(key_dict))]
        self.image_paths = [str(Path(images_filepath) / image_name) for image_name in key_dict.values()]

        # in time we have to get all features for all the data, we will start with
        # just the dummy feature
        self.available_features = ["dummy", "dino"]

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
        self.metadata = {}
        # read metadata for all image keys that were passed
        for key in key_dict.keys():
            metadata_key = da.read_metadata(datafile_path, key)
            self.metadata[key] = metadata_key

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

        for i, key in enumerate(self.image_keys):
            feature_dict_key = self.get_features_from_dataset(key)
            for feature_name, value in feature_dict_key.items():
                self.data_dict[feature_name]["projection"][i] = value['projection']
                self.data_dict[feature_name]["full"][i] = value['full']

        # ================ SETUP LEFT COLUMN ================

        # ---------------- STARTING IMG ----------------
        # load an initial first image to display
        default_image_key = list(key_dict.keys())[0]
        default_image_path = key_dict[default_image_key]
        default_image_path_absolute = str(Path(images_filepath) / default_image_path)
        
        self.left_img_key = default_image_key

        # display the base image
        self.display_photo_left(default_image_path_absolute)
        # data for left img feature, can come  from dataset or be calculated on the fly
        self.update_leftimg_data(self.left_img_key)
        # add additional data in box_left_low

        # ================ SETUP MIDDLE COLUMN ================

        # setup scatterplot
        # TODO: setup feature projection plot with combined and individual plots!
        self.scatterplot = ScatterplotWidget(
            self.data_dict['dino']["projection"], self.image_indices, self.image_paths, self.config, self.ui.scatterplot_frame
        )
        self.scatterplot.plot_widget.scene().sigMouseClicked.connect(
            self.on_canvas_click
        )
        self.ui.r_image_points.toggled.connect(self.change_scatterplot_pointtype)
        # toggle the the dots to images radio button
        self.ui.r_image_points.toggle()

        # ================ SETUP RIGHT COLUMN ================
        topk_dict = self.calculate_nearest_neighbours()
        self.display_nearest_neighbours(topk_dict)
        


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
            

    def calculate_nearest_neighbours(self, topk=5, combined=False, feature_weight_dict=None, use_projection=True):
        # get features for current image
        topk_results = {}
        distances_dict = {}
        
        # MANUAL OVERWRITE OF METRIC_WEIGHT_DICT
        print("performing manual overwrite of metric reweighting")
        feature_weight_dict = {
            "dummy": 0,
            "dino": 1,
        }

        if use_projection:
            print("using projection vectors to calculate distances instead of full vectors")
            vector_type_key = 'projection'
        else:
            vector_type_key = 'full'

        if combined:
            raise NotImplementedError("no combined features available yet")
        
        indices = np.arange(len(self.image_keys))
        for feature_name in self.available_features:

            current_vector = self.left_img_features[feature_name][vector_type_key]
            distances = np.zeros((self.data_dict[feature_name][vector_type_key].shape[0]))

            for i in range(self.data_dict[feature_name][vector_type_key].shape[0]):
                target_vector = self.data_dict[feature_name][vector_type_key][i]
                distances[i] = spatial.distance.cosine(current_vector, target_vector)
            
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
            # TODO: we would do reweighting of the different metrics here
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
        if self.ui.r_image_points.isChecked():
            self.scatterplot.draw_scatterplot()
        else:
            self.scatterplot.draw_scatterplot_dots()

    def on_canvas_click(self, ev):
        pos = ev.scenePos()
        print("on canvas click:", pos)
        if ev.button() == Qt.MouseButton.LeftButton:
            # print("self.scatterplot.image_items", self.scatterplot.image_items)
            for idx, index, item in self.scatterplot.image_items:
                # print("item.mapFromScene(pos)", item, item.mapFromScene(pos))
                if item.contains(item.mapFromScene(pos)):
                    self.scatterplot.selected_point = int(pos.x()), int(pos.y())
                    self.scatterplot.selected_index = index
                    self.scatterplot.plot_index = idx
                    print('selected_index==plot_index?',index==idx)
                    self.clicked_on_point()
                    break

    def clicked_on_point(self):
        print("point/ image clicked, load on the left")
        self.left_img_filename = self.image_paths[self.scatterplot.selected_index]
        self.left_img_key = self.image_keys[self.scatterplot.plot_index]
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
                
                img_hash = da.get_image_hash(current_filepath, is_filepath=True)
                print(f"hash: {img_hash}")
                self.left_img_key = img_hash

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
                self.metadata[img_hash]['artist_name'],
                self.metadata[img_hash]['style'],
                self.metadata[img_hash]['tags'],
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


    def initialize_images(self, init_point, filepath, init_key, upload=False, left=False):
        self.display_photo_left(filepath)
        nearest_indices = self.scatterplot.find_nearest_neighbors(init_point, n=3)
        print("nearest_indices", nearest_indices)

        nearest_images = []
        for near_idx in nearest_indices:
            nearest_images.append(self.image_paths[near_idx])
        self.display_preview_photos(nearest_images)

        if left:
            self.display_photo_left(filepath, init_key=init_key, upload=upload)

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