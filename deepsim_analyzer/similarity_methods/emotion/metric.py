import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

# Display

import matplotlib.cm as cm
from deepface import DeepFace
from deepface.basemodels import VGGFace

def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.utils.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.utils.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(img_path, heatmap, cam_path='Testdata/', alpha=0.2):
    # Load the original image
    img = keras.utils.load_img(img_path)
    img = keras.utils.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    # Save the superimposed image
    # cam_path += name + '.jpg'
    # superimposed_img.save(cam_path)

    # Display Grad CAM
    # display(Image(cam_path))
    return superimposed_img



def calc_and_save_features(images, datafile_path, save_feature_maps=True):
    from deepsim_analyzer.io import get_image_hash, load_image, save_feature
    # force to use cpu as something does not work with gpu

    # Make model
    model = VGGFace.loadModel()
    for image_path in tqdm(
        images, desc=f"calculating emotion features", total=len(images)
    ):
        image_path = str(image_path)  # in case image_path is a pathlib path
        img_hash = get_image_hash(image_path)
        feature_vector = np.array(DeepFace.represent(img_path = image_path, enforce_detection=False)[0]['embedding'])
        save_feature(datafile_path, img_hash, feature_vector, 'emotion')
        # preds = np.array(list(preds[0].values()))
        # print(preds)
        # print("Predicted:", decode_predictions(preds, top=1)[0])
        target_size = (224, 224)
        img_array = get_img_array(image_path, target_size)

        # Generate class activation heatmap
        heatmap = make_gradcam_heatmap(img_array, model, model.layers[-8].name)

        if save_feature_maps:
            save_feature(datafile_path, img_hash, heatmap, "emotion_fm")

        save_feature(datafile_path, img_hash, feature_vector, 'emotion')

    del DeepFace.model_obj
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"