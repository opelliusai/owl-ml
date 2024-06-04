'''
Created on Apr. 9 2024

@author: OpelliusAI
@summary: Useful functions for GradCam heatmap construction

'''
import matplotlib as mpl
from keras.utils import load_img,img_to_array,array_to_img
import numpy as np
from keras.models import Model
import tensorflow as tf
from src.config.log_config import logger

def get_img_array(img_path, size):
    """
    Loads an image from a specified file path and converts it into a numpy array suitable for model prediction.

    :param img_path (str): The file path to the image.
    :param size (int): The target size to which the image should be resized, represented as the length of one side 
      (since the target is a square image).

    :return: numpy.ndarray: A 4D numpy array of shape (1, size, size, 3) representing the image, suitable for use as input to a CNN model. The image data is of type float32.

    :raises FileNotFoundError: If the img_path does not exist.
    :raises Exception: If an unknown error occurs
    """
    logger.debug("-----------------get_img_array---------------")
    try:
        # `img` is a PIL image of size size x size
        img = load_img(img_path, target_size=(size, size))
    except FileNotFoundError as fnfe:
        raise FileNotFoundError(f"The file specified at {img_path} was not found.") from fnfe
    except Exception as e:
        raise ValueError(f"The file at {img_path} could not be opened as an image.") from e

    # `array` is a float32 Numpy array of shape (size, size, 3)
    array = img_to_array(img)

    # We add a dimension to transform our array into a "batch"
    # of size (1, size, size, 3)
    array = np.expand_dims(array, axis=0)

    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generates a Grad-CAM heatmap for a given model and image.

    :param img_array: The input image tensor.
    :param model: The pre-trained model.
    :param last_conv_layer_name: Name of the last convolutional layer in the model.
    :param pred_index: Index of the predicted class for which the heatmap is computed. If None, uses the class with the highest prediction score.
    :return: A heatmap array showing the areas of the image most important for the model's prediction.

    :raises ValueError: If `last_conv_layer_name` does not correspond to a layer in the model.
    """

    logger.debug(f"model output {model.output}")

    # Temporarily remove the activation function from the last layer to make predictions linear
    model.layers[-1].activation = None
    logger.debug(f"model output after -1 {model.output}")

    try:
        # Create a model that maps the input image to the activations of the last conv layer as well as the output predictions
        
        grad_model = Model(
            model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
            )
        
        '''
        #deactivated_last_conv_layer=um.get_last_conv_layer(model)
        #logger.debug(f"deactivated layer last conv layer {deactivated_last_conv_layer} vs {last_conv_layer_name}")
        grad_model = Model(
        inputs=model.inputs, 
        outputs=[model.get_layer(last_conv_layer_name).output] + model.output
        )
        '''
    
    except ValueError as e:
        raise ValueError(f"Layer {last_conv_layer_name} not found in the model.") from e
    
    logger.debug(f"make_gradcam_heatmap last conv layer name")

    # Use GradientTape to compute the gradients with respect to the predicted class
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Average the gradients spatially
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the output feature map with these averaged gradients
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize the heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def make_gradcam_heatmap_v215(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Tensorflow 2.15 version
    Generates a Grad-CAM heatmap for a given model and image.

    :param img_array: The input image tensor.
    :param model: The pre-trained model.
    :param last_conv_layer_name: Name of the last convolutional layer in the model.
    :param pred_index: Index of the predicted class for which the heatmap is computed. If None, uses the class with the highest prediction score.
    :return: A heatmap array showing the areas of the image most important for the model's prediction.

    :raises ValueError: If `last_conv_layer_name` does not correspond to a layer in the model.
    """
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    logger.debug(f"model output {model.output}")
    model.layers[-1].activation = None
    logger.debug(f"mode output after -1 {model.output}")
    '''
    grad_model = Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )
    '''
    grad_model = Model(
    inputs=model.inputs, 
    outputs=[model.get_layer(last_conv_layer_name).output] + model.output
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

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    """
    Saves and displays a visualization of the Grad-CAM heatmap superimposed on the original image.

    :param img_path: Path to the original image.
    :param heatmap: Heatmap generated for the image, indicating regions of interest.
    :param cam_path: Path where the CAM image will be saved. Defaults to "cam.jpg".
    :param alpha: Intensity factor for superimposing the heatmap onto the original image. Defaults to 0.4.

    :return: None. The function saves the superimposed image to the specified path and displays it.

    :raises FileNotFoundError: If the image at img_path does not exist.
    :raises Exception: If there are issues during the processing or saving of the image.
    """
    try:
        # Load the original image
        img = load_img(img_path)
        img = img_to_array(img)

        # Ensure heatmap is in a compatible format
        heatmap = np.uint8(255 * heatmap)

        # Use jet colormap to colorize heatmap
        jet = mpl.colormaps["jet"]
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        # Create an image with RGB colorized heatmap
        jet_heatmap = array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = img_to_array(jet_heatmap)

        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap * alpha + img
        superimposed_img = array_to_img(superimposed_img)

        # Save the superimposed image
        superimposed_img.save(cam_path)
        
        # Optional: Display the image (uncomment if needed)
        # display(Image.open(cam_path))

    except FileNotFoundError:
        raise FileNotFoundError(f"The image at {img_path} was not found.")
    except Exception as e:
        raise Exception(f"An error occurred while processing the image: {e}")

