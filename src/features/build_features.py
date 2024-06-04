'''
Created on: Apr. 9 2024

@author: OpelliusAI
@summary: preprocessing functions, adapts preprocessing to the architecture and label dictionary when provided
@info: Step 1 of the model execution 
@info: How to add a new model : 
> create the build_model function for the new architecture (ex build_model_efficientnetb0)
> update the main build_model function with a block that checks a pattern in the architecture name and forwards the hyperparameters to the function dedicated to the new architecture. ex: if 'efficientnetb0' in archi: build_model_efficientnetb0(hp, ml_hp)
> update the build_feature function and add a condition in preprocess_input_archi(img,archi) function. (and import the specific preprocess_input function from the model's modules)
> Add your line in the src/run_config file that calls the specific architecture and define the size
'''

#####
# Imports 
#####
# preprocessing inputs for each model architecture
from tensorflow.keras.applications.efficientnet import preprocess_input as pp_effnet
from tensorflow.keras.applications.vgg16 import preprocess_input as pp_vgg16 
from tensorflow.keras.applications.vgg19 import preprocess_input as pp_vgg19
from tensorflow.keras.applications.resnet50 import preprocess_input as pp_resnet50
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# useful modules
import numpy as np
import os

### Configuration file import

from src.config.run_config import paths,infolog
from src.config.log_config import logger

### Functions


def preprocess_data(image_paths, labels, size, dim, archi):
    """
    Function to preprocess image data for neural network models.

    :param image_paths: List of file paths to the images.
    :param labels: List of labels corresponding to each image.
    :param size: Integer, the target size to which each image is resized.
    :param dim: Integer, the number of color dimensions required (1 for grayscale, 3 for RGB).
    :param archi: String, the architecture name which determines specific preprocessing needs.

    :return: A list of tuples, each containing an array representation of an image and its corresponding label.

    :raises ValueError: If 'dim' is not 1 or 3.
    :raises FileNotFoundError: If any image path in 'image_paths' does not point to an existing file.
    :raises Exception: For any other issues during the loading or processing of images, such as incompatible file types or errors in preprocessing functions.

    """
    logger.debug("---------------preprocess_data------------")
    if dim not in [1, 3]:
        raise ValueError("Dimension 'dim' must be either 1 (grayscale) or 3 (RGB).")
    data = []
    for img_path, label in zip(image_paths, labels):
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image path {img_path} does not exist.")
        try:
            img = load_img(img_path, target_size=(size, size), color_mode='grayscale' if dim == 1 else 'rgb')
            img_array = img_to_array(img)
            img_array = preprocess_input_archi(img_array, archi)
            data.append((img_array, label))
        except Exception as e:
            raise Exception(f"Error processing image at {img_path}: {str(e)}")

    return data


# Preprocessing redirection to the model's function
def preprocess_input_archi(img, archi):
    """
    Function that Preprocesses the input image according to the architecture's requirements.
    
    :param img: The input image.
    :param archi: The architecture name (e.g., 'lenet', 'vgg16', 'vgg19', 'resnet50', 'efficientnetb0').
    
    :return: The preprocessed image.

    :raises ValueError: If an unrecognized architecture name is provided.
    """
    logger.debug("---------------preprocess_input_archi------------")
    archi = archi.lower()

    try:
        if archi == 'lenet':  # Min-max normalization
            img = img / 255.0
        elif archi == 'vgg16':
            logger.debug(f"Architecture {archi}")
            # Normalization by subtracting the mean of each image channel (RGB) and reversing the channel order from RGB to BGR
            img = pp_vgg16(img)
        elif archi == 'vgg19':
            logger.debug(f"Architecture {archi}")
            img = pp_vgg19(img)
        elif archi == 'resnet50':
            logger.debug(f"Architecture {archi}")
            img = pp_resnet50(img)
        elif archi == 'efficientnetb0':
            logger.debug(f"Architecture {archi}")
            img = pp_effnet(img)
        else:
            raise ValueError(f"Unrecognized architecture name: {archi}")

        return img
    except ValueError as e:
        logger.error(f"ValueError: {e}")
        raise
    
def get_data_from_parent_directory_labelled(parent_directory, size, dim, archi, labelling=None):
    """
    Loads and preprocesses images from a parent directory, where each subdirectory represents a class.
    
    :param parent_directory: The parent directory containing subdirectories for each class.
    :param labelling: Optional dictionary mapping from directory names to numeric labels. If None, automatic labeling based on subdirectory names is applied.
    :param size: The target size of the images after resizing.
    :param dim: The dimensionality of the images (1 for grayscale, otherwise color).
    :param archi: The architecture name to use for specific preprocessing.
    :return: Tuple of Numpy arrays: (X_train, y_train, X_test, y_test, directory_to_label)
    
    :raises FileNotFoundError: If the parent directory does not exist.
    :raises ValueError: If `labelling` does not match the subdirectories in `parent_directory`.
    """
    logger.debug("---------------get_data_from_parent_directory_labelled------------")
    
    if not os.path.exists(parent_directory):
        raise FileNotFoundError(f"Parent directory '{parent_directory}' not found.")
    
    try:
        directories = [os.path.join(parent_directory, d) for d in os.listdir(parent_directory) if os.path.isdir(os.path.join(parent_directory, d))]
        if not directories:
            raise ValueError("No subdirectories found in the parent directory.")
        
        image_paths, labels = [], []
        directory_to_label = labelling or get_auto_labelling_from_directory(parent_directory)
        
        for dir_path in directories:
            directory_name = os.path.basename(dir_path)
            label = directory_to_label.get(directory_name)
            if label is None:
                raise ValueError(f"No label found for directory '{directory_name}'. Check your labelling dictionary.")
            
            for img_file in os.listdir(dir_path):
                if os.path.isfile(os.path.join(dir_path, img_file)):
                    image_paths.append(os.path.join(dir_path, img_file))
                    labels.append(label)

        image_paths = np.array(image_paths)
        labels = np.array(labels)

        # Splitting into training and testing sets
        train_image_paths, test_image_paths, train_labels, test_labels = train_test_split(image_paths, labels, test_size=0.2, random_state=1234)

        # Preprocessing training data
        train_data = preprocess_data(train_image_paths, train_labels, size, dim, archi)
        X_train, y_train = zip(*train_data)
        X_train, y_train = np.array(X_train), np.array(y_train)

        # Preprocessing testing data
        test_data = preprocess_data(test_image_paths, test_labels, size, dim, archi)
        X_test, y_test = zip(*test_data)
        X_test, y_test = np.array(X_test), np.array(y_test)

        return X_train, y_train, X_test, y_test, directory_to_label
    except Exception as e:
        logger.error(f"Error processing images: {e}")
        raise
 
 

def get_auto_labelling_from_directory(parent_directory):
    """
    Automatically generates labels for subdirectories within a given parent directory.
    Each subdirectory represents a class and is assigned a unique label.

    :param parent_directory: String, the path to the parent directory containing class subdirectories.
    :return: Dictionary mapping each subdirectory name to a unique integer label.

    :raises FileNotFoundError: If the parent directory does not exist.
    :raises ValueError: If no subdirectories are found within the parent directory.
    """
    logger.debug("---------------get_auto_labelling_from_directory------------")

    if not os.path.exists(parent_directory):
        raise FileNotFoundError(f"Parent directory '{parent_directory}' not found.")

    directories = [os.path.join(parent_directory, d) for d in os.listdir(parent_directory) if os.path.isdir(os.path.join(parent_directory, d))]
    if not directories:
        raise ValueError("No subdirectories found in the parent directory.")

    directory_to_label = {}
    for i, directory in enumerate(directories):
        directory_name = os.path.basename(directory)
        directory_to_label[directory_name] = i
        logger.debug(f"Label assigned to {directory_name}: {i}")

    return directory_to_label


def invert_dict(d):
    """
    Inverts a dictionary, swapping its keys and values.
    This function is useful after a numeric based prediction to display the associated string class
    :param d: Dictionary, the dictionary to invert.
    :return: Dictionary where the keys are the original dictionary's values and the values are the original keys.

    :raises ValueError: If the original dictionary contains duplicate values, preventing them from being used as keys in the new dictionary.
    """
    
    logger.debug("---------------invert_dict------------")

    try:
        # Attempt to invert the dictionary. This will fail if there are duplicate values in the original dictionary.
        inverted_dict = {v: k for k, v in d.items()}
        return inverted_dict
    except TypeError as e:
        logger.error("TypeError encountered during dictionary inversion: non-hashable types present in values.")
        raise TypeError("Dictionary values must be hashable to be used as dictionary keys.") from e
    except Exception as e:
        logger.error(f"An error occurred during dictionary inversion: {str(e)}")
        raise
