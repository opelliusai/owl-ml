'''
Created on Apr. 9 2024

@author: OpelliusAI
@summary: Useful functions for AI models

'''

### External IMPORTS
import os
# Data load and save
from tensorflow.keras.models import load_model
import pickle
import json
# Usual functions for calculation, dataframes and plots
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shutil

# metrics
from tensorflow.keras.losses import CategoricalCrossentropy,SparseCategoricalCrossentropy,BinaryCrossentropy
from tensorflow.keras.layers import Conv2D
from keras.models import Model
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report, confusion_matrix
import keras

### Internal imports
from src.features import build_features as bf
from src.utils import utils_gradcam as ug
### Configuration file import
'''
src.config file contains information of the repository
paths[]: Folders and subfolders to find or generate information ex : paths['main'] is the project path
infolog[]: Logging information : Folder, logname ex : utils_models.log
'''
from src.config.log_config import logger



### FUNCTIONS
## Keras Models save and load
def save_model(model, save_path):
    """
    Saves a Keras model to the specified path.
    
    :param model: The model to save.
    :param save_path: The file FULL path where to save the model.
    :raises PermissionError: If there has been a write permision issue
    :raises IOError: if there has been an I/O error
    :raises Exception: If an unknown error occurs
    """
    logger.debug("--------------------save_model--------------------")
    try: # Saving the model
        # Extract the directory from the complete file path
        directory = os.path.dirname(save_path)
        
        # Check if the directory already exists
        if not os.path.exists(directory):
            # Create the directory if it does not exist
            os.makedirs(directory)
        
        # Save the model to the specified path
        model.save(save_path)
        
    except PermissionError:
        # Exception if there is no permission to write in the directory
        logger.error("Error: No permission to write to the specified directory.")
        raise
    except IOError as e:
        # Handle general I/O errors
        logger.error(f"An I/O error occurred: {e}")
        raise
    except Exception as e:
        # Handle other possible exceptions
        logger.error(f"An unexpected error occurred: {e}")
        raise

    return save_path

## Models loading. Names load_models to distinguish the function from the keras 'load_model' function
def load_models(load_path):
    """
    Loads a Keras model from a file.
    uses tensorflow.keras.models.load_model
    :param load_path: The FULL file path containing the model to load.
    :return: The loaded model.
    :raises IOError: if the file to load is not accessible
    :raises ValueError: If there has been an issue with the model interpretation (not a model or corrupted file)
    :raises Exception: If an unknown error occurs
    """
    logger.debug("--------------------load_models--------------------")
    try:
        # Load the model from the specified file path
        model = load_model(load_path)
        return model
    except IOError as e:
        # Handle errors related to file access issues
        logger.error(f"Error: Could not access file to load model. {e}")
        raise
    except ValueError as e:
        # Handle errors related to the model file being invalid or corrupted
        logger.error(f"Error: The file might not be a Keras model file or it is corrupted. {e}")
        raise
    except Exception as e:
        # Handle other possible exceptions
        logger.error(f"An unexpected error occurred while loading the model: {e}")
        raise

## Models training history save and load (pickle or json format)
def save_history(history, save_path):
    """
    Saves a training history to a file, supporting both Pickle, JSON and CSV formats.
    
    uses pickle.dump or json.dump depending on the specified file path extension
    :param history: The training history to save.
    :param save_path: The FULL file path where the training history should be saved.
    
    :raises If there has been an issue with the model interpretation (not a model or corrupted file)
    :raises PermissionError: If the write permission is not granted to the specified path
    :raises IOError: if there is an I/O issue
    :raises Exception: If an unknown error occurs
    """
    logger.debug("--------------------save_history--------------------")
    
    try:
        # Ensure the directory exists
        logger.debug(f"Saving file in {save_path}")
        directory = os.path.dirname(save_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Determine the file format from the extension and save accordingly
        _, ext = os.path.splitext(save_path)
        if ext == '.pkl':
            with open(save_path, 'wb') as f:
                pickle.dump(history, f)
        elif ext == '.json':
            with open(save_path, 'w') as f:
                json.dump(history, f)
        elif ext =='.csv':
            # We need to ensure that the history object is a dataframe, if not, saving might be corrupted and therefore the save request is rejected
            if isinstance(history, pd.DataFrame):
                history.to_csv(save_path, index=False)
            else:
                raise ValueError("Unsupported object format. Object must be a dataframe to be stored in a csv format")
        else:
            raise ValueError("Unsupported file format. Please use .pkl or .json.")
    except PermissionError:
        logger.error(f"Error: No permission to write to {save_path}.")
        raise
    except IOError as e:
        logger.error(f"An I/O error occurred while saving to {save_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise
    return save_path

## Models training history load (pickle, json or csv format)
def load_history(load_path):
    """
    Loads a training history from a file, supporting both pickle and JSON formats.
    uses pickle.load or json.load depending on the specified file path extension
    :param load_path: The file path of the training history to load.
    :return: The loaded training history.
    :raises ValueError: If the file format is not suppoeted
    :raises IOError: if there is an I/O issue
    :raises pickle.PickleError: if the pickle file might be corrupted
    :raises json.JSONDecodeError: if the JSON file might be corrupted
    :raises pd.errors.ParserError: if the CSV file to be stored in a dataframe might be corrupted
    :raises Exception: If an unknown error occurs
    """
    logger.debug("--------------------load_history--------------------")
    try:
        # Determine the file format from the extension
        _, file_extension = os.path.splitext(load_path)
        with open(load_path, 'rb' if file_extension.lower() == '.pkl' else 'r') as f:
            if file_extension.lower() == '.pkl':
                history = pickle.load(f)
            elif file_extension.lower() == '.json':
                history = json.load(f)
            elif file_extension.lower() == '.csv':
                history = pd.read_csv(f)
            else:
                raise ValueError("Unsupported file format. Please use .pkl, .json or .csv .")
        return history
    except IOError as e:
        logger.error(f"Error: Could not access file at {load_path}. {e}")
        raise
    except (pickle.PickleError, json.JSONDecodeError, pd.errors.ParserError) as e:
        logger.error(f"Error: The file might be corrupted or in an incorrect format. {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise

## Label based Predictions for one image
def get_one_prediction_val_label(model, image_path, label_to_directory, archi, size=224):
    """
    Generates a prediction for a single image using a trained model, and maps the prediction
    to a specific label and directory.

    uses src.features.build_features preprocess_image function to preprocess the image
    :param model: The trained model to use for prediction.
    :param image_path: Path to the image to predict on.
    :param label_to_directory: Mapping from predicted labels to directories. ex : {0:COVID,1:Lung_Opacity,2:Normal,3:Viral_Pneumonia}
    :param archi: The preprocessing model architecture to use the specific preprocess_input function. ex : VGG16, VGG19, EfficientNet etc.
    :param size: The size to which the image should be resized. Default is 224.
    :return: The directory corresponding to the prediction and the prediction probability 4 digit round
    
    :raises FileNotFoundError: If the file was not found
    :raises label_to_directory issue. If no directory correspondance was found
    :raises if the pickle file might be corrupted
    """
    logger.debug("--------------------get_one_prediction_val_label--------------------")
    try:
        # Preprocess the image according to the specified model and size
        img = bf.preprocess_image(image_path, size, archi)

        # Make a prediction using the model
        predictions = model.predict(img)
        logger.debug(f"Image Path: {image_path}")

        # Determine the number of classes based on the model output shape
        num_classes = model.outputs[0].shape[-1]

        if num_classes == 1:
            # Binary classification case
            result = int((predictions[0] > 0.5).astype(int))
            pred_directory = label_to_directory[result]
        else:
            # Multiclass classification case
            result = np.argmax(predictions)
            pred_directory = label_to_directory[result]

        logger.debug(f"Result: {result}")
        logger.debug(f"Predicted directory: {pred_directory}")

        # Calculate and print the prediction probability, rounded to 4 decimal places
        pred_probability = np.round(predictions[0], 4)
        logger.debug(f"Prediction probability: {pred_probability}")

        return pred_directory, pred_probability

    except FileNotFoundError:
        # Handle cases where the image file does not exist
        logger.error(f"Error: The file at {image_path} was not found.")
        raise
    except KeyError as e:
        # Handle cases where the prediction does not correspond to any directory
        logger.error(f"Error: label_to_directory issue. No directory corresponding to the prediction {result}. Error {e}")
        raise
    except Exception as e:
        # Handle other unforeseen errors
        logger.error(f"An unexpected error occurred: {e}")
        raise

## Saving Training visual reports (history) Pickle, JSON or CSV file
def generate_training_plots(history_file, output_filepath, run_info):
    """
    Generates and saves training and validation loss and accuracy plots from a history file.
    The history file can be in PKL, CSV, or JSON format.

    :param history_file: The FULL filepath of the training history file.
    :param output_folder: The output directory to save the plots.
    :param run_info: (currently run id) Information or identifier for the training run to be included in the plot titles.
    the file name format is "{run_info}_training_validation_plots.png"
    :raises Exception: If an error occurs
    """
    logger.debug("--------------------generate_training_plots--------------------")
    # Determine file extension and load history accordingly
    logger.debug(f"history_file {history_file}")
    logger.debug(f"output_folder {output_filepath}")
    logger.debug(f"run_id {run_info}")
    _, file_extension = os.path.splitext(history_file)
    
    # first step is to load the history file
    try:
        if file_extension.lower() == '.pkl':
            # Load history from PKL file
            logger.debug("Managing Pickle file")
            history = pd.read_pickle(history_file)
        elif file_extension.lower() == '.csv':
            # Load history from CSV file
            logger.debug("Managing CSV file")
            history = pd.read_csv(history_file)
        elif file_extension.lower() == '.json':
            # Load history from JSON file
            logger.debug("Managing JSON file")
            with open(history_file, 'r') as f:
                history = json.load(f)
        else:
            raise ValueError("Unsupported file format. Please use .pkl, .csv, or .json")

        # Create plot
        plt.figure(figsize=(12, 4))

        # Loss subplot
        plt.subplot(121)
        plt.plot(history['loss'], label='train')
        plt.plot(history['val_loss'], label='test')
        plt.title(f'Run {run_info} - Model loss by epoch')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(loc='upper right')

        # Accuracy subplot
        plt.subplot(122)
        plt.plot(history['accuracy'], label='train')
        plt.plot(history['val_accuracy'], label='test')
        plt.title(f'Run {run_info} - Model accuracy by epoch')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(loc='upper right')

        # Save the plot
        plt.savefig(output_filepath)
        plt.close()

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

## Save confusion matrix and classification report from dataframes
def save_dataframe_plot(df, output_filepath, plot_type, labels_dict=None):
    """
    Generates and saves a plot based on the specified type, appending the plot type as a suffix to the run ID for the filename.

    :param df: DataFrame to be plotted, can be a confusion matrix or a classification report.
    :param output_folder: The output directory to save the plot.
    :param run_id: The identifier for the run or experiment. Used as the base name for the output file.
    :param plot_type: Type of the plot to generate - 'confusion_matrix' or 'classification_report'. Also used as a suffix for the filename.
    :param labels_dict: Optional; needed if plot_type is 'confusion_matrix'. It maps class numbers to class names.
    """
    logger.debug("--------------------save_dataframe_plot--------------------")
    # Ensure plot_type is valid and prepare the plot accordingly
    if plot_type == 'confusion_matrix' and labels_dict is not None:
        # Generate labels for confusion matrix
        axis_labels = [labels_dict[i] for i in labels_dict]

        plt.figure(figsize=(8, 6))
        sns.heatmap(df, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=axis_labels, yticklabels=axis_labels)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Confusion Matrix')
        
    elif plot_type == 'classification_report':
        plt.figure(figsize=(10, len(df) * 0.5))
        sns.heatmap(data=df, annot=True, fmt=".2f", cmap="Blues", cbar=True)
        plt.title('Classification Report')
        plt.xlabel('Metrics')
        plt.ylabel('Classes')
    else:
        raise ValueError("Invalid plot type specified. Use 'confusion_matrix' or 'classification_report'.")

    
    # Save the plot
    plt.savefig(output_filepath)
    plt.close()

    logger.debug(f"Plot saved to {output_filepath}")


# Evaluation function for binary classification

def bin_get_prediction_metrics(model, X_eval, y_eval):
    """
    Calculates and prints various binary classification metrics including accuracy, recall, F1-score,
    classification report, and confusion matrix for the binary classification task.
    
    :param model: Trained model to evaluate.
    :param X_eval: Evaluation data features.
    :param y_eval: True labels for evaluation data.
    
    :return: A dictionary containing calculated metrics including accuracy, loss, recall, F1-score, 
             sensitivity, specificity, confusion matrix, and classification report.
             
    :raises ValueError: If prediction or metrics calculation fails.
    """
    metrics_dict = {}
    try:
        # Model prediction
        y_eval_pred = model.predict(X_eval)
        logger.debug('----------bin_get_prediction_metrics(model,X_eval,y_eval)-------')
        logger.debug(f"Y EVAL {y_eval}")
        logger.debug(f"Y EVAL PRED {y_eval_pred}")
        
        # Converting predictions to binary
        y_eval_pred_binary = (y_eval_pred > 0.5).astype(int)
        logger.debug(f"Y EVAL PRED Binary {y_eval_pred_binary}")
        
        # Loss calculation
        bce = BinaryCrossentropy()
        loss = bce(y_eval, y_eval_pred_binary).numpy()

        # Accuracy calculation
        accuracy_s = accuracy_score(y_eval, y_eval_pred_binary)
        logger.debug("Precision/accuracy :{accuracy_s}")

        # Recall calculation
        recall = recall_score(y_eval, y_eval_pred_binary, average="binary")
        logger.debug(f"Recall : {recall}")

        # F1-score calculation
        f1 = f1_score(y_eval, y_eval_pred_binary, average="binary")
        logger.debug(f"F1-Score : {f1}")

        # Classification report
        class_report = classification_report(y_eval, y_eval_pred_binary, output_dict=True)
        logger.debug(f"Classification Report :{class_report}")

        # Confusion matrix calculation
        conf_matrix = confusion_matrix(y_eval, y_eval_pred_binary)
        logger.debug(f"Confusion Matrix {conf_matrix}")

        # Sensitivity and Specificity calculations for each class
        sensitivities, specificities = [], []
        for i in range(len(conf_matrix)):
            tp = conf_matrix[i, i]
            fn = sum(conf_matrix[i, :]) - tp
            fp = sum(conf_matrix[:, i]) - tp
            tn = conf_matrix.sum() - (fp + fn + tp)
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            sensitivities.append(sensitivity)
            specificities.append(specificity)
            
            logger.debug(f"Class {i}: Sensitivity (Recall) = {sensitivity:.2f}, Specificity = {specificity:.2f}")

        # Metrics aggregation
        metrics_dict = {
            "Accuracy": accuracy_s,
            "Loss": loss,
            "Recall": recall,
            "F1-score": f1,
            "Confusion Matrix": conf_matrix,
            "Classification report": class_report
        }

        logger.debug(f" Metrics dict {metrics_dict}")
        return metrics_dict
    except Exception as e:
        raise ValueError(f"Error during model evaluation: {e}")


# multiple classification standard (not one hot encoded)
def multi_get_prediction_metrics(model, X_eval, y_eval):
    """
    Calculates and logs various metrics for multiclass classification including accuracy, recall, F1-score,
    classification report, and confusion matrix, specifically for models trained with
    sparse categorical cross-entropy loss.

    :param model: The trained model to evaluate.
    :param X_eval: Evaluation data features.
    :param y_eval: True labels for evaluation data as integers (not one-hot encoded).

    :return: A dictionary containing calculated metrics such as accuracy, loss, recall, F1-score, 
             confusion matrix, and classification report.

    :raises ValueError: If an error occurs during the prediction or metrics calculation process.
    """
    metrics_dict = {}
    try:
        logger.debug('----------multi_get_prediction_metrics(model,X_eval,y_eval)-------')

        # Model prediction
        y_eval_pred = model.predict(X_eval)
        logger.debug(f"Y EVAL {y_eval}")
        logger.debug(f"Y EVAL PRED {y_eval_pred}")

        # Converting predictions to class indices
        y_eval_pred = np.argmax(y_eval_pred, axis=1)
        logger.debug(f"Y EVAL PRED Classes {y_eval_pred}")

        # Loss calculation using SparseCategoricalCrossentropy
        sce = SparseCategoricalCrossentropy(from_logits=False)
        loss = sce(y_eval, model.predict(X_eval)).numpy()

        logger.debug("MULTICLASS CASE")
        logger.debug(f"NEW Y EVAL PRED {y_eval_pred}")

        # Metrics calculation
        accuracy_s = accuracy_score(y_eval, y_eval_pred)
        recall = recall_score(y_eval, y_eval_pred, average="macro")
        f1 = f1_score(y_eval, y_eval_pred, average="macro")
        class_report = classification_report(y_eval, y_eval_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_eval, y_eval_pred)

        # Logging calculated metrics
        logger.debug("Precision/accuracy :", accuracy_s)
        logger.debug("Loss (Sparse Categorical Cross-Entropy) :", loss)
        logger.debug("Recall :", recall)
        logger.debug("F1-Score :", f1)
        logger.debug(f"Classification Report :{class_report}")
        logger.debug(f"Confusion Matrix {conf_matrix}")

        # Metrics aggregation
        metrics_dict = {
            "Accuracy": accuracy_s,
            "Loss": loss,
            "Recall": recall,
            "F1-score": f1,
            "Confusion Matrix": conf_matrix,
            "Classification report": class_report
        }

        return metrics_dict
    except Exception as e:
        logger.error(f"Error during multi-class model evaluation: {e}")
        raise ValueError(f"Error during multi-class model evaluation: {e}")

#### GRADCAM
preprocess_input = keras.applications.xception.preprocess_input

def get_img_gradcam(img_path, model, output_folder, model_prefix):
    
    """
    Generates a Grad-CAM heatmap for a given image and model, and saves the result.

    :param img_path: Path to the input image.
    :type img_path: str
    :param model: The Keras model for generating the Grad-CAM.
    :type model: keras.models.Model
    :param output_folder: Folder where the Grad-CAM output image will be saved.
    :type output_folder: str
    :param model_prefix: Prefix indicating the model type, used in naming the output file.
    :type model_prefix: str
    :raises FileNotFoundError: If the input image path does not exist.
    :raises ValueError: If the `model_prefix` does not contain recognized model identifiers.
    :returns: The path to the saved Grad-CAM image.
    :rtype: str

    This function checks if the provided image path is valid and not already processed.
    It preprocesses the input image, identifies the last convolutional layer of the model,
    generates a Grad-CAM heatmap, and saves the heatmap overlay on the original image.
    The path to the saved image is returned. It supports models with specific prefixes
    indicating the model architecture (e.g., 'vgg' for VGG models, 'eff' for EfficientNet models).
    """
    # Check if the image path exists
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"The image path {img_path} does not exist.")
    
    # Initialize the results list and compute the CAM path
    file_name = os.path.basename(img_path)
    cam_path = os.path.join(output_folder, f"{model_prefix}_{file_name}")
    logger.debug(f"Output folder {output_folder}")
    # Process the image if it has not been processed before
    if "gradcam_" not in img_path and img_path != '.DS_Store' and not os.path.exists(cam_path):
        # Log the file name
        logger.debug(f"file_name {file_name}")
        
        # Get the image array
        img_array = preprocess_input(ug.get_img_array(img_path, size=224))
        logger.debug(f"img_array {img_array}")
        
        # Determine the last convolutional layer name based on the model prefix
        '''
        if "vgg" in model_prefix.lower():
            last_conv_layer_name = "block5_conv3"
        elif "eff" in model_prefix.lower():
            last_conv_layer_name = "top_conv"
        else
            raise ValueError(f"Model prefix {model_prefix} is not recognized.")
        '''
        # Override with dynamic detection of the last conv layer
        last_conv_layer_name = get_last_conv_layer(model)
        logger.debug(f"last_conv_layer_name {last_conv_layer_name}")
        
        # Generate the Grad-CAM heatmap
        heatmap = ug.make_gradcam_heatmap(img_array, model, last_conv_layer_name)
        logger.debug(f"heatmap {heatmap}")
        
        # Save and display the Grad-CAM image
        logger.debug(f"cam_path {cam_path}")
        ug.save_and_display_gradcam(img_path, heatmap, cam_path)
    
    # Return the path to the generated Grad-CAM image
    return cam_path


def get_last_conv_layer(model):
    """
    Returns the name of the last convolutional layer in a Keras model.

    :param model: The Keras model to inspect.
    :type model: keras.models.Model
    :raises TypeError: If the input is not a Keras model.
    :returns: The name of the last convolutional layer. Returns None if no convolutional layer is found.
    :rtype: str or None

    This function iterates through the model's layers in reverse order, checking each layer to determine
    if it is a convolutional layer by checking its instance type against Conv2D. It returns the name of the
    first convolutional layer it finds when iterating in reverse, which is the model's last convolutional layer.
    If the model does not contain any convolutional layers, the function returns None.
    """
    # Ensure the input is a Keras model instance
    if not isinstance(model, Model):
        raise TypeError("The provided model is not a valid Keras model.")
    
    # Iterate through the model's layers in reverse order
    for layer in reversed(model.layers):
        # Check if the layer is a convolutional layer
        if isinstance(layer, Conv2D):
            logger.debug(f"getting last cnv layer name {layer.name} output ")
            logger.debug(f"{model.get_layer(layer.name).output}")
            #logger.debug("Getting after deactivation")
            #model.layers[-1].activation = None
            #logger.debug(f"{model.get_layer(layer.name).output}")
            
            return layer.name
        
    # Return None if no convolutional layer was found
   
    return None

def remove_old_images(dir_path):
    """
    Removes all files and subdirectories from a specified directory except for those named with a 'gradcam_' prefix.

    :param dir_path: Path to the directory from which files and directories will be removed.

    :raises FileNotFoundError: If the specified directory does not exist.
    :raises PermissionError: If the function lacks permission to delete files or directories.
    :raises Exception: For other issues that prevent file or directory deletion.
    """
    logger.info(f"Starting to remove old images from {dir_path}")

    if not os.path.exists(dir_path):
        logger.error(f"Directory not found: {dir_path}")
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    try:
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            try:
                # Delete files that do not start with 'gradcam_', preserve those that do
                if os.path.isfile(file_path) and "gradcam_" not in filename:
                    os.unlink(file_path)
                    logger.debug(f"Deleted file: {file_path}")
                # Recursively delete directories
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    logger.debug(f"Deleted directory: {file_path}")
            except PermissionError as e:
                logger.error(f"Permission denied for deleting {file_path}. Reason: {e}")
                raise PermissionError(f"Failed to delete {file_path}. Reason: {e}") from e
            except Exception as e:
                logger.error(f"Failed to delete {file_path}. Reason: {e}")
                raise Exception(f"Failed to delete {file_path}. Reason: {e}") from e
    except Exception as e:
        logger.error(f"Error processing directory {dir_path}. Reason: {e}")
        raise Exception(f"Error processing directory {dir_path}. Reason: {e}") from e
