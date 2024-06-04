'''
Created on Apr. 9 2024

@author: OpelliusAI
@summary: 
'''
import os
from src.models import build_model as build,train_model as train,predict_model as predict
from src.utils import utils_models as um
import mlflow

from src.config.log_config import logger
import shutil
import random

def run_create_subfolders(full_classif_folder, run_folder, heatmap_output_folder, keras_tuning_folder):
    """
    Helper function to create subfolders for classification outputs.

    :param full_classif_folder: str, the base directory under which all other folders will be created.
    :param run_folder: str, name of the subfolder to hold run-specific data.
    :param heatmap_output_folder: str, name of the subfolder to store heatmap images within the run folder.
    :param keras_tuning_folder: str, name of the subfolder for storing Keras tuner project files within the run folder.
    
    :returns: tuple containing the paths to the main run folder, heatmap subfolder, and keras tuner project folder.

    :raises OSError: If any folder creation fails due to OS-related issues.
    """

    logger.debug("-------------run_create_subfolders-------------")
    
    try:
        # 1- create the run folder if it does not exist
        full_run_folder = os.path.join(full_classif_folder, run_folder)
        logger.debug(f"Target folder for run: {full_run_folder}")
        os.makedirs(full_run_folder, exist_ok=True)

        # 2- create the heatmap subfolder
        full_heatmap_subfolder = os.path.join(full_run_folder, heatmap_output_folder)
        logger.debug(f"Target folder for heatmap subfolder: {full_heatmap_subfolder}")
        os.makedirs(full_heatmap_subfolder, exist_ok=True)
        
        # 3- create the keras tuner project folder
        full_kt_folder = os.path.join(full_run_folder, keras_tuning_folder)
        logger.debug(f"Target keras tuner folder: {full_kt_folder}")
        os.makedirs(full_kt_folder, exist_ok=True)
        
        return full_run_folder, full_heatmap_subfolder, full_kt_folder
    except PermissionError as e:
        logger.error("Permission denied: unable to create directories.")
        raise PermissionError("Failed to create directories due to insufficient permissions.") from e
    except OSError as e:
        logger.error("OS error occurred while creating directories.")
        raise OSError("Failed to create directories due to an OS error.") from e

    return full_run_folder, full_heatmap_subfolder, full_kt_folder


def run_log_initial_run_info(run_id, dataset_tagname, dataset_desc, dataset_path, hyperparams):
    """
    Logs initial setup information about a machine learning run using MLflow, including details about the dataset and hyperparameters used.

    :param run_id: Unique identifier for the run.
    :param dataset_tagname: Name of the dataset used.
    :param dataset_desc: Description of the dataset.
    :param dataset_path: File path to the dataset.
    :param hyperparams: Dictionary containing hyperparameters settings.

    :raises FileNotFoundError: If the dataset_path does not exist.
    :raises Exception: For any other MLflow related issues or logging errors.
    """
    logger.debug("-------------run_log_initial_run_info-------------")
    logger.debug(f"Run ID: {run_id}, Dataset: {dataset_tagname}, Hyperparameters: {hyperparams}")

    try:
        # Log dataset name and description as MLflow tags and artifacts
        mlflow.set_tag("dataset_name", dataset_tagname)
        if os.path.exists(dataset_desc):
            mlflow.log_artifact(dataset_desc, "dataset_info")
        else:
            raise FileNotFoundError(f"Dataset description file {dataset_desc} not found.")

        # Log dataset path if it points to a valid directory or file
        if os.path.exists(dataset_path):
            mlflow.log_artifacts(dataset_path)
        else:
            raise FileNotFoundError(f"Dataset path {dataset_path} not found.")

        # Log hyperparameters
        for param, value in hyperparams.items():
            mlflow.log_param(param, value)
        
    except FileNotFoundError as e:
        logger.error(f"File not found during logging: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error during MLflow logging: {str(e)}")
        raise Exception("An error occurred during MLflow operations.") from e


def run_train_and_evaluate_model(X_train, y_train, X_eval, y_eval, hyperparams, full_run_folder, full_kt_folder, keras_tuning_history_filename, model_training_history_csv, classes_dic):
    """
    Constructs, trains, and evaluates a machine learning model using the provided datasets and hyperparameters.

    :param X_train: Training feature data.
    :param y_train: Training target data.
    :param X_eval: Evaluation feature data.
    :param y_eval: Evaluation target data.
    :param hyperparams: Dictionary of hyperparameters for model and training configuration.
    :param full_run_folder: The directory where training artifacts should be stored.
    :param full_kt_folder: The directory for Keras Tuner artifacts.
    :param keras_tuning_history_filename: Filename for saving Keras Tuner history.
    :param model_training_history_csv: CSV filename to save the training history.
    :param classes_dic: Dictionary mapping class labels to their corresponding indices.
    
    :return: Tuple containing the trained model, training history, final evaluation metrics, confusion matrix, and classification report.

    :raises Exception: If any part of the model construction, training, or evaluation fails.
    """
    logger.debug("-------------run_train_and_evaluate_model-------------")
    logger.debug(f"Number of classes: {len(classes_dic)}")

    try:
        # 1- Model construction with Keras Tuner RandomSearch to get the best hyperparams
        logger.debug("Starting model construction with Keras Tuner RandomSearch.")
        actual_num_classes=len(classes_dic)
        if hyperparams['num_classes']==1:
            # binary classification, force actual_num_classes to 1 even if we have two subfolders
            actual_num_classes=1
        best_model = build.tuner_randomsearch(hyperparams, full_run_folder, full_kt_folder, keras_tuning_history_filename, X_train, y_train, actual_num_classes)
        # 2 - Best Model training. X_train and y_train will be shuffled again
        logger.debug("Training the best model found.")
        trained_model, metrics, history = train.train_model(best_model, hyperparams, X_train, y_train, full_run_folder, model_training_history_csv)
        logger.debug(f"Initial training metrics (unlogged in MLFlow): {metrics}")

        # 3 - Model evaluation with X_eval and y_eval, and additional metrics generation
        logger.debug("Evaluating the trained model.")
        final_metrics, conf_matrix_df, class_report_df = predict.evaluate_model(trained_model, X_eval, y_eval, hyperparams["num_classes"], classes_dic)
        
    except Exception as e:
        logger.error(f"Error during model training and evaluation: {e}")
        raise Exception(f"An error occurred in training and evaluating the model: {str(e)}") from e

    return trained_model, history, final_metrics, conf_matrix_df, class_report_df

def run_save_experiment_artifacts(full_run_folder, model_filename, model_training_history_json, model_training_history_png, model_conf_m_csv, model_conf_m_png, model_class_report_csv, model_class_report_png, trained_model, history, metrics, confusion_matrix_df, classification_report_df, run_id, classes_dic):
    """
    Saves all artifacts related to a machine learning experiment, including the model, training history, and evaluation metrics.
    
    :param full_run_folder: The directory path to save all outputs.
    :param model_filename: Filename for the saved model.
    :param model_training_history_json: Filename for the training history JSON.
    :param model_training_history_png: Filename for the training history plot PNG.
    :param model_conf_m_csv: Filename for the confusion matrix CSV.
    :param model_conf_m_png: Filename for the confusion matrix plot PNG.
    :param model_class_report_csv: Filename for the classification report CSV.
    :param model_class_report_png: Filename for the classification report plot PNG.
    :param trained_model: The trained model object.
    :param history: Training history object.
    :param metrics: Dictionary of evaluation metrics.
    :param confusion_matrix_df: DataFrame containing the confusion matrix.
    :param classification_report_df: DataFrame containing the classification report.
    :param run_id: Identifier for the run.
    :param classes_dic: Dictionary of class labels and their corresponding indices.

    :raises Exception: If there is any error in saving files or logging artifacts.
    """
    logger.debug("-------------run_save_experiment_artifacts-------------")

    try:
        # Step 1 - Log metrics in MLFlow
        mlflow.log_metrics(metrics)
        logger.debug("Metrics logged in MLFlow.")

        # Step 2 - Saving the Model / history / training plots / confusion matrix / classification report
        # Step 2b - Saving the model
        model_path = um.save_model(trained_model, os.path.join(full_run_folder, model_filename))
        logger.debug(f"Model saved in path: {model_path}")
        
        # Step 2c - Saving the history file in a JSON format
        history_filepath = um.save_history(history, os.path.join(full_run_folder, model_training_history_json))
        logger.debug(f"History file saved in path: {history_filepath}")
        
        # Step 2d - Writing the files (CSV and PNG) for history, confusion matrix, and classification reports
        confusion_matrix_df.to_csv(os.path.join(full_run_folder, model_conf_m_csv))
        logger.debug(f"Confusion Matrix CSV file saved in path: {os.path.join(full_run_folder, model_conf_m_csv)}")
        
        classification_report_df.to_csv(os.path.join(full_run_folder, model_class_report_csv))
        logger.debug(f"Classification Report CSV file saved in path: {os.path.join(full_run_folder, model_class_report_csv)}")
        
        # Step 2e - Saving the plot versions of the files for history, confusion matrix, and classification reports
        um.generate_training_plots(history_filepath, os.path.join(full_run_folder, model_training_history_png), run_id)  # history plots (accuracy and loss)
        logger.debug(f"History file PNG saved in path: {os.path.join(full_run_folder, model_training_history_png)}")
        um.save_dataframe_plot(confusion_matrix_df, os.path.join(full_run_folder, model_conf_m_png), "confusion_matrix", classes_dic)
        logger.debug(f"Confusion Matrix PNG file saved in path: {os.path.join(full_run_folder, model_conf_m_png)}")
        um.save_dataframe_plot(classification_report_df, os.path.join(full_run_folder, model_class_report_png), "classification_report", classes_dic)
        logger.debug(f"Classification Report PNG file saved in path: {os.path.join(full_run_folder, model_class_report_png)}")
        
    except Exception as e:
        logger.error(f"Error saving experiment artifacts: {str(e)}")
        raise Exception(f"Failed to save experiment artifacts due to an error: {str(e)}") from e
  
def run_apply_gradcam(trained_model, test_images, full_heatmap_subfolder, prefix):
    """
    Applies GradCAM to images in a specified directory to provide visual explanations for model predictions.

    :param trained_model: The trained model on which GradCAM will be applied.
    :param test_images: Path to the directory containing test images.
    :param full_heatmap_subfolder: Directory path where GradCAM output images will be stored.
    :param prefix: Prefix for naming the output GradCAM images.

    :raises FileNotFoundError: If the test_images directory does not exist.
    :raises Exception: For issues related to reading images or applying GradCAM.
    """
    logger.debug("-------------run_apply_gradcam-------------")

    if not os.path.exists(test_images):
        logger.error(f"Test images directory not found: {test_images}")
        raise FileNotFoundError(f"Test images directory not found: {test_images}")

    # File extensions to consider for processing
    valid_extensions = ('.png', '.jpg', '.jpeg')

    try:
        # Traverse the directory to process each image
        for file in os.listdir(test_images):
            if file.lower().endswith(valid_extensions) and "gradcam_" not in file.lower():
                # Construct the full path to the image file
                full_path = os.path.join(test_images, file)
                # Apply GradCAM and get the path of the output image
                img_cam_path = um.get_img_gradcam(full_path, trained_model, full_heatmap_subfolder, prefix)
                logger.debug(f"Generated GradCAM image at: {img_cam_path}")
    except Exception as e:
        logger.error(f"Error applying GradCAM: {e}")
        raise Exception(f"Error applying GradCAM: {str(e)}") from e
    


def extract_items_from_subdirectories(source_directory, target_directory, num_items):
    """
    Extracts a specified number of items randomly from each subdirectory of the source directory and copies them to a corresponding structure in the target directory.

    :param str source_directory: The path to the source directory containing subdirectories from which files are to be randomly copied.
    :param str target_directory: The path to the target directory where files will be copied.
    :param int num_items: The maximum number of items to copy from each subdirectory.

    :raises FileNotFoundError: If the source directory does not exist.
    :raises ValueError: If num_items is less than 1 or greater than the number of files in any subdirectory.
    """
    
    # Validate input parameters
    if not os.path.exists(source_directory):
        raise FileNotFoundError(f"The source directory {source_directory} does not exist.")
    
    # Ensure the target directory exists
    os.makedirs(target_directory, exist_ok=True)
    
    # Walk through the subdirectories in the source directory
    for subdir, _, files in os.walk(source_directory):
        if subdir != source_directory:  # Ignore the root directory
            if num_items > len(files):
                raise ValueError(f"num_items must be less than or equal to the number of files in each subdirectory. Subdirectory {subdir} has only {len(files)} files.")
            
            # Randomly select the specified number of items (files) to extract
            selected_files = random.sample(files, num_items)
            
            # Copy each selected file to the target directory
            for file in selected_files:
                # Full path of the source file
                source_file_path = os.path.join(subdir, file)
                # Full path of the target file
                target_file_path = os.path.join(target_directory, os.path.basename(subdir), file)
                
                # Ensure the target subdirectory exists
                os.makedirs(os.path.dirname(target_file_path), exist_ok=True)
                
                # Copy the file
                shutil.copy2(source_file_path, target_file_path)

