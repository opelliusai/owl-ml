'''
Created on: Apr. 9 2024

@author: OpelliusAI
@summary: model predictions and evaluation functions. Common to all model architectures
@info: Step 3 of the model execution 'Step 1 being the build feature and Step 2 build model'
1- >>features/build_features.py<< Preprocessing: Use for sampling and preprocessing.
2- >>models/build_model.py<< Define a building block for each model architecture (option to Freeze or Unfreeze layers).
3- >>models/build_model.py<< Identify the specifics between Binary and Multiple in model construction and result visualization.
4- >>models/build_model.py<< Identify parameters to adjust for Keras_tuner.
5- >>models/build_model.py<< Define EarlyStopping conditions.
6- >>models/build_model.py<< Launch execution on X epochs with Hyperband.
7- >>models/build_model.py<< Retrieval of the best model.
8- >>models/train_model.py<< Training the model on the number of epochs.
9- >>models/train_model.py<< Retrieval of the trained model.
10->>models/train_model.py<< Retrieval of metrics.
11->>models/train_model.py<< Advanced metrics: Retrieval of the name of the last convolutional layer for GradCam display.
12->>models/predict_model.py<< 
13->>(not implemented)visualization/visualize.py<< Display of the Confusion Matrix with the business objective.
'''

#####
# Imports 
#####
## Useful modules import
import pandas as pd

## Internal imports
from src.utils import utils_models as um
### Configuration file import
'''
src.config file contains information of the repository
paths[]: Folders and subfolders to find or generate information ex : paths['main'] is the project path
infolog[]: Logging information : Folder, logname ex : utils_models.log
'''
from src.config.log_config import logger

# Model evaluation function
def evaluate_model(model, X_eval, y_eval, num_classes, classes):
    """
    Evaluates the model and returns metrics, confusion matrix, and classification report.
    
    :param model: Model to be evaluated.
    :param X_eval: Evaluation images.
    :param y_eval: Evaluation targets.
    :param num_classes: Number of classes, defining binary or multiclass classification.
    :param classes: Dictionary for class mapping.
    
    :return: A tuple containing metrics dictionary, confusion matrix dataframe, and classification report dataframe.
             Confusion matrix and Classification report will be saved as files by MLFlow.
             Metrics will be stored in the RUN.

    :raises Exception: If an error occurs during the model evaluation process.
    """

    try:
        # 1- Retrieving additional metrics. 
        # Specific functions have been created for binary and multiple classification because of the high divergence of the metrics generation and management
        metrics_dict = um.bin_get_prediction_metrics(model, X_eval, y_eval) if num_classes == 1 else um.multi_get_prediction_metrics(model, X_eval, y_eval)
        logger.debug(f"metrics_dict {metrics_dict}")
        # Building the metrics dictionary
        final_metrics = {
            "Accuracy": metrics_dict["Accuracy"],
            "Recall": metrics_dict["Recall"],
            "F1-score": metrics_dict["F1-score"]
        }
        '''
        for i, (sensitivity, specificity) in enumerate(zip(metrics_dict["Sensitivity - Recall"], metrics_dict["Specificity"])):
            final_metrics[f'Recall sensitivity_class_{i}'] = sensitivity
            final_metrics[f'Recall specificity_class_{i}'] = specificity
        '''
        logger.debug(f"Metrics: {final_metrics}")
        logger.debug(f"Confusion Matrix: {metrics_dict['Confusion Matrix']}")
        logger.debug(f"Classification Report: {metrics_dict['Classification report']}")

        # Confusion matrix as dataframe
        conf_matrix_df = pd.DataFrame(metrics_dict["Confusion Matrix"], index=[i for i in classes], columns=[i for i in classes])
        # Classification report as dataframe
        class_report_df = pd.DataFrame(metrics_dict["Classification report"]).transpose()
        
        return final_metrics, conf_matrix_df, class_report_df

    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise Exception(f"Error during model evaluation: {e}")
