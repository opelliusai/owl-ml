'''
Created on: Apr. 9 2024

@author: OpelliusAI
@summary: The main module that executes mlflow experiments for each classification and model type, using the config hyperparams file 
@info: Uses experiments_ and datasets_ to select the datasets and classification and experiment name information.
## Example: 
experiments_mc = {
    "experiment_name": "Multiclasses",
    "run_name": "MC",
    "classes_dic":{0: 'COVID', 1: 'Lung_Opacity', 2: 'Normal', 3: 'Viral_Pneumonia'},
    "classif_folder":"classif_mc"
}

# Dataset : to join with the paths['datasets_folder'] 
dataset_mc = { 
    "tag_name": "120 Multiclasses",
    "current_dataset_path": "Covid-19_MC_120_Dataset_All",
    "dataset_desc": "Covid_MC_120_Metadata.json"
}

'''
#####
# Imports 
#####
# MLFlow
import mlflow

## Useful modules imports
import os

## Import of python modules for each step of model management
from src.features import build_features as bf
from src.utils import utils_runs as ur

## Dictionaries to customize based on your needs
# Here we use 4 experiments: Healthy/Sick, Not Covid/Covid, 3 classes (without lung opacity) and multiple classes
#from src.config.run_config import experiments_hs,dataset_hs_no_lo,dataset_hs
#from src.config.run_config import experiments_cv,dataset_cv_no_lo,dataset_cv
from src.config.run_config import experiments_mc,dataset_mc,experiments_hs, dataset_hs
from src.config.run_config import paths,hyperparams_list
from src.config.log_config import logger


dataset_folder = paths["full_datasets_folder"]
# Each experiment function retrieves the datasets associated with the classification.
# Each experiment uses two dictionaries from the configuration file:
## - paths
## - dataset_
## - experiments_

def run_experiment_generic(hyperparams,experiment,dataset):
    """
    Execute experiments for multi-class classification. This involves setting up directories for various outputs (classification, history, training plots, etc.), preparing the dataset, splitting it, training the model with hyperparameters, and saving all relevant artifacts, including the trained model and evaluation metrics.

    :param hyperparams: A dictionary containing hyperparameters and configurations for the experiment, including image size, dimensions, architecture, and more.
    :type hyperparams: dict
    
    :raises FileNotFoundError: If specified dataset paths or directories do not exist.
    :raises Exception: General exception for any unforeseen errors during the experiment's execution.

    The function logs all steps, creates necessary directories if they don't exist, splits the dataset, trains the model, and evaluates it. It concludes by saving the model, metrics, and various reports.
    """

    try:
        logger.debug("-----------run_experiment_GENERIC------------")
        logger.debug(f"Execution for experiment {experiment['experiment_name']}")
        logger.debug(f"Execution for Dataset {dataset['tag_name']}")
        
        # 1 - Retrieving information on the data to use
        logger.debug(f"Retrieving information on the data to use")
        dataset_path = os.path.join(dataset_folder, dataset["current_dataset_path"])
        full_classif_folder = os.path.join(paths["main_path"], paths["model_outputs"], experiment["classif_folder"])
        
        # 2 - Create the classification folder if it does not exist
        logger.debug(f"Create the classification folder if it does not exist")
        os.makedirs(full_classif_folder, exist_ok=True)
    
        # 3 - Retrieving the name of the description file (JSON)
        logger.debug(f"Retrieving the name of the description file (JSON)")
        dataset_desc = os.path.join(dataset_path, dataset["dataset_desc"])
        classes_dic = experiment["classes_dic"]
        dataset_tagname = dataset["tag_name"]

        logger.debug(f"Dataset path: {dataset_path}")
        logger.debug(f"Classes dictionary: {classes_dic}")
                
        # 4 - Splitting data into training/validation and evaluation sets
        logger.debug(f"Splitting data into training/validation and evaluation sets")
        X_train, y_train, X_eval, y_eval, dic = bf.get_data_from_parent_directory_labelled(
            dataset_path, hyperparams['img_size'], hyperparams['img_dim'], hyperparams['archi']
        ) # classes_dic

        logger.debug("Directory to Label Mapping: ")
        logger.debug(f"{dic}")

        # 5 - Initiating the experiment
        logger.debug("Initiating the experiment")
        mlflow.set_experiment(experiment["experiment_name"])

        with mlflow.start_run(run_name=experiment["run_name"]) as run:
            logger.debug("Initial run log info")
            ur.run_log_initial_run_info(run.info.run_id, dataset_tagname, dataset_desc, dataset_path, hyperparams)

            # 1b - Folders and files naming preparation to ensure common naming conventions and logic
            logger.debug(f"--------------------Files and folder naming preparation--------------------")
            # run folder
            run_folder= experiment['classif_folder']+"_"+hyperparams['archi']+"_"+run.info.run_id
            logger.debug(f"Target RUN folder {run_folder}")
            # keras tuning ex: classif_hs_VGG16_RS_7d1a75a08fca46d6adf5879dbf12c91a
            keras_tuning_folder=experiment['classif_folder']+"_"+hyperparams['archi']+"_RS_"+run.info.run_id
            logger.debug(f"Target Keras tuning folder {keras_tuning_folder}")
            ## Keras tuning history csv file
            keras_tuning_history_filename=experiment['classif_folder']+"_"+hyperparams['archi']+"_tuning_history_"+run.info.run_id+".csv"
            logger.debug(f"Target Keras tuning history filename {keras_tuning_history_filename}")
            ## Model file
            model_filename=experiment['classif_folder']+"_"+hyperparams['archi']+"_model_"+run.info.run_id+".keras"
            logger.debug(f"Target Model filename {model_filename}")
            ## Model training history JSON, csv and png
            model_training_history=experiment['classif_folder']+"_"+hyperparams['archi']+"_model_history_"+run.info.run_id
            model_training_history_json=model_training_history+".json"
            model_training_history_csv=model_training_history+".csv"
            model_training_history_png=model_training_history+".png"
            logger.debug(f"Target Model training history JSON {model_training_history_json}")
            logger.debug(f"Target Model training history CSV {model_training_history_csv}")
            logger.debug(f"Target Model training history PNG {model_training_history_png}")
            ## Model Confusion Matrix PNG
            model_conf_m=experiment['classif_folder']+"_"+hyperparams['archi']+"_confusion_matrix_"+run.info.run_id
            model_conf_m_csv=model_conf_m+".csv"
            model_conf_m_png=model_conf_m+".png"
            logger.debug(f"Target Model Confusion matrix CSV {model_conf_m_csv}")
            logger.debug(f"Target Model Confusion matrix PNG {model_conf_m_png}")
            ## Model Classification Report PNG
            model_class_report=experiment['classif_folder']+"_"+hyperparams['archi']+"_classification_report_"+run.info.run_id
            model_class_report_csv=model_class_report+".csv"
            model_class_report_png=model_class_report+".png"
            logger.debug(f"Target Model Classification report CSV {model_class_report_csv}")
            logger.debug(f"Target Model Classification report PNG {model_class_report_png}")
            ## Heatmap folder
            heatmap_output_folder=experiment['classif_folder']+"_"+hyperparams['archi']+"_heatmaps_"+run.info.run_id
            logger.debug(f"Target Heatmap output folder {heatmap_output_folder}")
            
            logger.debug(f"---------------------------------END---------------------------------")
            
            '''
            Variable names to use
            run_folder
            keras_tuning_folder
            keras_tuning_history_filename
            model_filename
            model_training_history_json
            model_training_history_csv
            model_training_history_png
            model_conf_m_csv
            model_conf_m_png
            model_class_report_csv
            model_class_report_png
            heatmap_output_folder
            '''
            # 2 - Additional directories for storing various outputs: Run folder and heatmap subfolder, returns the foldernames
            logger.debug("Run subfolders creation")
            full_run_folder, full_heatmap_subfolder, full_kt_folder=ur.run_create_subfolders(full_classif_folder, run_folder, heatmap_output_folder, keras_tuning_folder)
            
            # Model construction, training, and evaluation
            logger.debug("Model construction, training, and evaluation")
            trained_model, history, metrics, confusion_matrix_df, classification_report_df = ur.run_train_and_evaluate_model(X_train, y_train, X_eval, y_eval, hyperparams, full_run_folder, full_kt_folder, keras_tuning_history_filename, model_training_history_csv, dic)
            
            # Saving generated elements: model, history, plots, confusion matrix, classification report
            logger.debug("Saving generated elements - model, history, plots etc.")
            ur.run_save_experiment_artifacts(
                full_run_folder,model_filename,model_training_history_json,model_training_history_png,model_conf_m_csv,model_conf_m_png,model_class_report_csv,model_class_report_png,
                trained_model, 
                history, 
                metrics, 
                confusion_matrix_df, 
                classification_report_df,
                run.info.run_id, 
                dic)
    
            ''' 
            # Gradcam for interpretability if applicable - NEXT ITERATION
            logger.debug("Gradcam Interpretability (TO DO)")
            prefix=experiment["classif_folder"]+"_"+hyperparams['archi']+"_gradcam"
            logger.debug(f"full_heatmap_subfolder {full_heatmap_subfolder}")
            ur.run_apply_gradcam(trained_model, paths["full_test_images"],full_heatmap_subfolder, prefix)
            '''
            
    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}")
        raise
    except Exception as e:
        logger.error(f"An error occurred during the experiment: {e}")
        raise


# Main function
if __name__ == "__main__":
    for hyperparams in hyperparams_list:
        if hyperparams["num_classes"]==1: #binary
            logger.debug("Binary classification")
            run_experiment_generic(hyperparams, experiments_hs, dataset_hs)
        elif hyperparams["num_classes"]>1: #multiple dynamic
            logger.debug("Multiple dynamic")
            run_experiment_generic(hyperparams, experiments_mc, dataset_mc)
        '''
        # Other example
        elif hyperparams["num_classes"]==4: # multiple  
            # Muliclass experiment execution on data
            run_experiment_generic(hyperparams, experiments_mc, dataset_mc)
        elif hyperparams["num_classes"]==3: # 3 classes
            logger.debug("3 classes")
            # 3 class experiment execution on data and classes without Lung Opacity
            #run_experiment_generic(hyperparams, experiments_3c, dataset_3c)
            pass
        '''
