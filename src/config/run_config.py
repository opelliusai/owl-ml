'''
Created on: Apr. 12 2024

@author: OpelliusAI
@summary: Configuration file that contains information about the main run paths and datasets, and hyperparameterrs

## Example is provided in data/raw/datasets folder: 
for sick / Healthy classification, we use two datasets options : with and without lung opacity to check if there are any improvements of the predictions when not using lung opacity is removed
# experiment contain the experiment_name and run_name as displayed on MLFlow
# experiment Healthy/Sick
experiments_hs = {
    "experiment_name": "Classe_Healthy_Sick",
    "run_name": "Healthy_Sick",
    "classes":["Healthy","Sick"],
    "classes_dic":{0: 'Healthy', 1: 'Sick'},
    "classif_folder":"classif_hs"
}
# Dataset : Uses the "full_datasets_folder" value and contains dataset path and json file that will be displayed in MLFlow . tag_name is used for MLFlow filtering
dataset_hs_no_lo = { 
    "tag_name": "750 no_lo Healthy/Sick",
    "current_dataset_path": "Healthy_Sick_750_Dataset_HorsLO",
    "dataset_desc": "Healthy_Sick_no_lo_750_Metadata.json",
}
dataset_hs = { 
    "tag_name": "750 Healthy/Sick",
    "current_dataset_path": "Healthy_Sick_750_Dataset_All",
    "dataset_desc": "Healthy_Sick_750_Metadata.json",
}
'''

# set the absolute path
import os
import sys
sys.path.insert(0,os.path.abspath("."))

######
## PROJECT and RUN Paths and file creation information (prefix)
######

paths = {
    # Full dataset folder
    "full_datasets_folder": "data/raw/datasets", # isolate the folder path from the project path in order to use external sources that are common to other projects
    # Folder test images folder
    "full_test_images":"data/raw/test_images",
    ## based on the main_path:
    "main_path": ".",
    "model_outputs": "data/processed/mlflow", # main_path/data/processed/models/mlflow
    "data_folder": "data",
    "models_path": "models",
    "csv_path":"csv",
    "figures_path":"figures",    
    "full_external_images":"data/raw/external_images",
    "sm_subfolder":"classif_sm",
    "cv_subfolder":"classif_cv",
    "mc_subfolder":"classif_mc",
    "3c_subfolder":"classif_3c",
    # mlflow information
    "artifact_path" : "COVID19-Project",
    
}

######
## LOG management
######
infolog = {
    "project_name":"COVID19-Project",
    "logs_folder": "logs",
    "logfile_name": "mlflow.log",
}


######
## Experiments and associated datasets
######

## EXAMPLES:

## Experiment BINARY Healthy/Sick
experiments_hs = {
    "experiment_name": "Class_Healthy_Sick",
    "run_name": "Healthy_Sick",
    "classes_dic":{0: 'Healthy', 1: 'Sick'},
    "classif_folder":"classif_hs"
}


# Datasets : to join with the paths['datasets_folder']  Each dataset should contain Healthy and Sick subfolders with the associated images
dataset_hs = { 
    "tag_name": "120 Healthy/Sick",
    "current_dataset_path": "Healthy_Sick_120_Dataset_All",
    "dataset_desc": "Healthy_Sick_120_Metadata.json",
}


## Experiment Multiclasses
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

hyperparams_list = [
    ### EfficientNet
    # Binary Healthy/Sick with VGG16
    {"archi": "VGG16", "img_size": 224, "img_dim": 3, "num_classes": 1, "hl_activation": "relu", "max_epochs": 30, "factor": 3, "num_trials": 5, "data_lo_no_lo": "NA","last_conv_layer":"top_conv"},
    # 4 classes
    {"archi": "EfficientNetB0", "img_size": 224, "img_dim": 3, "num_classes": 4, "hl_activation": "relu", "max_epochs": 30, "factor": 3, "num_trials": 5, "data_lo_no_lo": "NA","last_conv_layer":"top_conv"},
       
]
