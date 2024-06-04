"""
.. module:: build_model
   :synopsis: Module for building and tuning models, either pretrained or from scratch.

    Created on: Apr. 9 2024

.. moduleauthor:: OpelliusAI

This module is designed for building and tuning machine learning models using Keras and MLFlow. 
The primary entry points are the functions :func:`tuner_randomsearch` and :func:`tuner_hyperband`, 
which orchestrate the model building process by calling the main :func:`build_model` function.

The :func:`build_model` function leverages both Keras tuning hyperparameters (``hp``) and MLFlow hyperparameters (``ml_hp``) 
for maximum flexibility. It directs these parameters to a specific ``build_model_ARCHITECTURE`` function based on the model architecture.

**Execution Flow:**

1. **Preprocessing** (:mod:`features.build_features`): Sampling and preprocessing.
2. Define a building block for each model architecture, with options to freeze or unfreeze layers.
3. Differentiate between binary and multiple class models in terms of construction and result visualization.
4. Identify parameters for Keras tuner adjustment.
5. Define conditions for early stopping.
6. Execute model tuning over X epochs using RandomSearch.
7. Retrieve the best model post-tuning.
8. Train the best model on the designated number of epochs (:mod:`models.train_model`).
9. Store the trained model.
10. Gather model metrics and store them in json/csv and png files (train history, confusion matrix and classification report)
11. Model prediction (:mod:`models.predict_model`).
Run information are stored in data/processed/mlflow/classif_mc (for multiple classes), see run_config file for the output folder
**Adding a New Model:**

To integrate a new model architecture:

- Create the ``build_model`` function specific to the new architecture (e.g., :func:`build_model_efficientnetb0`).
- Update the main :func:`build_model` function to include a conditional block that checks the architecture name and forwards the hyperparameters to the dedicated function (e.g., if 'efficientnetb0' in archi: build_model_efficientnetb0(hp, ml_hp)).
- Modify the :mod:`features.build_features` function and add a condition in the ``preprocess_input_archi(img, archi)`` function, including importing the specific ``preprocess_input`` function from the model's modules.
"""


#####
# Imports 
#####
### Pretrained Models object and preprocessing function imports, renamed with the model architecture  name
# VGG16
from tensorflow.keras.applications.vgg16 import VGG16
# VGG19
from tensorflow.keras.applications.vgg19 import VGG19 # VGG16
# ResNet50
from tensorflow.keras.applications.resnet50 import ResNet50
# EfficientNetB0
from tensorflow.keras.applications.efficientnet import EfficientNetB0

## Imports related to the Neural Networks construction
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization,Flatten,Dropout,MaxPooling2D,Conv2D # Layers etc.
from tensorflow.keras.models import Model,Sequential # Models objects
from tensorflow.keras.optimizers import Adam,SGD # Optimizers
from tensorflow.keras.losses import BinaryCrossentropy # Loss function
from tensorflow.keras.regularizers import l2 # Regularizers 
from tensorflow.keras.callbacks import EarlyStopping,CSVLogger,LearningRateScheduler# Callbacks

## Import keras tuner
from kerastuner.tuners import RandomSearch, Hyperband

## Useful modules import
import numpy as np
import os
from sklearn.model_selection import train_test_split
import time

### Internal imports
from src.features import build_features as bf

### Configuration file import
'''
src.config.run_config file contains information of the repository
paths[]: Folders and subfolders to find or generate information ex : paths['main_path'] is the project path
infolog[]: Logging information : Folder, logname ex : utils_models.log
src.config.log_config uses run_config to get information from the log generation name and target path
'''
from src.config.log_config import logger

### FUNCTIONS
# Keras Tuning - Random Search
def tuner_randomsearch(ml_hp, full_run_folder, full_kt_folder, keras_tuning_history_filename, X, y,actual_num_classes):
    """
    Performs hyperparameter tuning using RandomSearch with specified machine learning hyperparameters, 
    logging the process, and utilizing callbacks like EarlyStopping and LearningRateScheduler.

    uses the local build_model function.
    :param ml_hp: Dictionary containing machine learning hyperparameters including 'max_epochs', 
                  'num_trials', and 'archi' for the architecture.
    :param run_id: Identifier for the current run, used for logging purposes.
    :param directory: Directory path where the tuning artifacts will be stored.
    :param project_name: Name of the project, used for organizing the tuning results.
    :param X: Features dataset, expected to be in a format suitable for model training (e.g., numpy array).
    :param y: Labels dataset, expected to be in a format suitable for model training (e.g., numpy array).
    :param actual_num_classes: Number of classes is dynamic therefore it is provided after the subfolders processing
    
    Tuning training logs are stored in the main_model_logs folder with the naming pattern: rs_tuning_training_log_{archi}_{run_id}.csv using ';' separator
    :return: The best model found during the RandomSearch tuning process.
    
    :raises KeyError: If required keys are missing in the `ml_hp` dictionary.
    :raises FileNotFoundError: If the specified directory does not exist.
    :raises Exception: For other exceptions not explicitly caught related to model building and tuning.
    """
    try:
        logger.debug("---------tuner_randomsearch------------")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1234)

        # Logging split information
        logger.debug(f"Sizes X: {len(X)}, y: {len(y)}, X_train: {len(X_train)}, y_train: {len(y_train)}, X_val: {len(X_val)}, y_val: {len(y_val)}")
        logger.debug(f"Shapes X: {X.shape}, y: {y.shape}, X_train: {X_train.shape}, y_train: {y_train.shape}, X_val: {X_val.shape}, y_val: {y_val.shape}")

        max_epochs = ml_hp['max_epochs']
        num_trials = ml_hp['num_trials']
                # Callbacks configuration
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
        
        logger.debug(f" KT search history file name  {keras_tuning_history_filename}")
        full_csv_logger_path=os.path.join(full_run_folder,keras_tuning_history_filename)
        csv_logger = CSVLogger(full_csv_logger_path, append=True, separator=';')
        logger.debug(f" KT search history file name stored in  {full_csv_logger_path}")
        
        # Learning rate adjustment
        def scheduler(epoch, lr):
            if epoch < 10:
                return lr
            else:
                return lr * np.exp(-0.1)
        lr_scheduler = LearningRateScheduler(scheduler, verbose=1)
        
        # Keras Tuner configuration
        tuner = RandomSearch(
            hypermodel=lambda hp: build_model(hp=hp, ml_hp=ml_hp,actual_num_classes=actual_num_classes),
            objective='val_accuracy',
            max_trials=num_trials,
            directory=full_kt_folder,
            project_name="trials"
        )
        
        # Tuning execution
        start_time = time.time()
        tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=max_epochs, callbacks=[early_stopping, lr_scheduler, csv_logger])
        end_time = time.time()
        tuning_time = round(end_time - start_time, 2) / 60
        logger.debug(f"Keras Tuning time {tuning_time} min")
        
        # Best model retrieval
        best_hp = tuner.get_best_hyperparameters()[0]
        best_model = tuner.hypermodel.build(best_hp)
        
        logger.debug(f"Best Hyperparameters {best_hp}")
        return best_model

    except KeyError as e:
        logger.error(f"Missing key in ml_hp: {e}")
        raise KeyError(f"Missing key in ml_hp: {e}") from None
    except FileNotFoundError as e:
        logger.error(f"Directory not found: {e}")
        raise FileNotFoundError(f"Directory not found: {e}") from None
    except Exception as e:
        logger.error(f"An error occurred during tuning: {e}")
        raise Exception(f"An error occurred during tuning: {e}") from None

# Keras Tuning - Hyperband
def tuner_hyperband(ml_hp, run_id, directory, project_name, X, y):
    """
    CURRENTLY NOT USED
    Performs hyperparameter tuning using the Hyperband optimization algorithm on a given dataset.

    :param ml_hp: Dictionary containing MLFlow hyperparameters, including 'max_epochs', 'factor', and 'archi' for architecture.
    :param run_id: Identifier for the current run, used for logging and file naming.
    :param directory: Directory path where the tuning artifacts will be stored.
    :param project_name: Name of the project, used for organizing the tuning results.
    :param X: Features dataset, expected to be in a format suitable for model training (e.g., numpy array).
    :param y: Labels dataset, expected to be in a format suitable for model training (e.g., numpy array).
    
    :return: The best model found during the Hyperband tuning process.

    :raises ValueError: If 'max_epochs' or 'factor' keys are missing from ml_hp.
    :raises FileNotFoundError: If the specified directory does not exist.
    """
    logger.debug("---------tuner_hyperband------------")

    try:
        # Data splitting might not directly slow down execution but is part of preprocessing that precedes tuning.
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1234)
        
        # Retrieving necessary hyperparameters from ml_hp
        max_epochs = ml_hp.get('max_epochs')
        factor = ml_hp.get('factor')
        archi = ml_hp.get('archi')
        if max_epochs is None or factor is None:
            raise ValueError("'max_epochs' and 'factor' must be specified in ml_hp.")

        # Setting up callbacks
        main_models_log = os.path.join(directory, f"hb_tuning_training_log_{archi}_{run_id}.csv")
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
        csv_logger = CSVLogger(main_models_log, append=True, separator=';')

        def scheduler(epoch, lr):
            return lr * np.exp(-0.1) if epoch >= 10 else lr
        lr_scheduler = LearningRateScheduler(scheduler, verbose=1)
        
        # The Hyperband tuner can be computationally intensive and slow depending on 'max_epochs', 'factor', and the complexity of the model.
        tuner = Hyperband(
            hypermodel=lambda hp: build_model(hp=hp, ml_hp=ml_hp),
            objective='val_accuracy',
            max_epochs=max_epochs,
            factor=factor,
            directory=directory,
            project_name=project_name
        )

        # The search method is where the bulk of computation time will be spent.
        start_time = time.time()
        tuner.search(X_train, y_train, validation_data=(X_val, y_val), callbacks=[early_stopping, lr_scheduler, csv_logger])
        end_time = time.time()
        
        execution_time = round(end_time - start_time, 2) / 60
        logger.debug(f"Keras Tuning time {execution_time} min")
        
        best_hp = tuner.get_best_hyperparameters()[0]
        best_model = tuner.hypermodel.build(best_hp)

        return best_model

    except ValueError as ve:
        logger.error(f"ValueError: {ve}")
        raise
    except FileNotFoundError as fnfe:
        logger.error(f"FileNotFoundError: {fnfe}")
        raise

# Main build model function
def build_model(hp, ml_hp, actual_num_classes):
    """
    Builds a Keras model based on specified architecture and hyperparameters. 
    It delegates the model construction to architecture-specific functions, allowing for customization 
    of pre-trained models, including hyperparameters tuning and layer unfreezing. 
    This design supports team collaboration by enabling individual contributions to model configuration.

    :param hp: Hyperparameter object provided by Keras Tuner for dynamic hyperparameter tuning.
    :param ml_hp: Dictionary containing machine learning hyperparameters, including model architecture under the 'archi' key.
    :param actual_num_classes: Number of classes is dynamic therefore it is provided after the subfolders processing
    
    :return: A configured Keras model based on the specified architecture and hyperparameters.
    
    :raises KeyError: If the 'archi' key is missing from the ml_hp dictionary.
    :raises ValueError: If the architecture specified in ml_hp['archi'] is not recognized or supported.
    """
    logger.debug("---------build_model------------")
    
    try:
        archi = ml_hp['archi'].lower()
    except KeyError:
        logger.error("KeyError: 'archi' key missing from ml_hp dictionary.")
        raise KeyError("'archi' key is missing from the ml_hp dictionary.")
    
    if archi == 'lenet':
        return build_model_lenet(hp, ml_hp,actual_num_classes)
    elif archi == 'vgg16':
        return build_model_vgg16(hp, ml_hp,actual_num_classes)
    elif archi == 'vgg19':
        return build_model_vgg19(hp, ml_hp,actual_num_classes)
    elif archi == 'resnet50':
        return build_model_resnet50(hp, ml_hp,actual_num_classes)
    elif archi == 'efficientnetb0':
        return build_model_efficientnetb0(hp, ml_hp,actual_num_classes)
    else:
        logger.error(f"ValueError: Unsupported model architecture - {archi}")
        raise ValueError(f"Unsupported model architecture specified: {archi}")


#######################
#   MODELS SECTION    #
#######################

######
## Model : LeNet
######

def build_model_lenet(hp, ml_hp, actual_num_classes):
    """
    Constructs a LeNet architecture model for image classification (binary or multi-class) using both
    hyperparameters for Keras Tuner optimization and general hyperparameters from a configuration object.
    This configuration object contains information such as:
    - 'archi': defining the architecture to build,
    - 'img_size': defining the image size for model input,
    - 'img_dim': color dimensions (RGB 3 or Grayscale 1),
    - etc.

    :param hp: HyperParameters object from Keras Tuner for optimizing model hyperparameters.
    :param ml_hp: MLFlow HyperParameters object containing configuration details.
    :param actual_num_classes: Number of classes is dynamic therefore it is provided after the subfolders processing
    
    :return: A compiled Keras model optimized with hyperparameters.

    :raises KeyError: If essential keys are missing from the ml_hp dictionary.
    :raises ValueError: If the configuration values in ml_hp are invalid or unsupported, such as an unsupported number of classes.
    """
    logger.debug("---------build_model_lenet------------")

    try:
        # 1a - Retrieving MLFlow hyperparameters
        archi = ml_hp["archi"]
        img_size = ml_hp["img_size"]
        img_dim = ml_hp["img_dim"]
        num_classes = ml_hp["num_classes"]
        
        # 1b - Optional parameters will have a default value
        hidden_layers_activation = ml_hp.get("hl_activation", "relu")

        # 1c - Initialization of additional variables based on mlflow_archive hyperparameters
        shape = (img_size, img_size, img_dim)

        # 2 - Classification specifics: loss function
        if num_classes == 1:
            logger.debug("--- BINARY CLASSIFICATION ------")
            loss_function = 'binary_crossentropy'
            output_activation = "sigmoid"
        else: #0 refers to dynamic multiple classification , therefore any num_classes other than 1 will be non binary
            logger.debug("--- MULTICLASS CLASSIFICATION ------")
            loss_function = 'sparse_categorical_crossentropy'
            output_activation = "softmax"
        
        logger.debug("--- Hyperparameters ---")
        # 3- Logging hyperparameters
        for param in ['archi', 'img_size', 'img_dim', 'shape', 'num_classes', 'hidden_layers_activation', 'loss_function', 'output_activation']:
            logger.debug(f"{param} = {locals()[param]}")
        
        logger.debug(f"actual_num_classes = {actual_num_classes}")
        # 4 - Defining hyperparameters to be optimized with Keras Tuner
        learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        #units = hp.Int('units', min_value=32, max_value=512, step=32)
        dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
        l2_lambda = hp.Choice('l2_lambda', values=[1e-3, 1e-4, 1e-5])
        num_dropout_layers = hp.Int('num_dropout_layers', min_value=1, max_value=5)

        # 5 - Building the base model
        model = Sequential([
            Conv2D(30, (5, 5), input_shape=shape, activation=hidden_layers_activation, kernel_regularizer=l2(l2_lambda)),
            MaxPooling2D((2, 2)),
            Conv2D(16, (3, 3), activation=hidden_layers_activation, kernel_regularizer=l2(l2_lambda)),
            MaxPooling2D((2, 2)),
            Dropout(dropout_rate),
            Flatten(),
            Dense(128, activation=hidden_layers_activation, kernel_regularizer=l2(l2_lambda)),
            *[Dropout(dropout_rate) for _ in range(num_dropout_layers)],
            Dense(actual_num_classes, activation=output_activation)
        ])

        # 6 - Compiling the model
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss_function, metrics=['accuracy'])

        return model
    except KeyError as e:
        logger.error(f"KeyError: Missing key in ml_hp: {e}")
        raise KeyError(f"Missing key in ml_hp: {e}") from None
    except ValueError as e:
        logger.error(f"ValueError: {e}")
        raise ValueError(e) from None
    
######
## Model : VGG16
######
def build_model_vgg16(hp, ml_hp, actual_num_classes):
    """
    Builds a VGG16 architecture model for image classification, supporting both binary and multi-class scenarios.
    Utilizes hyperparameters for Keras Tuner optimization as well as general configuration parameters from a configuration object.
    This configuration includes details such as:
    - 'archi': defining the architecture to build,
    - 'img_size': defining the image size for model input,
    - 'img_dim': color dimensions (RGB 3 or Grayscale 1),
    - etc.

    :param hp: HyperParameters object from Keras Tuner for optimizing model hyperparameters.
    :param ml_hp: MLFlow HyperParameters object containing configuration details.
    :param actual_num_classes: Number of classes is dynamic therefore it is provided after the subfolders processing
    
    :return: A compiled Keras model with optimized hyperparameters.

    :raises KeyError: If essential keys are missing from the ml_hp dictionary.
    :raises ValueError: If the configuration values in ml_hp are invalid or unsupported, such as an unsupported number of classes.
    """
    logger.debug("---------build_model_vgg16------------")
                                                               
    try:
        # 1a - Retrieving MLFlow hyperparameters
        archi = ml_hp["archi"]
        img_size = ml_hp["img_size"]
        img_dim = ml_hp["img_dim"]
        num_classes = ml_hp["num_classes"]

        # 1b - Optional parameters will have a default value
        hidden_layers_activation = ml_hp.get("hl_activation", "relu")

        # 1c - Initialization based on mlflow_archive hyperparameters
        shape = (img_size, img_size, img_dim)

        # 2 - Classification specifics: loss function
        if num_classes == 1:
            logger.debug("--- BINARY CLASSIFICATION ------")
            loss_function = 'binary_crossentropy'
            output_activation = "sigmoid"
        else: #0 refers to dynamic multiple classification , therefore any num_classes other than 1 will be non binary
            logger.debug("--- MULTICLASS CLASSIFICATION ------")
            loss_function = 'sparse_categorical_crossentropy'
            output_activation = "softmax"
        
        logger.debug("--- Hyperparameters ---")
        
        # 3 - Logging hyperparameters
        for param in ['archi', 'img_size', 'img_dim', 'shape', 'num_classes', 'hidden_layers_activation', 'loss_function', 'output_activation']:
            logger.debug(f"{param} = {locals()[param]}")

        logger.debug(f"actual_num_classes = {actual_num_classes}")
        
        # 4 - Defining hyperparameters to be optimized with Keras Tuner
        learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        units = hp.Int('units', min_value=32, max_value=512, step=32)
        dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
        l2_lambda = hp.Choice('l2_lambda', values=[1e-3, 1e-4, 1e-5])
        num_dropout_layers = hp.Int('num_dropout_layers', min_value=1, max_value=5)

        # 5 - Loading the base model
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=shape)

        # 6 - Layer adjustments for fine-tuning
        # Fine-tuning strategy: Unfreezing layer by block name, here 'block5'
        for layer in base_model.layers:
            layer.trainable = 'block5' in layer.name

        # 7 - Adding Fully Connected Layers
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(units, activation=hidden_layers_activation, kernel_regularizer=l2(l2_lambda))(x)

        # 8 - Adding Dropout Layers
        for _ in range(num_dropout_layers):
            x = Dropout(dropout_rate)(x)

        # 9 - Output Layer
        output = Dense(actual_num_classes, activation=output_activation)(x)

        # 10 - Creating the model
        model = Model(inputs=base_model.input, outputs=output)

        # 11 - Compiling the model with hyperparameters and loss functions
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss_function, metrics=['accuracy'])
        return model
    except KeyError as e:
        logger.error(f"KeyError: Missing key in ml_hp: {e}")
        raise KeyError(f"Missing key in ml_hp: {e}") from None
    except ValueError as e:
        logger.error(f"ValueError: {e}")
        raise ValueError(e) from None
    
    
######
## Model : VGG19
######

def build_model_vgg19(hp, ml_hp,actual_num_classes):
    """
    Constructs a VGG19 architecture model for image classification, compatible with both binary and multi-class scenarios.
    This method utilizes hyperparameters suitable for Keras Tuner optimization and general configuration parameters from a configuration object,
    including:
    - 'archi': defining the architecture to be constructed,
    - 'img_size': defining the size of the image for model input,
    - 'img_dim': color dimensions (RGB 3 or Grayscale 1),
    - etc.

    :param hp: HyperParameters object from Keras Tuner for model hyperparameter optimization.
    :param ml_hp: MLFlow HyperParameters object containing configuration details.
    :param actual_num_classes: Number of classes is dynamic therefore it is provided after the subfolders processing
    
    :return: A compiled Keras model optimized with hyperparameters.

    :raises KeyError: If essential keys are missing from the ml_hp dictionary.
    :raises ValueError: If the configuration values in ml_hp are invalid or unsupported, such as an unsupported number of classes.
    """
    logger.debug("---------build_model_vgg19------------")

    try:
        # 1a - Retrieving MLFlow hyperparameters
        archi = ml_hp["archi"]
        img_size = ml_hp["img_size"]
        img_dim = ml_hp["img_dim"]
        num_classes = ml_hp["num_classes"]
        
        # 1b - Optional parameters will have a default value
        hidden_layers_activation = ml_hp.get("hl_activation", "relu")

        # 1c - Initialization based on mlflow_archive hyperparameters
        shape = (img_size, img_size, img_dim)

        # 2 - Classification specifics: loss function
        if num_classes == 1:
            logger.debug("--- BINARY CLASSIFICATION ------")
            loss_function = 'binary_crossentropy'
            output_activation = "sigmoid"
        else: #0 refers to dynamic multiple classification , therefore any num_classes other than 1 will be non binary
            logger.debug("--- MULTICLASS CLASSIFICATION ------")
            loss_function = 'sparse_categorical_crossentropy'
            output_activation = "softmax"
        
        logger.debug("--- Hyperparameters ---")
        # 3 - Logging hyperparameters
        for param in ['archi', 'img_size', 'img_dim', 'shape', 'num_classes', 'hidden_layers_activation', 'loss_function', 'output_activation']:
            logger.debug(f"{param} = {locals()[param]}")

        logger.debug(f"actual_num_classes = {actual_num_classes}")
        
        # 4 - Defining hyperparameters to be optimized with Keras Tuner
        learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        units = hp.Int('units', min_value=32, max_value=512, step=32)
        dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
        l2_lambda = hp.Choice('l2_lambda', values=[1e-3, 1e-4, 1e-5])
        num_dropout_layers = hp.Int('num_dropout_layers', min_value=1, max_value=5)

        # 5 - Loading the base model
        base_model = VGG19(weights='imagenet', include_top=False, input_shape=shape)

        # 6 - Layer adjustments for fine-tuning
        # Fine-tuning strategy: Unfreezing layer by block name, here 'block5'
        for layer in base_model.layers:
            layer.trainable = 'block5' in layer.name

        # 7 - Adding Fully Connected Layers
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(units, activation=hidden_layers_activation, kernel_regularizer=l2(l2_lambda))(x)

        # 8 - Adding Dropout Layers
        for _ in range(num_dropout_layers):
            x = Dropout(dropout_rate)(x)

        # 9 - Output Layer
        output = Dense(actual_num_classes, activation=output_activation)(x)

        # 10 - Finalizing model construction
        model = Model(inputs=base_model.input, outputs=output)

        # 11 - Compiling the model with hyperparameters and classification-specific loss functions
        model.compile(optimizer=SGD(learning_rate=learning_rate), loss=loss_function, metrics=['accuracy'])
        return model
    except KeyError as e:
        logger.error(f"KeyError: Missing key in ml_hp: {e}")
        raise KeyError(f"Missing key in ml_hp: {e}") from None
    except ValueError as e:
        logger.error(f"ValueError: {e}")
        raise ValueError(e) from None
  
    
######
## Model : ResNet50
######

def build_model_resnet50(hp, ml_hp,actual_num_classes):
    """
    Constructs a ResNet50 architecture model for image classification, supporting both binary and multi-class scenarios.
    Utilizes hyperparameters for Keras Tuner optimization and general configuration parameters from a configuration object.
    This configuration includes details such as:
    - 'archi': defining the architecture to be constructed,
    - 'img_size': defining the size of the image for model input,
    - 'img_dim': color dimensions (RGB 3 or Grayscale 1),
    - etc.

    :param hp: HyperParameters object from Keras Tuner for model hyperparameter optimization.
    :param ml_hp: MLFlow HyperParameters object containing configuration details.
    :param actual_num_classes: Number of classes is dynamic therefore it is provided after the subfolders processing
    
    :return: A compiled Keras model optimized with hyperparameters.

    :raises KeyError: If essential keys are missing from the ml_hp dictionary.
    :raises ValueError: If the configuration values in ml_hp are invalid or unsupported, such as an unsupported number of classes.
    """
    logger.debug("---------build_model_resnet50------------")
    try:
        # 1a - Retrieving MLFlow hyperparameters
        archi = ml_hp["archi"]
        img_size = ml_hp["img_size"]
        img_dim = ml_hp["img_dim"]
        num_classes = ml_hp["num_classes"]
        
        # 1b - Optional parameters will have a default value
        hidden_layers_activation = ml_hp.get("hl_activation", "relu")

        # 1c - Initialization based on mlflow_archive hyperparameters
        shape = (img_size, img_size, img_dim)

        # 2 - Classification specifics: loss function
        if num_classes == 1:
            logger.debug("--- BINARY CLASSIFICATION ------")
            loss_function = 'binary_crossentropy'
            output_activation = "sigmoid"
        else: #0 refers to dynamic multiple classification , therefore any num_classes other than 1 will be non binary
            logger.debug("--- MULTICLASS CLASSIFICATION ------")
            loss_function = 'sparse_categorical_crossentropy'
            output_activation = "softmax"
        
        logger.debug("--- Hyperparameters ---")
        # 3 - Logging hyperparameters
        for param in ['archi', 'img_size', 'img_dim', 'shape', 'num_classes', 'hidden_layers_activation', 'loss_function', 'output_activation']:
            logger.debug(f"{param} = {locals()[param]}")
            
        logger.debug(f"actual_num_classes = {actual_num_classes}")
        
        # 4 - Defining hyperparameters to be optimized with Keras Tuner
        learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        units = hp.Int('units', min_value=32, max_value=512, step=32)
        dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
        l2_lambda = hp.Choice('l2_lambda', values=[1e-3, 1e-4, 1e-5])
        num_dropout_layers = hp.Int('num_dropout_layers', min_value=1, max_value=5)

        # 5 - Loading the base model
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=shape)

        # 6 - Layer adjustments for fine-tuning
        # Fine-tuning strategy: Unfreezing from a certain numver of layers
        base_model.trainable = True
        fine_tune_at = 100  # Example value, adjust based on your needs
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

        # 7 - Adding Fully Connected Layers
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(units, activation=hidden_layers_activation, kernel_regularizer=l2(l2_lambda))(x)

        # 8 - Adding Dropout Layers
        for _ in range(num_dropout_layers):
            x = Dropout(dropout_rate)(x)

        # 9 - Output Layer
        output = Dense(actual_num_classes, activation=output_activation)(x)

        # 10 - Finalizing model construction
        model = Model(inputs=base_model.input, outputs=output)

        # 11 - Compiling the model with hyperparameters and classification-specific loss functions
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss_function, metrics=['accuracy'])
        
        return model
    except KeyError as e:
        logger.error(f"KeyError: Missing key in ml_hp: {e}")
        raise KeyError(f"Missing key in ml_hp: {e}") from None
    except ValueError as e:
        logger.error(f"ValueError: {e}")
        raise ValueError(e) from None
    

######
## Model : EfficientNetB0
######
def build_model_efficientnetb0(hp, ml_hp,actual_num_classes):
    """
    Constructs an EfficientNetB0 architecture model for image classification, compatible with binary and multi-class scenarios.
    Utilizes both Keras Tuner optimization hyperparameters and general configuration parameters from a configuration object.
    This configuration includes:
    - 'archi': defining the architecture to be built,
    - 'img_size': defining the image size for model input,
    - 'img_dim': color dimensions (RGB 3 or Grayscale 1),
    - etc.

    :param hp: HyperParameters object from Keras Tuner for model hyperparameter optimization.
    :param ml_hp: MLFlow HyperParameters object containing configuration details.
    :param actual_num_classes: Number of classes is dynamic therefore it is provided after the subfolders processing
    
    :return: A compiled Keras model optimized with hyperparameters.

    :raises KeyError: If essential keys are missing from the ml_hp dictionary.
    :raises ValueError: If configuration values in ml_hp are invalid or unsupported, such as an unsupported number of classes.
    """
    logger.debug("---------build_model_efficientnetb0------------")

    try:
        # 1a - Retrieving MLFlow hyperparameters
        archi = ml_hp["archi"]
        img_size = ml_hp["img_size"]
        img_dim = ml_hp["img_dim"]
        num_classes = ml_hp["num_classes"]

        # 1b - Optional parameters will have a default value
        hidden_layers_activation = ml_hp.get("hl_activation", "relu")

        # 1c - Initialization based on mlflow_archive hyperparameters
        shape = (img_size, img_size, img_dim)

        # 2 - Classification specifics: loss function
        if num_classes == 1:
            logger.debug("--- BINARY CLASSIFICATION ------")
            loss_function = 'binary_crossentropy'
            output_activation = "sigmoid"
        else: #0 refers to dynamic multiple classification , therefore any num_classes other than 1 will be non binary
            logger.debug("--- MULTICLASS CLASSIFICATION ------")
            loss_function = 'sparse_categorical_crossentropy'
            output_activation = "softmax"
        
        logger.debug("--- Hyperparameters ---")
        # 3 - Logging hyperparameters
        for param in ['archi', 'img_size', 'img_dim', 'shape', 'num_classes', 'hidden_layers_activation', 'loss_function', 'output_activation']:
            logger.debug(f"{param} = {locals()[param]}")

        logger.debug(f"actual_num_classes = {actual_num_classes}")
        
        # 4 - Defining hyperparameters to be optimized with Keras Tuner
        learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')
        units = hp.Int('units', min_value=32, max_value=512, step=32)
        dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
        l2_lambda = hp.Choice('l2_lambda', values=[1e-3, 1e-4, 1e-5])
        num_dropout_layers = hp.Int('num_dropout_layers', min_value=1, max_value=5)
        dropout_connect_rate = hp.Float('dropout_connect_rate', min_value=0.2, max_value=0.4, step=0.1)

        logger.debug("--- Architecture-specific details ---")
        logger.debug(f"dropout_connect_rate = {dropout_connect_rate}")

        # 5 - Loading the base model
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=shape)

        # 6 - Layer adjustments for fine-tuning
        # Fine-tuning strategy: Unfreezing convolutional layers while keeping batch normalization layers frozen
        for layer in base_model.layers:
            layer.trainable = isinstance(layer, Conv2D)

        # 7 - Adding Fully Connected Layers
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(units, activation=hidden_layers_activation, kernel_regularizer=l2(l2_lambda))(x)
        x = BatchNormalization()(x)

        # 8 - Adding Dropout Layers
        for _ in range(num_dropout_layers):
            x = Dropout(dropout_rate)(x)

        # 9 - Output Layer
        output = Dense(actual_num_classes, activation=output_activation)(x)

        # 10 - Finalizing model construction
        model = Model(inputs=base_model.input, outputs=output)

        # 11 - Compiling the model with hyperparameters and classification-specific loss functions
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss_function, metrics=['accuracy'])
        return model
    except KeyError as e:
        logger.error(f"KeyError: Missing key in ml_hp: {e}")
        raise KeyError(f"Missing key in ml_hp: {e}") from None
    except ValueError as e:
        logger.error(f"ValueError: {e}")
        raise ValueError(e) from None
