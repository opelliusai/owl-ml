OWL ML - by OpelliusAI
=======================================================================

This repo is the first step of the OWL ML project, which aims to adapt MLFlow execution to any model architecture and dataset. 
Currently adapted to Computer Vision projects

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- Should be in your computer but not on Github (only in .gitignore)
    │   ├── processed      <- The processed data during the training and evaluation phase (models, training history, confusion matrix and classification reports, gradcam etc.)
    │   └── raw            <- The dataset used, organised by scope > Contains Examples of binary and multiple classification dataset from https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
    │
    ├── logs                <- Logs folder
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── config       <- Scripts to turn raw data into features for modeling
    │   │   ├── log_config.py
    │   │	└── run_config.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make predictions. 
    │   │	│ 			The content is fully customizable, however keep in mind that the keras tuner random search is the one called by run_mlflow.py
    │   │	│ 				If you want to call directly your build model functions
    │   │   │                 
    │   │   ├── run_mlflow.py 		<- The MAIN python file to execute when everything is setup
    │   │   ├── build_model.py 		<- Module used to find the best hyperparameters and build models depending on their architecture (VGG, EfficientNet etc.)
    │   │   ├── train_model.py		<- Module used to train the best_model and generate reports
    │   │	└── predict_model.py 	<- Module used for predictions and model evaluation for additional reports
    │   │
    │   ├── utils         <- Utilities 
    │   │   │                 
    │   │   ├── utils_runs.py						<- Useful functions to execute the main steps for each experiment type
    │   │   ├── utils_models.py 					<- Useful functions for model load, save etc.
    │   │   └── utils_gradcam.py 					<- (Next iteration) Useful Gradcam functions
    │   │

--------
by Opellius AI - <a href="https://github.com/opelliusai">Github </a>
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
