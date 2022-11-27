# Comparison of different approaches to Natural Language Generation
Project created for the Bachelor Thesis at University of Warsaw.

### Install
Run inside the repository directory:
```commandline
pip install -e .
```

### Train models
Run training based on the setup in ```cfg/train_config.yaml```:
```commandline
python nlg_train.py --config-file cfg/train_config.yaml 
```
Check ```nlg_analysis/cfg/__init__.py``` for possible changes in config file
like setting the output name for the trained model.

### Run analysis
Run analysis based on the setup in ```cfg/analysis_config.yaml```:
```commandline
python nlg_analysis.py --config-file cfg/analysis_config.yaml
```
Check ```nlg_analysis/cfg/__init__.py``` for possible changes in config file
like setting the path to the trained model checkpoint.

### Code maintenance
There is Black, iSort, Flake8 and pre-commit set up. After installation run:
```commandline
pre-commit install
```
Before committing any changes.