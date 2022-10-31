# AgroVisionUnetBS

config.py - configuration script with constants and variables needed during training and testing 

data_utils.py - script with functions for loading and preparing data

loss_utils.py - script with loss related functions

metrics_utils.py - script with metrics related functions

model_utils.py - script with model initialization, saving and testing functions

print_utils.py - script with print functions

tb_utils.py - script with TensorBoard functions

Unet_LtS.py - script with Unet architectures

Train_BGFG_BCE_with_weights.py - main script with training and testing pipeline

requirements.txt - environment libraries that are necessary


## Some important config params

In config.py majority of parameters are constants but some of them are needed to be changed in order to set appropriate configuration. Parameters are divided in groups based on the type of data:
  - boolean
  - strings
  - integer

There are also some containers that are used during testing, that are simply unnecessary in main scripts.
Some of the parameters that are changeable:
  - server : When True device is set to 'cuda' and training is running on GPU, if False device is set on 'cpu' 
  - zscore : When True zscore normalization is applied to data
  - binary : When True Background/Foreground training configuration is enabled 
  - early_stop : When True early stopping is enabled 
  - do_testing : When True, immediately after training, testing is done 
  - dataset : 'full' or 'mini' depending on what dataset size we want 
  - year : '2020' or '2021' depending on what year dataset we want 
  - loss_type : 'ce' or 'bce'
  - epochs : number of training epochs
  - k_index : number of k top and k worst images we want to show in TensorBoard 

### important: Change paths in config file for legend, datasets

## Training

1. Edit hyperparameters in config.py and then run that script to create json files
  - server : True
  - zscore : False
  - binary : False
  - epochs : 20
  - dataset : 'full' / 'mini'
  - year : '2020' / '2021'
  
 Sale - Cross Entropy configuration:
  - loss_type : 'ce'
  
 Dimitrije (Nina, Marko) - Binary Cross Entropy configuration:
  - loss_type : 'bce'
    
Some of these hyperparameters enable new parameters that are defined in config.py and used in training and testing. Some of them are redundant, but they'll be     fixed after code review.
As a result of running config.py script, config.json and config_test.json files are created and those are further used in training and testing.
  
2. Run Train_BGFG_BCE_with_wieghts.py for training and testing

Parameters that can be changed:
  - lr : learning rate
  - lambda_parameter : denotes parameter that will multiplicate learning rate when lr scheduler is triggered
  - stepovi_arr : denotes lr scheduler step
  - param_ponovljivosti (optional, default = 1) : defines number of training with the same hyperparameters when testing repeatability.
 
## Testing 

To run testing after model training:
config.py > do_testing = True 

Separate script for testing - In progress...

