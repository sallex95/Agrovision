import torch
import numpy as np
import json
import pandas as pd
import sys, os


#######################
### Hyperparameters ###
#######################
def run_config():
    # bool  
    server = True  # Flag that indicates whether to use server or local machine

    load_numpy = True  # Flag that indicates what type of data we use as input

    use_mask = True  # Flag that indicates are we masking data with boundery and valid masks

    use_weights = False  # Flag that indicates whether to use class weights when initializing loss function

    do_testing = True  # Flag that indicates whether to do testing after the training is done

    count_logs_flag = False  # Flag that indicates whether to plot number of classes and pixels in tensorboard, classwise and batch-wise

    zscore = 0  # 0-Instance, 1 - Zscore 2- /255 # Flag that indicates whether we use zscore normalization in preprocessing

    binary = False  # Flag that indicates whether we do binary semantic segmentation

    freeze_backbone_weights = False  # Flag that indicates whether to freeze backbone weights

    early_stop = False  # Initial early stopping flag

    save_best_model = True  # Initial "best model" flag, indicates whether to save model in corresponding epoch

    # strings

    scheduler_lr = 'multiplicative'  # Indicates which scheduler to use

    dataset = "mini"  # "mini" or "full" Indicates whether we want to use full dataset for training, validation and testing or decimated version
    year = "2021"  # "2020" or "2021"
    if server:  # Depending on server flag, we use different device settings:
        device = "cuda"  # if server is True, that is, if we are using server machine, device will be set as "cuda"
    elif server == False:
        device = "cpu"  # else if server is False and we are using local machine or server access node, device will be set as "cpu"

    if year == "2020":
        if dataset == 'mini' and server:  # paths to the datasets
            numpy_path = r"/storage/home/antica/DATASETS/Agriculture_Vision_2020_mini/numpy_new"
            numpy_valid_path = r"/storage/home/antica/DATASETS/Agriculture_Vision_2020_mini/numpy_new_valid"
            numpy_test_path = r"/storage/home/antica/DATASETS/Agriculture_Vision_2020_mini/numpy_new_test"
        elif dataset == 'full' and server:
            numpy_path = r"/storage/home/antica/DATASETS/Agriculture_Vision_2020/train/numpy_new"  # full train dataset
            numpy_valid_path = r"/storage/home/antica/DATASETS/Agriculture_Vision_2020/val/numpy_new_valid"  # full validation dataset
            numpy_test_path = r"/storage/home/antica/DATASETS/Agriculture_Vision_2020/val/numpy_new_test"  # full test dataset

        classes_labels = ['foreground']  # Classes that we are trying to detect. For BCE, background class is rejected
        classes_labels2 = ['background', 'foreground']  # classes_labels + background

        multiclasses_labels = ['cloud_shadow', 'double_plant', 'planter_skip', 'standing_water', 'waterway',
                               'weed_cluster']
        multiclasses_labels2 = ['background', 'cloud_shadow', 'double_plant', 'planter_skip', 'standing_water',
                                'waterway', 'weed_cluster']

    elif year == "2021":
        if dataset == 'mini' and server:  # paths to the datasets
            numpy_path = r"/home/antica/DATASETS/Agriculture_Vision_2021_mini/numpy_new"
            numpy_valid_path = r"/home/antica/DATASETS/Agriculture_Vision_2021_mini/numpy_new_valid"
            numpy_test_path = r"/home/antica/DATASETS/Agriculture_Vision_2021_mini/numpy_new_test"

        elif dataset == 'full' and server:
            numpy_path = r"/home/antica/DATASETS/Agriculture_Vision_2021/train/numpy_new"
            numpy_valid_path = r"/home/antica/DATASETS/Agriculture_Vision_2021/val/numpy_new_valid"
            numpy_test_path = r"/home/antica/DATASETS/Agriculture_Vision_2021/val/numpy_new_test"

        elif dataset == 'mini' and server == False:  # paths to the datasets
            numpy_path = r"D:\BIOSENS_programi\DATASETS\Agriculture_Vision_2021_mini_NEW/numpy_new"
            numpy_valid_path = r"D:\BIOSENS_programi\DATASETS\Agriculture_Vision_2021_mini_NEW/numpy_new_valid"
            numpy_test_path = r"D:\BIOSENS_programi\DATASETS\Agriculture_Vision_2021_mini_NEW/numpy_new_test"
        elif dataset == 'full' and server == False:
            numpy_path = r"D:\BIOSENS_programi\DATASETS\Agriculture_Vision_2021/train/numpy_new"  # full train dataset
            numpy_valid_path = r"D:\BIOSENS_programi\DATASETS\Agriculture_Vision_2021/val/numpy_new_valid"  # full validation dataset
            numpy_test_path = r"D:\BIOSENS_programi\DATASETS\Agriculture_Vision_2021/val/numpy_new_test"  # full test dataset
        else:
            print("Error: wrong dataset dimension")
            sys.exit(0)
        classes_labels = ['foreground']  # Classes that we are trying to detect. For BCE, background class is rejected
        classes_labels2 = ['background', 'foreground']  # classes_labels + background

        multiclasses_labels = ["double_plant", "drydown", "endrow", "nutrient_deficiency", "planter_skip",
                               "storm_damage", "water", "waterway", "weed_cluster"]
        multiclasses_labels2 = ["background", "double_plant", "drydown", "endrow", "nutrient_deficiency",
                                "planter_skip", "storm_damage", "water", "waterway", "weed_cluster"]

    else:
        print("Error: Wrong dataset year")
        sys.exit(0)
    if year == "2020":
        tb_img_list = [  ########## Train samples which we want to visualize in tensorboard during the training
            'MWK791B1K_648-7567-1160-8079', '1DJX4RH9N_6972-614-7484-1126',
            '1DJX4RH9N_768-380-1280-892', '2CVV62WDV_7422-650-7934-1162',
            '2CVV62WDV_9600-696-10112-1208', '2FPNYIQY1_944-1011-1456-1523',
            '2J7111V4A_1162-3466-1674-3978', '2HFHD3N34_658-426-1170-938',
            ########## Validation samples which we want to visualize in tensorboard during the training
            '18X67NLZ2_1219-1603-1731-2115', 'HLXDJEHJT_766-3006-1278-3518'
            , 'A88IZM6PD_817-8408-1329-8920',
            '6ZNXWNPB7_597-3417-1109-3929', 'G9N8LNVPV_625-1306-1137-1818']
    elif year == "2021":
        tb_img_list = [  ########## Train samples which we want to visualize in tensorboard during the training
            'MN7WEQIE8_4629-2036-5141-2548', '1DEGPM1UA_2649-2251-3161-2763',
            '1DQUZKY8G_2170-927-2682-1439', '2CRC6AJW8_4068-12144-4580-12656',
            '2CRC6AJW8_6116-10608-6628-11120', '2FPNYIQY1_4844-1462-5356-1974.npy',
            '2L3ZGKHJL_2632-3137-3144-3649', '3E3FNZJIG_5474-5851-5986-6363',
            ########## Validation samples which we want to visualize in tensorboard during the training
            'MQ894WR89_433-7960-945-8472', 'H3AZ2WRAL_3187-2388-3699-2900'
            , 'ABNF9NY9E_6519-2905-7031-3417',
            'VCJUGJWMT_4241-9561-4753-10073', 'YYFXA7BRW_1392-7023-1904-7535']
    else:
        print("Error: wrong year parameter or there is no dataset for that year")
    ###############
    loss_type = 'ce'  # Indicates loss type we want to use: bce, ce, ce_1 BITNO U ZAVISNOSTI KO KORISTI KOD, SALE KORISTI CE, BCE DIMITRIJE, NINA I MARKO
    ###############
    if loss_type == 'bce':
        background_flag = False
    elif loss_type == 'ce':
        background_flag = True
    else:
        sys.exit(0)
    if background_flag:
        multiclasses_labels = multiclasses_labels2
    if not (binary):
        classes_labels = multiclasses_labels
        classes_labels2 = multiclasses_labels2

    net_type = "UNet16_2"  # UNet16,UNet32,UNet0  # Indicates Architecture that we want to use: UNet3, Unet_orig,...

    img_data_format = '.npy'  # Indicated the type of data we use as input

    # Integer

    epochs = 50  # Number of epochs  we want our model training

    set_random_seed = 15  # Setting random seed for torch random generator

    batch_size = 4  # Size of batch during the training,validation and testing

    shuffle_state = 1  # Random shuffle seeed

    GPU_list = [0]  # Indices of GPUs that we want to allocate and use during the training

    weight_decay = 0  # L2 penalty

    optimizer_patience = 10  # When using ReduceLR scheduler: Indicates number of epochs without loss minimum decrease,
    # after which the learning rate will be multiplied by a lambda parameter

    save_checkpoint_freq = 1  # Initial frequency parameter for model saving

    if save_checkpoint_freq < 10:  # Further updating saving frequency parameter
        save_checkpoint_freq = 0
    if save_checkpoint_freq >= 100:
        save_checkpoint_freq = int(epochs / 10)  # frequency of saving checkpoints in epochs
    if save_checkpoint_freq == 0:
        save_checkpoint_freq = 1000

    num_channels = 4  # Number of input channels: Red, Green, Blue, NIR

    # if binary:
    #     num_channels_lab = 1 # 2 # Number of output channels and label channels: num_channels_lab = 1 for binary semantic segmentation: BCE loss without background class
    # else:
    #     num_channels_lab = 6 # 7                                       # num_channels_lab = 2 for binary semantic segmentation: CE loss with background class
    # num channels_lab = 6 for multiclass semantic segmentation: BCE loss without background class
    # num channels_lab = 7 for multiclass semantic segmentation: CE loss with background class
    num_channels_lab = len(classes_labels)

    img_h = 512  # Input image Height

    img_w = 512  # Input image Weight

    img_size = [img_h, img_w]  # Input channel and label shape

    #######################
    ### loss containers ###
    #######################
    train_losses = []  # Container in which we store losses for each training batch

    validation_losses = []  # Container in which we store losses for each validation batch

    test_losses = []
    #######################

    ################
    ### Counters ###
    ################

    epoch_model_last_save = 0  # Counter that counts number of epochs since the most recent model saving

    count_train = 0  # Counter that counts number of training batches

    count_val = 0  # Counter that counts number of validation batches

    count_train_tb = 0

    ######################
    ### early stopping ###
    ######################

    es_min = 1e9  # Initial minimum parameter for early stopping

    es_epoch_count = 0  # Epoch counter for early stopping

    es_check = 5  # Number of epochs after wich we dont have new minimal validation loss and after wich we apply early stopping

    # Dictionary creation

    dictionary = {

        "save_checkpoint_freq": save_checkpoint_freq,
        "set_random_seed": set_random_seed,
        "epochs": epochs,
        "use_mask": use_mask,
        "count_train": count_train,
        "count_train_tb": count_train_tb,
        "count_val": count_val,
        "GPU_list": GPU_list,
        "shuffle_state": shuffle_state,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "es_min": es_min,
        "es_check": es_check,
        "es_epoch_count": es_epoch_count,
        "optimizer_patience": optimizer_patience,
        "num_channels": num_channels,
        "num_channels_lab": num_channels_lab,
        "img_h": img_h,
        "img_w": img_w,
        "img_size": img_size,
        "epoch_model_last_save": epoch_model_last_save,
        "scheduler_lr": scheduler_lr,
        "classes_labels": classes_labels,
        "classes_labels2": classes_labels2,
        "dataset": dataset,
        "tb_img_list": tb_img_list,
        "server": server,
        "img_data_format": img_data_format,
        "net_type": net_type,
        "device": device,
        "loss_type": loss_type,
        "numpy_path": numpy_path,
        "numpy_valid_path": numpy_valid_path,
        "numpy_test_path": numpy_test_path,
        "load_numpy": load_numpy,
        "use_weights": use_weights,  # for CE loss
        "do_testing": do_testing,
        "count_logs_flag": count_logs_flag,
        "freeze_backbone_weights": freeze_backbone_weights,
        "zscore": zscore,
        "binary": binary,
        "early_stop_flag": early_stop,
        "save_best_model": save_best_model,
        "train_losses": train_losses,
        "validation_losses": validation_losses,
        "test_losses": test_losses,
        "background_flag": background_flag,
        "year": year,
    }

    # Serializing json
    json_object = json.dumps(dictionary, indent=4)

    # Writing to sample.json
    # base_folder_path = os.getcwd()
    base_folder_path = os.path.dirname(__file__)
    base_folder_path = base_folder_path.replace("\\", "/")
    with open(base_folder_path + "/config.json", "w") as outfile:
        outfile.write(json_object)

    legend_path = r"/storage/home/antica/PYTHON_projekti/Agrovision_Torch/Legend_Classes.png"
    background_names = []
    background_area = []
    foreground_names = []
    foreground_area = []
    cloud_shadow_names = [];
    double_plant_names = [];
    planter_skip_names = [];
    standing_water_names = [];
    waterway_names = [];
    weed_cluster_names = []
    drydown_names = [];
    endrow_names = [];
    nutrient_deficiency_names = [];
    storm_damage_names = [];
    water_names = [];

    cloud_shadow_area = [];
    double_plant_area = [];
    planter_skip_area = [];
    standing_water_area = [];
    waterway_area = [];
    weed_cluster_area = []
    drydown_area = [];
    endrow_area = [];
    nutrient_deficiency_area = [];
    storm_damage_area = [];
    water_area = [];
    test_losses = []
    # 2020
    iou_per_test_image_bg = [];
    iou_per_test_image_cs = [];
    iou_per_test_image_dp = [];
    iou_per_test_image_ps = [];
    iou_per_test_image_sw = [];
    iou_per_test_image_ww = [];
    iou_per_test_image_wc = [];
    iou_per_test_image_fg = [];
    # 2021
    iou_per_test_image_dd = [];
    iou_per_test_image_er = [];
    iou_per_test_image_nd = [];
    iou_per_test_image_sd = [];
    iou_per_test_image_w = [];

    k_index = 1
    dictionary_test = {

        "legend_path": legend_path,
        "test_losses": test_losses,
        "background_names": background_names,
        "foreground_names": foreground_names,
        "cloud_shadow_names": cloud_shadow_names,
        "double_plant_names": double_plant_names,
        "planter_skip_names": planter_skip_names,
        "standing_water_names": standing_water_names,
        "waterway_names": waterway_names,
        "weed_cluster_names": weed_cluster_names,

        "drydown_names": drydown_names,
        "endrow_names": endrow_names,
        "nutrient_deficiency_names": nutrient_deficiency_names,
        "storm_damage_names": water_names,
        "water_names": water_names,

        "background_area": background_area,
        "foreground_area": foreground_area,
        "cloud_shadow_area": cloud_shadow_area,
        "double_plant_area": double_plant_area,
        "planter_skip_area": planter_skip_area,
        "standing_water_area": standing_water_area,
        "waterway_area": waterway_area,
        "weed_cluster_area": weed_cluster_area,

        "drydown_area": drydown_area,
        "endrow_area": endrow_area,
        "nutrient_deficiency_area": nutrient_deficiency_area,
        "storm_damage_area": storm_damage_area,
        "water_area": water_area,

        "iou_per_test_image_bg": iou_per_test_image_bg,
        "iou_per_test_image_cs": iou_per_test_image_cs,
        "iou_per_test_image_dp": iou_per_test_image_dp,
        "iou_per_test_image_ps": iou_per_test_image_ps,
        "iou_per_test_image_sw": iou_per_test_image_sw,
        "iou_per_test_image_ww": iou_per_test_image_ww,
        "iou_per_test_image_wc": iou_per_test_image_wc,
        "iou_per_test_image_fg": iou_per_test_image_fg,

        "iou_per_test_image_dd": iou_per_test_image_dd,
        "iou_per_test_image_er": iou_per_test_image_er,
        "iou_per_test_image_nd": iou_per_test_image_nd,
        "iou_per_test_image_sd": iou_per_test_image_sd,
        "iou_per_test_image_w": iou_per_test_image_w,

        "k_index": k_index,
        "binary": binary,
        "use_mask": use_mask,
        "dataset": dataset,
        "background_flag": background_flag,
        "year": year,
        "loss_type": loss_type,
    }
    # Serializing json 
    json_object = json.dumps(dictionary_test, indent=4)

    # Writing to sample.json
    with open(base_folder_path + "/config_test.json", "w") as outfile:
        outfile.write(json_object)
