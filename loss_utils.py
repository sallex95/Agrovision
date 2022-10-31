import sys
import numpy as np
import torch.nn as nn
import torch


def loss_init(use_weights, loss_type, dataset, num_channels_lab, device, year):
    if use_weights:
        if year == "2020":
            if dataset == "full":
                if num_channels_lab == 7:
                    class_weights = np.load(
                        r'/home/stefanovicd/DeepSleep/agrovision/class_weights_full_multiclass_with_background.npy')
                    # class_weights = [ 0.29468473, 17.64593792, 144.45582386, 61.6571218 , 36.81921342, 49.89357621, 8.40231562]
                elif num_channels_lab == 6:
                    class_weights = np.load(
                        r'/home/stefanovicd/DeepSleep/agrovision/class_weights_full_multiclass_without_background.npy')
                    # class_weights = [17.64593792, 144.45582386, 61.6571218 , 36.81921342, 49.89357621, 8.40231562]
                elif num_channels_lab == 2:
                    class_weights = np.load(
                        r'/home/stefanovicd/DeepSleep/agrovision/class_weights_full_binary_with_background.npy')
                elif num_channels_lab == 1:
                    class_weights = np.load(
                        r'/home/stefanovicd/DeepSleep/agrovision/class_weights_full_binary_without_background.npy')
                    # class_weights = [2.19672858]
                else:
                    print("Error: wrong dataset")
                    sys.exit(0)
            elif dataset == "mini":
                if num_channels_lab == 7:
                    class_weights = np.load(
                        r'/home/stefanovicd/DeepSleep/agrovision/class_weights_mini_multiclass_with_background.npy')
                    # class_weights = [ 0.29468473, 17.64593792, 144.45582386, 61.6571218 , 36.81921342, 49.89357621, 8.40231562]
                elif num_channels_lab == 6:
                    class_weights = np.load(
                        r'/home/stefanovicd/DeepSleep/agrovision/class_weights_mini_multiclass_without_background.npy')
                    #  class_weights = [17.64593792, 144.45582386, 61.6571218 , 36.81921342, 49.89357621, 8.40231562]
                elif num_channels_lab == 2:
                    class_weights = np.load(
                        r'/home/stefanovicd/DeepSleep/agrovision/class_weights_mini_binary_with_background.npy')
                elif num_channels_lab == 1:
                    class_weights = np.load(
                        r'/home/stefanovicd/DeepSleep/agrovision/class_weights_mini_binary_with_background.npy')[1:]
                else:
                    print("Error: wrong dataset")
                    sys.exit(0)
            else:
                print("Error: wrong dataset")
                sys.exit(0)
        elif year == "2021":
            if dataset == "full":
                if loss_type == 'ce':
                    class_weights = np.load(
                        r'/home/stefanovicd/DeepSleep/agrovision/class_weights_Agriculture_Vision_2021.npy')
                    # class_weights = [1.11184401e-01, 3.52867505e+03, 6.79132825e+02, 3.33448320e+03,
                    #                   5.93360836e+02, 4.62366548e+04, 3.38008533e+03, 4.00859072e+03, 6.15548006e+02]
                elif loss_type == 'bce':
                    print("waiting for weights to be computed. To be fixed...")
                    sys.exit(0)
                    # class_weights = np.load(r'/home/stefanovicd/DeepSleep/agrovision/class_weights_full_multiclass_without_background.npy')
                    # class_weights = [17.64593792, 144.45582386, 61.6571218 , 36.81921342, 49.89357621, 8.40231562]
                # elif num_channels_lab == 2:
                #     class_weights = np.load(r'/home/stefanovicd/DeepSleep/agrovision/class_weights_full_binary_with_background.npy')
                # elif num_channels_lab == 1:
                #     class_weights = np.load(r'/home/stefanovicd/DeepSleep/agrovision/class_weights_full_binary_without_background.npy')
                #     # class_weights = [2.19672858]
                else:
                    print("Error: wrong dataset")
                    sys.exit(0)
            elif dataset == "mini":
                if num_channels_lab == 7:
                    class_weights = np.load(
                        r'/home/stefanovicd/DeepSleep/agrovision/class_weights_mini_multiclass_with_background.npy')
                    # class_weights = [ 0.29468473, 17.64593792, 144.45582386, 61.6571218 , 36.81921342, 49.89357621, 8.40231562]
                elif num_channels_lab == 6:
                    class_weights = np.load(
                        r'/home/stefanovicd/DeepSleep/agrovision/class_weights_mini_multiclass_without_background.npy')
                    #  class_weights = [17.64593792, 144.45582386, 61.6571218 , 36.81921342, 49.89357621, 8.40231562]
                # elif num_channels_lab == 2:
                #     class_weights = np.load(r'/home/stefanovicd/DeepSleep/agrovision/class_weights_mini_binary_with_background.npy')
                # elif num_channels_lab == 1:
                #     class_weights = np.load(r'/home/stefanovicd/DeepSleep/agrovision/class_weights_mini_binary_with_background.npy')[1:]
                else:
                    print("Error: wrong dataset")
                    sys.exit(0)
            else:
                print("Error: wrong dataset")
                sys.exit(0)
        else:
            print("Error: wrong year parameter or there is no dataset for specific year")
            sys.exit(0)

    if loss_type == "ce_1":
        if use_weights:
            class_weights = torch.tensor(class_weights, dtype=torch.float, device=device)
            criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="none")
        else:
            criterion = nn.CrossEntropyLoss(reduction="none")
        return criterion
    elif loss_type == "bce":
        if use_weights:
            class_weights = torch.tensor(class_weights, dtype=torch.float, device=device).reshape(1, num_channels_lab,
                                                                                                  1, 1)
            criterion_bce = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights, reduction="none")
        else:
            criterion_bce = torch.nn.BCEWithLogitsLoss(reduction="none")
        return criterion_bce
    elif loss_type == "ce":
        if use_weights:
            class_weights = torch.tensor(class_weights, dtype=torch.float, device=device)
            criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="none")
        else:
            criterion = nn.CrossEntropyLoss(reduction="none")
        return criterion


def loss_calc(classwise_loss, batch_names, z_class, img_count_index, loss_type, criterion, model_output, target_var,
              mask_train='None', num_channels_lab=2,
              use_mask=True):  ### num_channels_lab = 2 u slucaju kada imamo 2 klase, bg i fg, Za Saletov slucaj to ce biti 7
    ### Kada se koristi bce ili ce kod kog nemamo racunanje verovatnoca argument num_channels_lab nije potrebno
    if loss_type == "bce":  ### proslediti
        loss = criterion(model_output, target_var)
        if use_mask:
            loss = loss[mask_train.unsqueeze(1).repeat(1, num_channels_lab, 1, 1)]

        loss = loss.mean()
        return loss

    elif loss_type == 'ce_1':
        loss = criterion(model_output, torch.argmax(target_var, 1))
        if use_mask:
            loss = loss[mask_train]
        if use_mask:
            loss = torch.multiply(loss, mask_train[:, 0, :, :])
            loss = torch.multiply(loss, mask_train[:, 1, :, :])
        loss = loss.mean()
        return loss

    elif loss_type == 'ce':
        target_var_ce = torch.nan_to_num(torch.div(target_var, torch.repeat_interleave(
            torch.square(torch.sum(target_var, dim=1)).unsqueeze(dim=1), repeats=num_channels_lab, dim=1)))
        # maska = torch.multiply(mask_train[:, 1, :, :], mask_train[:, 0, :, :]).unsqueeze(dim=1)
        target_var_ce = torch.multiply(target_var_ce, mask_train.unsqueeze(dim=1))
        model_output = torch.multiply(model_output, mask_train.unsqueeze(dim=1))
        loss = criterion(model_output, target_var_ce)
        class_indices_id = 0
        for batch_im_id in range(len(batch_names)):
            loss_imwise = loss[batch_im_id, :, :]
            # for id_class_slika in range(torch.sum(z_class[batch_im_id].int())): # onoliko puta koliko imam klasa na slici da upisem vrednost losa na odgovarajuce indexe
            #     classwise_loss[img_count_index+batch_im_id,class_indices[class_indices_id+id_class_slika][1].int()] = loss_imwise
            # classwise_loss[img_count_index+batch_im_id][z_class[batch_im_id].byte()] = loss_imwise[target_var[batch_im_id,z_class[batch_im_id].byte(),:,:].bool()].mean()
            classwise_loss[img_count_index + batch_im_id][z_class[batch_im_id].byte()] = loss_imwise.mean()
            class_indices_id += torch.sum(mask_train[batch_im_id]).int()

        loss = loss.mean()
        return loss, classwise_loss