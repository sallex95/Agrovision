import matplotlib.pyplot as plt
import torch.utils.data.dataloader
from torch.utils.tensorboard import SummaryWriter
import random
from torchsummary import summary
import os, sys
from print_utils import *
from data_utils import *
from loss_utils import *
from model_utils import *
from tb_utils import *
from metrics_utils import *
from config import *
import torch
import numpy as np
import json
import pandas as pd
from measure import *
import gc


# from focal_loss import FocalLoss2

def upisivanje(ispis, ime_foldera):
    fff = open(ime_foldera, "a")
    fff.write(str(ispis) + "\n")
    fff.close()


def set_seed(seed):
    torch_manual_seed = torch.manual_seed(seed)
    torch_manual_seed_cuda = torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)
    torch.use_deterministic_algorithms(False)
    torch.random.seed()
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    return torch_manual_seed, torch_manual_seed_cuda


def main(lambda_parametri, stepovi, lr, p_index,jupyter=False):
    if jupyter==False:
        run_config()
        
    tmp = get_args('train')
    globals().update(tmp)
    base_folder_path = os.getcwd()
    base_folder_path = base_folder_path.replace("\\", "/")

    ime_foldera_za_upis, logs_path, save_model_path = pretraining_prints(p_index, lr, stepovi, lambda_parametri,
                                                                         batch_size, loss_type, net_type,jupyter)

    ####################
    ### data loading ###
    ####################

    train_loader, valid_loader, test_loader = data_loading(ime_foldera_za_upis, numpy_path, numpy_valid_path, binary,
                                                           background_flag)

    ####################
    after_data_loading_prints(lr, ime_foldera_za_upis, train_loader, valid_loader, test_loader)
    ####################

    torch_manual_seed, torch_manual_seed_cuda = set_seed(set_random_seed)

    tb = SummaryWriter(log_dir=logs_path)
    ############################
    ### model initialization ###
    ############################

    segmentation_net = model_init(num_channels, num_channels_lab, img_h, img_w, zscore, net_type, device, server,
                                  GPU_list,jupyter)
    if server==True and jupyter==False:
        print(summary(segmentation_net,(4,512,512)))
    ############################
    ### model initialization ###
    ############################

    optimizer, scheduler = optimizer_init(segmentation_net, lr, weight_decay, scheduler_lr, lambda_parametri,
                                          optimizer_patience)

    ############################
    ### Loss initialization ###
    ############################

    criterion = loss_init(use_weights, loss_type, dataset, num_channels_lab, device, year)

    if server and jupyter==False:
        start_train = torch.cuda.Event(enable_timing=True)
        start_val = torch.cuda.Event(enable_timing=True)
        start_test = torch.cuda.Event(enable_timing=True)
        end_train = torch.cuda.Event(enable_timing=True)
        end_val = torch.cuda.Event(enable_timing=True)
        end_test = torch.cuda.Event(enable_timing=True)

    # Brojanje Iteracija
    global count_train
    global count_val
    global es_min
    global epoch_model_last_save
    epoch_list = np.zeros([epochs])
    all_train_losses = torch.zeros([epochs])
    all_validation_losses = torch.zeros([epochs])
    all_test_losses = torch.zeros([epochs])
    all_lr = torch.zeros([epochs])
    val_loss_es = torch.zeros(epochs)

    for epoch in range(epochs):

        train_part = "Train"
        segmentation_net.train(mode=True)
        print("Epoch %d: Train[" % epoch, end="")

        if server and jupyter==False:
            start_train.record()
            torch.cuda.empty_cache()

        index_start = 0

        batch_iou = torch.zeros(size=(len(train_loader.dataset.img_names), num_channels_lab * 4), device=device,
                                dtype=torch.float32)
        prec_rec = torch.zeros(size=(len(train_loader.dataset.img_names), num_channels_lab * 2)).float().to(device)
        classwise_loss = torch.zeros(size=(len(train_loader.dataset.img_names), num_channels_lab)).float().to(device)
        img_count_index = 0
        if loss_type == 'bce':
            batch_iou_bg = torch.zeros(size=(len(train_loader.dataset.img_names), 2), device=device,
                                       dtype=torch.float32)

        for input_var, target_var, batch_names_train, mask_train, z_class in train_loader:

            set_zero_grad(segmentation_net)

            model_output = segmentation_net.forward(input_var)
            mask_train = torch.logical_and(mask_train[:, 0, :, :], mask_train[:, 1, :, :])
            loss, classwise_loss = loss_calc(classwise_loss, batch_names_train, z_class, img_count_index, loss_type,
                                             criterion, model_output, target_var, mask_train, num_channels_lab,
                                             use_mask)
            loss.backward()

            optimizer.step()  # mnozi sa grad i menja weightove

            train_losses.append(loss.data)
            img_count_index += len(batch_names_train)

            # acc, acc_cls, mean_iu, iu, f1 = evaluate(model_output, target_var, num_channels_lab)

            ######## update!!!!

            index_end = index_start + len(batch_names_train)
            if loss_type == 'bce':
                batch_iou[index_start:index_end, :], batch_iou_bg[index_start:index_end] = calc_metrics_pix(
                    model_output, target_var, mask_train, num_channels_lab, device, use_mask, loss_type)
            elif loss_type == 'ce':
                batch_iou[index_start:index_end, :], prec_rec[index_start:index_end, :] = calc_metrics_pix(model_output,
                                                                                                           target_var,
                                                                                                           mask_train,
                                                                                                           num_channels_lab,
                                                                                                           device,
                                                                                                           use_mask,
                                                                                                           loss_type)
            else:
                print("Error: unimplemented loss type")
                sys.exit(0)
            index_start += len(batch_names_train)
            ###########################################################
            ### iscrtavanje broja klasa i broja piskela i tako toga ###
            ###########################################################

            if epoch == 0 and count_logs_flag:
                count_freq = 2
                tb_num_pix_num_classes(tb, count_train, count_train_tb, count_freq, num_channels_lab, \
                                       batch_names_train, target_var, classes_labels, loss)

            #########################################################################
            ### Iscrtavanje trening uzoraka sa predefinisane liste u tensorboard  ###
            #########################################################################

            # tb_image_list_plotting(tb, tb_img_list, num_channels_lab, epoch, input_var, target_var,\
            #      mask_train, model_output, train_part, device, batch_names_train,use_mask,dataset,loss_type,year)

            count_train += 1
            print("*", end="")

        #########################################################
        ### Racunanje finalne metrike nad celim trening setom ###
        #########################################################
        if loss_type == 'bce':
            final_metric_calculation(tensorbd=tb, loss_type=loss_type, epoch=epoch, num_channels_lab=num_channels_lab,
                                     classes_labels=classes_labels, \
                                     batch_iou_bg=batch_iou_bg, batch_iou=batch_iou, train_part=train_part,
                                     ime_foldera_za_upis=ime_foldera_za_upis)
        elif loss_type == 'ce':
            final_metric_calculation(tensorbd=tb, loss_type=loss_type, epoch=epoch, num_channels_lab=num_channels_lab,
                                     classes_labels=classes_labels, \
                                     batch_iou=batch_iou, train_part=train_part,
                                     ime_foldera_za_upis=ime_foldera_za_upis, prec_rec=prec_rec,
                                     classwise_loss=classwise_loss)
        else:
            print("Error: Unimplemented loss type!")
            sys.exit(0)

        print("] ", end="")

        if server and jupyter==False:
            end_train.record()
            torch.cuda.synchronize()
            ispis = ("Time Elapsed For Train epoch " + str(epoch) + " " + str(
                start_train.elapsed_time(end_train) / 1000))
            print(ispis)
            upisivanje(ispis, ime_foldera_za_upis)

        all_train_losses[epoch] = (torch.mean(torch.tensor(train_losses, dtype=torch.float32)))
        all_lr[epoch] = (optimizer.param_groups[0]['lr'])

        if epoch != 0 and (epoch % stepovi) == 0:
            print("epoha: " + str(epoch) + " , uradjen step!")
            scheduler.step()

        del batch_iou, prec_rec, classwise_loss
        gc.collect()

        print(" Validation[", end="")
        del train_part

        if server and jupyter==False:
            torch.cuda.empty_cache()

        train_part = "Valid"
        segmentation_net.eval()

        with torch.no_grad():

            if server and jupyter==False:
                start_val.record()

            index_start = 0

            batch_iou = torch.zeros(size=(len(valid_loader.dataset.img_names), num_channels_lab * 4), device=device,
                                    dtype=torch.float32)
            prec_rec = torch.zeros(size=(len(valid_loader.dataset.img_names), num_channels_lab * 2), device=device,
                                   dtype=torch.float32)
            classwise_loss = torch.zeros(size=(len(valid_loader.dataset.img_names), num_channels_lab)).float().to(
                device)
            img_count_index = 0
            if loss_type == 'bce':
                batch_iou_bg = torch.zeros(size=(len(valid_loader.dataset.img_names), 2), device=device,
                                           dtype=torch.float32)

            for input_var, target_var, batch_names_valid, mask_val, z_class in valid_loader:

                model_output = segmentation_net.forward(input_var)
                mask_val = torch.logical_and(mask_val[:, 0, :, :], mask_val[:, 1, :, :])
                val_loss, classwise_loss = loss_calc(classwise_loss, batch_names_valid, z_class, img_count_index,
                                                     loss_type, criterion, model_output, target_var, mask_val,
                                                     num_channels_lab, use_mask)

                validation_losses.append(val_loss.data)
                img_count_index += len(batch_names_valid)

                # acc, acc_cls, mean_iu, iu, f1 = evaluate(model_output, target_var, num_channels_lab)

                index_end = index_start + len(batch_names_valid)
                if loss_type == 'bce':
                    batch_iou[index_start:index_end, :], batch_iou_bg[index_start:index_end] = calc_metrics_pix(
                        model_output, target_var, mask_val, num_channels_lab, device, use_mask, loss_type)
                elif loss_type == 'ce':
                    batch_iou[index_start:index_end, :], prec_rec[index_start:index_end, :] = calc_metrics_pix(
                        model_output, target_var, mask_val, num_channels_lab, device, use_mask, loss_type)
                else:
                    print("Error: unimplemented loss type")
                    sys.exit(0)

                index_start += len(batch_names_valid)

                ##############################################################################
                ### iscrtavanje validacionih uzoraka sa predefinisane liste u tensorboard  ###
                ##############################################################################

                # tb_image_list_plotting(tb, tb_img_list, num_channels_lab, epoch, input_var, target_var,\
                #      mask_val, model_output, train_part, device, batch_names_valid,use_mask,dataset,loss_type,year)

                count_val += 1
                print("*", end="")

            index_miou = 0

            ##############################################################
            ### Racunanje finalne metrike nad celim validacionim setom ###
            ##############################################################
            if loss_type == 'bce':
                final_metric_calculation(tensorbd=tb, loss_type=loss_type, epoch=epoch,
                                         num_channels_lab=num_channels_lab, classes_labels=classes_labels, \
                                         batch_iou_bg=batch_iou_bg, batch_iou=batch_iou, train_part=train_part,
                                         ime_foldera_za_upis=ime_foldera_za_upis)
            elif loss_type == 'ce':
                final_metric_calculation(tensorbd=tb, loss_type=loss_type, epoch=epoch,
                                         num_channels_lab=num_channels_lab, classes_labels=classes_labels, \
                                         batch_iou=batch_iou, train_part=train_part,
                                         ime_foldera_za_upis=ime_foldera_za_upis, prec_rec=prec_rec,
                                         classwise_loss=classwise_loss)
            else:
                print("Error: Unimplemented loss type!")
                sys.exit(0)
            print("] ", end="")
            if server and jupyter==False:
                end_val.record()
                torch.cuda.synchronize()
                ispis = ("Time Elapsed For Valid epoch " + str(epoch) + " " + str(
                    start_val.elapsed_time(end_val) / 1000))
                print(ispis)
                upisivanje(ispis, ime_foldera_za_upis)

            epoch_list[epoch] = epoch
            all_validation_losses[epoch] = (torch.mean(torch.tensor(validation_losses, dtype=torch.float32)))

            del batch_iou, prec_rec, classwise_loss
            gc.collect()

        print(" Testing[", end="")
        del train_part

        if server and jupyter==False: 
            torch.cuda.empty_cache()
        train_part = "Test_1"
        segmentation_net.eval()

        if server and jupyter==False:
            start_test.record()

        with torch.no_grad():

            index_start = 0

            batch_iou = torch.zeros(size=(len(test_loader.dataset.img_names), num_channels_lab * 4), device=device,
                                    dtype=torch.float32)
            prec_rec = torch.zeros(size=(len(test_loader.dataset.img_names), num_channels_lab * 2), device=device,
                                   dtype=torch.float32)
            classwise_loss = torch.zeros(size=(len(test_loader.dataset.img_names), num_channels_lab)).float().to(device)
            img_count_index = 0

            if loss_type == 'bce':
                batch_iou_bg = torch.zeros(size=(len(test_loader.dataset.img_names), 2), device=device,
                                           dtype=torch.float32)

            for input_var, target_var, batch_names_test, mask_val, z_class in test_loader:

                model_output = segmentation_net.forward(input_var)
                mask_val = torch.logical_and(mask_val[:, 0, :, :], mask_val[:, 1, :, :])
                test_loss, classwise_loss = loss_calc(classwise_loss, batch_names_test, z_class, img_count_index,
                                                      loss_type, criterion, model_output, target_var, mask_val,
                                                      num_channels_lab, use_mask)

                test_losses.append(test_loss.data)
                img_count_index += len(batch_names_test)

                # acc, acc_cls, mean_iu, iu, f1 = evaluate(model_output, target_var, num_channels_lab)

                index_end = index_start + len(batch_names_test)
                if loss_type == 'bce':
                    batch_iou[index_start:index_end, :], batch_iou_bg[index_start:index_end] = calc_metrics_pix(
                        model_output, target_var, mask_val, num_channels_lab, device, use_mask, loss_type)
                elif loss_type == 'ce':
                    batch_iou[index_start:index_end, :], prec_rec[index_start:index_end, :] = calc_metrics_pix(
                        model_output, target_var, mask_val,
                        num_channels_lab, device, use_mask, loss_type)
                else:
                    print("Error: unimplemented loss type")
                    sys.exit(0)

                index_start += len(batch_names_test)

                ##############################################################################
                ### iscrtavanje validacionih uzoraka sa predefinisane liste u tensorboard  ###
                ##############################################################################

                # tb_image_list_plotting(tb, tb_img_list, num_channels_lab, epoch, input_var, target_var, \
                #                        mask_val, model_output, train_part, device, batch_names_test, use_mask, dataset,
                #                        loss_type, year)

                count_val += 1
                print("*", end="")

            index_miou = 0

            ##############################################################
            ### Racunanje finalne metrike nad celim validacionim setom ###
            ##############################################################
            if loss_type == 'bce':
                final_metric_calculation(tensorbd=tb, loss_type=loss_type, epoch=epoch,
                                         num_channels_lab=num_channels_lab,
                                         classes_labels=classes_labels, \
                                         batch_iou_bg=batch_iou_bg, batch_iou=batch_iou, train_part=train_part,
                                         ime_foldera_za_upis=ime_foldera_za_upis)
            elif loss_type == 'ce':
                final_metric_calculation(tensorbd=tb, loss_type=loss_type, epoch=epoch,
                                         num_channels_lab=num_channels_lab,
                                         classes_labels=classes_labels, \
                                         batch_iou=batch_iou, train_part=train_part,
                                         ime_foldera_za_upis=ime_foldera_za_upis, prec_rec=prec_rec,
                                         classwise_loss=classwise_loss)
            else:
                print("Error: Unimplemented loss type!")
                sys.exit(0)

            print("] ", end="")

            if server and jupyter==False:
                end_test.record()
                torch.cuda.synchronize()
                print("] ", end="")
                ispis = ("Time Elapsed For Test epoch " + str(epoch) + " " + str(
                    start_test.elapsed_time(end_test) / 1000))
                print(ispis)
                upisivanje(ispis, ime_foldera_za_upis)

            epoch_list[epoch] = epoch
            all_test_losses[epoch] = (torch.mean(torch.tensor(test_losses, dtype=torch.float32)))

            del batch_iou, prec_rec, classwise_loss
            gc.collect()
            # print(acc)
            # print(acc_cls)
            # print(mean_iu)
            # print(iu)
            # print(f1)
        end_of_epoch_print(epoch, all_train_losses, all_validation_losses, all_test_losses, optimizer,
                           ime_foldera_za_upis)

        ##############################################################
        ### ispisivanje loss vrednosti za datu epohu u tensorboard ###
        ##############################################################

        tb_add_epoch_losses(tb, train_losses, validation_losses, test_losses, epoch)

        early_stop = early_stopping(epoch, val_loss_es, all_validation_losses, es_check, \
                                    segmentation_net, save_model_path, save_checkpoint_freq, ime_foldera_za_upis,
                                    es_min, epoch_model_last_save, es_epoch_count, save_best_model, early_stop_flag)
        if early_stop:
            break

    ##### upitno da li je neophodno #####
    if not (early_stop):
        fully_trained_model_saving(segmentation_net, save_model_path, epoch, ime_foldera_za_upis)
    #####################################
    if server and jupyter==False:
        torch.cuda.empty_cache()

    post_training_prints(ime_foldera_za_upis)

    np.save(logs_path + "/all_train_losses.npy", all_train_losses)
    np.save(logs_path + "/all_lr.npy", all_lr)
    np.save(logs_path + "/epoch_list.npy", epoch_list)

    ###############
    ### TESTING ###
    ###############

    if do_testing:
        criterion_1 = criterion

        test_loader = AgroVisionDataLoader(img_size, numpy_test_path, img_data_format, shuffle_state,
                                           batch_size, device, zscore, binary, dataset, background_flag)
        uporedna_tabela = pd.DataFrame()
        IOU = run_testing(segmentation_net, test_loader, ime_foldera_za_upis, logs_path, device, num_channels_lab,
                          classes_labels, classes_labels2,
                          criterion_1, loss_type, tb, zscore, server)

    end_prints(ime_foldera_za_upis)
    # return IOU
    # VISUALIZE TENSORBOARD
    ## tensorboard dev upload --logdir=logs/Train_Main_New
    # # tensorboard --logdir=Agrovision_Main/logs/Train_Main_New --host localhost
    # # tensorboard --logdir=logs/Train_Main_New --host localhost


if __name__ == '__main__':

    # lr = [1e-2,1e-3,1e-4]
    lr = [1e-3]
    lambda_parametar = [1]
    stepovi_arr = [5]
    # classes_labels2 = ['background','foreground']
    # classes_labels2 = ['background','cloud_shadow','double_plant','planter_skip','standing_water','waterway','weed_cluster']
    uporedna_tabela = pd.DataFrame()
    param_ponovljivosti = 1
    for p_index in range(
            param_ponovljivosti):  # petlja kojom ispitujemo ponovljivost istog eksperimenta, p_idex - broj trenutne iteracije
        for step_index in range(len(
                stepovi_arr)):  # petlja kojom ispitujemo kako se trening menja za razlicite korake promene lr-a, step = broj iteracija nakon kojeg ce se odraditi scheduler.step(loss)
            for lambd_index in range(len(
                    lambda_parametar)):  # petlja kojom ispitujemo kako se trening menja za razlicite labmda parametre kojim mnozimo lr kada dodje do ispunjavanja uslova za scheduler.step(loss)
                for lr_index in range(len(lr)):  # petlja kojom ispitujemo kako se trening menja za razlicite lr-ove
                    main(lambda_parametar[lambd_index], stepovi_arr[step_index], lr[lr_index], p_index)
        # uporedna_tabela['TestSet IoU Metric '+str(p_index)] = IOU

    # uporedna_tabela = uporedna_tabela.set_axis(classes_labels2).T                
    # uporedna_tabela.to_csv("Weighted BCE without background class, mini dataset, BGFG lr 1e-3 1 epochs.csv")
