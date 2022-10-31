import numpy as np
import torch
import os, sys, time
from data_utils import upisivanje


##### train mean iou classwise promeniti da bude opste, posto treba i za valid
def final_metric_calculation_test(tensorbd='None', loss_type='bce', epoch=0, num_channels_lab=1, classes_labels='None',
                             batch_iou_bg='None', batch_iou='None', train_part='Test', ime_foldera_za_upis='None'):
    index_miou = 0
    IOU = list()
    if loss_type == 'bce':
        iou_int_bg = batch_iou_bg[:, 0]
        iou_un_bg = batch_iou_bg[:, 1]
        iou_calc_bg = torch.div(torch.sum(iou_int_bg), torch.sum(iou_un_bg))
        if train_part == 'Test':
            ispis = train_part + " Mean IOU Classwise/" + "Background" + " " + str(
                np.round(iou_calc_bg.detach().cpu(), 4))
            IOU.append(np.round(iou_calc_bg.detach().cpu().numpy(), 4))
            print(ispis)
            upisivanje(ispis, ime_foldera_za_upis)
        else:
            ispis = train_part + " Mean IOU Classwise/" + "Background" + " "
            tensorbd.add_scalar(ispis, np.round(iou_calc_bg.detach().cpu(), 4), epoch)

    for klasa in range(num_channels_lab):

        iou_int = batch_iou[:, index_miou]
        iou_un = batch_iou[:, index_miou + 1]
        iou_un_1 = batch_iou[:, index_miou + 2]
        iou_un_2 = batch_iou[:, index_miou + 3]
        # iou_calc = torch.div(torch.sum(iou_int),(torch.sum(iou_un)-torch.sum(iou_int)))
        # iou_calc = torch.div(torch.sum(iou_int), torch.sum(iou_un))
        iou_calc = torch.div(torch.sum(iou_int),( torch.sum(iou_un_1)+ torch.sum(iou_un_2) -torch.sum(iou_int)))

        index_miou += 4
        if train_part == 'Test':
            ispis = train_part + "Mean IOU Classwise/" + classes_labels[klasa] + " " + str(
                np.round(iou_calc.detach().cpu(), 4))
            IOU.append(np.round(iou_calc.detach().cpu().numpy(), 4))
            print(ispis)
            upisivanje(ispis, ime_foldera_za_upis)
        else:
            ispis = train_part + " Mean IOU Classwise/" + classes_labels[klasa] + " "
            tensorbd.add_scalar(ispis, np.round(iou_calc.detach().cpu(), 4), epoch)

    return IOU


def final_metric_calculation(tensorbd='None', loss_type='bce', epoch=0, num_channels_lab=1, classes_labels='None',
                             batch_iou_bg='None', batch_iou='None', train_part='Test', ime_foldera_za_upis='None', prec_rec='None',classwise_loss='None'):
    index_miou = 0
    IOU = list()
    if loss_type == 'bce':
        iou_int_bg = batch_iou_bg[:, 0]
        iou_un_bg = batch_iou_bg[:, 1]
        iou_calc_bg = torch.div(torch.sum(iou_int_bg), torch.sum(iou_un_bg))
        if train_part == 'Test':
            ispis = train_part + " Mean IOU Classwise/" + "Background" + " " + str(
                np.round(iou_calc_bg.detach().cpu(), 4))
            IOU.append(np.round(iou_calc_bg.detach().cpu().numpy(), 4))
            print(ispis)
            upisivanje(ispis, ime_foldera_za_upis)
        else:
            ispis = train_part + " Mean IOU Classwise/" + "Background" + " "
            tensorbd.add_scalar(ispis, np.round(iou_calc_bg.detach().cpu(), 4), epoch)


    index_miou = 0
    index_prec = 0
    total_iou = np.zeros(num_channels_lab)
    for klasa in range(num_channels_lab):
        iou_int = batch_iou[:, index_miou]
        iou_un = batch_iou[:, index_miou + 1]
        iou_un_1 = batch_iou[:, index_miou + 2]
        iou_un_2 = batch_iou[:, index_miou + 3]

        # iou_calc = torch.div(torch.sum(iou_int), torch.sum(iou_un))
        # iou_calc = torch.div(torch.sum(iou_int),( torch.sum(iou_un)-torch.sum(iou_int)))
        iou_calc = torch.div(torch.sum(iou_int),( torch.sum(iou_un_1)+ torch.sum(iou_un_2) -torch.sum(iou_int)))

        if tensorbd != "None":
            prec = torch.mean(prec_rec[:, index_prec])
            rec = torch.mean(prec_rec[:, index_prec + 1])
            index_prec += 2
            index_miou += 4
            ispis = str(train_part) + " Precision Classwise/" + classes_labels[klasa] + " "
            tensorbd.add_scalar(ispis, np.round(prec.detach().cpu(), 4), epoch)

            ispis = str(train_part) + " Recall Classwise/" + classes_labels[klasa] + " "
            tensorbd.add_scalar(ispis, np.round(rec.detach().cpu(), 4), epoch)

            ispis = str(train_part) + "Mean IOU Classwise/" + classes_labels[klasa] + " "
            tensorbd.add_scalar(ispis, np.round(iou_calc.detach().cpu(), 4), epoch)
            total_iou[klasa] = np.round(iou_calc.detach().cpu(), 4)

            current_class_loss = classwise_loss[:, klasa]
            current_class_loss = current_class_loss[current_class_loss != 0]
            class_loss = torch.mean(current_class_loss)
            ispis = str(train_part) + " Mean Loss Classwise/" + classes_labels[klasa] + " "
            tensorbd.add_scalar(ispis, np.round(class_loss.detach().cpu(), 4), epoch)

        if train_part == 'Test':
            ispis = train_part + "Mean IOU Classwise/" + classes_labels[klasa] + " " + str(
                np.round(iou_calc.detach().cpu(), 4))
            IOU.append(np.round(iou_calc.detach().cpu().numpy(), 4))
            print(ispis)
            upisivanje(ispis, ime_foldera_za_upis)
        else:
            if tensorbd!="None":
                ispis = train_part + " Mean IOU Classwise/" + classes_labels[klasa] + " "
                tensorbd.add_scalar(ispis, np.round(iou_calc.detach().cpu(), 4), epoch)

    return IOU


def iou_pix(target, pred, mask_var, use_mask):
    if torch.sum(target) == 0 and torch.sum(pred) == 0:
        arr = torch.full(size=(target.shape[0], target.shape[1]), fill_value=2)
        return arr[arr != 2].sum(), arr[arr != 2].sum(), target.flatten().sum(),pred.flatten().sum()
    else:
        if use_mask:
            intersection = torch.logical_and(target.bool(), pred.bool())[mask_var].sum()
            # intersection = torch.logical_and(torch.logical_and(target.bool(), pred.bool()),mask_bool).sum()
            union = torch.logical_or(target.bool(), pred.bool())[mask_var].sum()
            return intersection, union, target.flatten().sum(),pred.flatten().sum()
        else:
            intersection = torch.logical_and(target.bool(), pred.bool()).sum()
            union = torch.logical_or(target.bool(), pred.bool()).sum()
            return intersection, union, target.flatten().sum(),pred.flatten().sum()

def precision_pix(target, pred, z_test, loss_mask):

    # z_bool = torch.logical_and(z_test[0, :, :].bool(), z_test[0, :, :].bool())
    target = torch.logical_and(target.bool(),z_test).flatten()
    pred = torch.logical_and(pred.bool(),z_test).flatten()

    epsilon = 1e-10
    TP = (pred & target).sum().float()
    TN = ((~pred) & (~target)).sum().float()
    FP = (pred & (~target)).sum().float()
    FN = ((~pred) & target).sum().float()
    #accuracy = (TP+TN)/(TP+TN+FP+FN)
    precision = torch.mean(TP / (TP + FP + epsilon))
    recall = torch.mean(TP / (TP + FN + epsilon))

    return precision, recall

def calc_metrics_pix(model_output, target_var, mask_var, num_classes, device, use_mask, loss_type):
    iou_res = torch.zeros((target_var.shape[0], target_var.shape[1] * 4), device=device)
    prec_rec = torch.zeros((target_var.shape[0], target_var.shape[1] * 2)).to(device)

    if loss_type == 'bce':
        iou_res_bg = torch.zeros((target_var.shape[0], 2), device=device)
    for im_number in range(target_var.shape[0]):

        tresholded = model_output[im_number, :, :, :] > 0.5
        tresholded = tresholded.byte()
        tresholded_tmp = torch.max(tresholded, dim=0).values
        if loss_type == 'bce':
            # bg_tresholded = torch.tensor(tresholded_tmp == 0).byte()
            bg_tresholded = (tresholded_tmp == 0)
            bg_target_var = torch.max(target_var[im_number, :, :, :], dim=0).values
            bg_target_var = (bg_target_var == 0)
            # bg_target_var = torch.tensor(bg_target_var == 0).byte()
            iou_res_bg[im_number, 0], iou_res_bg[im_number, 1] = iou_pix(bg_target_var, bg_tresholded,
                                                                         mask_var[im_number], use_mask)

        ind_iou = 0
        ind_prec = 0
        for klasa_idx in range(num_classes):
            iou_res[im_number, ind_iou], iou_res[im_number, ind_iou + 1], iou_res[im_number, ind_iou + 2], iou_res[im_number, ind_iou + 3] = iou_pix(
                target_var[im_number, klasa_idx, :, :], tresholded[klasa_idx, :, :], mask_var[im_number], use_mask)
            prec_rec[im_number, ind_prec], prec_rec[im_number, ind_prec + 1] = precision_pix(target_var[im_number, klasa_idx, :, :],
             tresholded[klasa_idx, :, :],  mask_var[im_number], use_mask)
            ind_prec += 2
            ind_iou += 4
    if loss_type == 'ce':
        return iou_res,prec_rec
    elif loss_type == 'bce':
        return iou_res, iou_res_bg
    else:
        print("Error: Unimplemented loss type")
        sys.exit(0)


def calc_metrics_tb(model_output, target_var, mask_var, num_classes, use_mask):
    miou_mean = []

    for batch in range(target_var.shape[0]):
        miou_res = torch.zeros([num_classes])
        tresholded = model_output[batch, :, :, :] > 0.5
        tresholded = tresholded.byte()

        for klasa_idx in range(num_classes):
            miou_res[klasa_idx] = iou_coef(target_var.permute(0, 2, 3, 1)[batch, :, :, klasa_idx].byte(),
                                           tresholded[klasa_idx, :, :], mask_var[batch, :, :], use_mask)

        miou_res = [x for x in miou_res if torch.isnan(x) == False]

        miou_mean.append(torch.mean(torch.tensor(miou_res, dtype=torch.float32)))

    return miou_mean


def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = torch.sum(y_true_f * y_pred_f)
    # smooth = 0.0001
    smooth = 0
    return (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)


def iou_coef(y_true, y_pred, mask_var, use_mask):
    if use_mask:
        y_true_f = y_true[mask_var]
        y_pred_f = y_pred[mask_var]
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = torch.sum(y_true_f * y_pred_f)
    union = torch.sum(y_true_f) + torch.sum(y_pred_f) - intersection
    # smooth = 0.0001
    smooth = 0
    return (intersection + smooth) / (union + smooth)

# ### https://www.jeremyjordan.me/semantic-segmentation/#loss
# def dice_pix(target, pred):
#     if torch.sum(target) == 0 and torch.sum(pred) == 0:
#         arr = torch.full(size=(target.shape[0], target.shape[1]),fill_value=2)
#         return arr[arr!=2].sum(), arr[arr!=2].sum(), arr[arr!=2].sum()

#     else:
#         intersection = torch.logical_and(target.bool(), pred.bool())
#         return intersection[intersection!=2].sum(), target[target!=2].sum(), pred[pred!=2].sum()


# ### https://www.jeremyjordan.me/semantic-segmentation/#loss
# def dice_tb(im1, im2):

#     if torch.sum(im1) >= 0 and torch.sum(im2) == 0:
#         return 0
#     im1 = torch.tensor(im1,dtype = torch.bool)
#     im2 = torch.tensor(im2,dtype = torch.bool)

#     if im1.shape != im2.shape:
#         raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

#     intersection = torch.logical_and(im1, im2)
#     return 2. * intersection.sum() / (im1.sum() + im2.sum())
#     # return np.asarray(intersection),np.asarray(im1),np.asarray(im2)


# ### https://www.jeremyjordan.me/evaluating-image-segmentation-models/
# def iou_tb(im1, im2):

#     if torch.sum(im1) >= 0 and torch.sum(im2) == 0:
#         return 0
#     im1 = torch.tensor(im1,dtype = torch.bool)
#     im2 = torch.tensor(im2,dtype = torch.bool)
#     if im1.shape != im2.shape:
#         raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

#     intersection = torch.logical_and(im1, im2)
#     union = torch.logical_or(im1, im2)
#     return torch.sum(intersection) / torch.sum(union)

# def calc_metrics_test(model_output, target_var, num_classes):

#     for im_number in range(target_var.shape[0]):
#         dice_res = torch.zeros([num_classes])
#         miou_res = torch.zeros([num_classes])

#         tresholded = model_output[im_number, :, :, :]>0.5
#         tresholded = tresholded.byte()

#         for i in range(num_classes):
#             miou_res[i] = iou_coef(target_var.permute(0, 2, 3, 1)[im_number, :, :, i].byte(),
#                                  tresholded[i,:,:])
#             dice_res[i] = dice_coef(target_var.permute(0, 2, 3, 1)[im_number, :, :, i].byte(),
#                                  tresholded[i,:,:])

#     return miou_res,dice_res

# def iou_coef_pix(y_true,y_pred):
#     y_true_f = y_true.flatten()
#     y_pred_f = y_pred.flatten()
#     intersection = torch.sum(y_true_f * y_pred_f)
#     union = torch.sum(y_true_f) + torch.sum(y_pred_f) - intersection
#     # smooth = 0.0001
#     smooth = 0
#     return intersection,union

# def dice_coef_pix(y_true, y_pred):
#     y_true_f = y_true.flatten()
#     y_pred_f = y_pred.flatten()
#     intersection = (y_true_f * y_pred_f)
#     # smooth = 0.0001
#     smooth = 0
#     return intersection, y_true_f, y_pred_f