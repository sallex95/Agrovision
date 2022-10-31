import json, os, sys
import time
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import seaborn as sns
from torchvision import transforms
from torch import distributed, nn
import copy

# base_folder_path = os.getcwd()
base_folder_path = os.path.dirname(__file__)

base_folder_path = base_folder_path.replace("\\", "/")


def upisivanje(ispis, ime_foldera):
    fff = open(ime_foldera, "a")
    fff.write(str(ispis) + "\n")
    fff.close()


def get_args(phase):
    if phase == 'test':
        # json_path = r"/home/stefanovicd/DeepSleep/agrovision/AgroVisionUnetBS/config_test.json"
        json_path = base_folder_path + "/config_test.json"
    else:
        # json_path = r"/home/stefanovicd/DeepSleep/agrovision/AgroVisionUnetBS/config.json"
        json_path = base_folder_path + "/config.json"

    with open(json_path) as f:
        tmp = json.load(f)
        # tmp = json.load()
    return tmp


def distributed_is_initialized():
    return distributed.is_available() and distributed.is_initialized()


def zscore_func(img, device, dataset):
    ###  vrednosti mean-a i stda trenutno nebitne s obzirom da se funkcija ne koristi, ali ih treba proveriti u trenutku kada odlucimo da je koristimo
    if dataset == 'full':
        mean = [109.26965719710468, 104.92564362077304, 104.91644731235822, 91.07115091256287]
        std = [54.893713004157824, 54.88445400637186, 55.82919057285158, 50.98149954879407]
    else:
        mean = [106.99351300380026, 99.899188265936, 100.73070236570767, 84.18472954407527]
        std = [56.57514489959256, 55.184922355586096, 57.359027987793745, 52.03757791560299]
    for ch in range(len(mean)):
        img[ch, :, :] = (img[ch, :, :] - mean[ch]) / std[ch]
    img = torch.from_numpy(np.ascontiguousarray(img)).to(device)
    return img


def inv_zscore_func(img, dataset):
    img_tmp = copy.deepcopy(img)

    ###  vrednosti mean-a i stda trenutno nebitne s obzirom da se funkcija ne koristi, ali ih treba proveriti u trenutku kada odlucimo da je koristimo
    if dataset == 'full':
        mean = [109.26965719710468, 104.92564362077304, 104.91644731235822, 91.07115091256287]
        std = [54.893713004157824, 54.88445400637186, 55.82919057285158, 50.98149954879407]
    else:
        mean = [106.99351300380026, 99.899188265936, 100.73070236570767, 84.18472954407527]
        std = [56.57514489959256, 55.184922355586096, 57.359027987793745, 52.03757791560299]
    for ch in range(len(mean)):
        img_tmp[ch, :, :] = (img_tmp[ch, :, :] * std[ch]) + mean[ch]

    return img_tmp


def norm_func(img, device):
    for ch in range(4):
        img[ch, :, :] = img[ch, :, :] / 255
    img = torch.from_numpy(np.ascontiguousarray(img.detach().cpu())).to(device)
    return img


def inv_norm_func(img):
    img_tmp = copy.deepcopy(img)

    for ch in range(4):
        img_tmp[ch, :, :] = img_tmp[ch, :, :] * 255.0
    return np.array(img_tmp, dtype='uint8')


class AgroVisionDataSet(Dataset):
    def __init__(self, img_size,
                 root_dir, data_format, shuffle_state,
                 transform, device, zscore, binary, dataset, background_flag):

        imgs_num = int(len(os.listdir(root_dir)))
        img_paths_1 = os.listdir(root_dir)
        img_paths_1 = shuffle(img_paths_1, random_state=shuffle_state)
        img_paths = img_paths_1[:imgs_num]
        img_paths_valid = img_paths_1[imgs_num:]

        self.img_names_valid = [p.split('/')[-1][:-4] for p in img_paths_valid]
        self.img_names = [p.split('/')[-1][:-4] for p in img_paths]

        self.img_size = img_size
        self.root_dir = root_dir
        self.transform = transform
        self.data_format = data_format
        self.device = device
        self.zscore = zscore
        self.binary = binary
        self.background_flag = background_flag
        self.dataset = dataset

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]

        if self.zscore == False:
            img = torch.from_numpy(np.ascontiguousarray(
                np.load(os.path.join(self.root_dir, img_name + self.data_format), allow_pickle=False).astype(
                    np.float32))).to(self.device)

            img = norm_func(img, self.device)
        elif self.zscore:

            img = np.load(os.path.join(self.root_dir, img_name + self.data_format), allow_pickle=False).astype(
                np.float32)
            img = zscore_func(img, self.device, self.dataset)
        x = img[0:4, :, :]
        masks = img[-2:, :, :]
        if self.binary:
            if self.background_flag:
                y_foreground = ((img[5, :, :] + img[6, :, :] + img[7, :, :] + img[8, :, :] + img[9, :, :] + img[10, :,
                                                                                                            :]) > 0).float()
            else:
                y_foreground = ((img[4, :, :] + img[5, :, :] + img[6, :, :] + img[7, :, :] + img[8, :, :] + img[9, :,
                                                                                                            :] + img[10,
                                                                                                                 :,
                                                                                                                 :]) > 0).float()

            y = torch.tensor(y_foreground.unsqueeze(0))
        else:
            if self.background_flag:
                y = img[4:-2, :, :]
            else:
                y = img[5:-2, :, :]

        classes = torch.zeros([y.shape[0]])
        for i in range(y.shape[0]):
            if torch.sum(y[i, :, :]) > 0:
                classes[i] += 1

        return x, y, img_name, masks, classes


class AgroVisionDataLoader(DataLoader):
    def __init__(self, img_size, root_dir, data_format, shuffle_state, batch_size, device, zscore=False, binary=True,
                 dataset='mini', background_flag=False):
        transform = transforms.ToTensor()

        train_dataset = AgroVisionDataSet(img_size=img_size,
                                          root_dir=root_dir,
                                          data_format=data_format,
                                          shuffle_state=shuffle_state,
                                          transform=transform, device=device, zscore=zscore, binary=binary,
                                          dataset=dataset, background_flag=background_flag)

        sampler = None
        if distributed_is_initialized():
            sampler = DistributedSampler(train_dataset)

        super(AgroVisionDataLoader, self).__init__(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler  # , num_workers=2,persistent_workers=True
        )


def data_loading(ime_foldera_za_upis, numpy_path, numpy_valid_path, binary, background_flag):
    tmp = get_args('train')
    globals().update(tmp)
    time_start_load = time.time()

    train_loader = AgroVisionDataLoader(img_size, numpy_path, img_data_format, shuffle_state,
                                        batch_size, device, zscore, binary, dataset, background_flag)

    valid_loader = AgroVisionDataLoader(img_size, numpy_valid_path, img_data_format, shuffle_state,
                                        batch_size, device, zscore, binary, dataset, background_flag)

    test_loader = AgroVisionDataLoader(img_size, numpy_test_path, img_data_format, shuffle_state,
                                       batch_size, device, zscore, binary, dataset, background_flag)
    time_end_load = time.time()

    ispis = ("Time Elapsed Load Dataset", str(time_end_load - time_start_load))
    print(ispis)
    upisivanje(ispis, ime_foldera_za_upis)

    return train_loader, valid_loader, test_loader


def decode_segmap2(image, nc, device, loss_type, year):
    if year == '2020':
        if loss_type == 'bce':

            label_colors = torch.tensor([  # 0=background
                # 1=cloud_shadow, 2=double_plant, 3=planter_skip, 4=standing_water, 5=waterway
                (255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 0, 255), (255, 0, 255),
                # 6=weed_cluster
                (0, 255, 255)]).byte()

        elif loss_type == "ce":
            label_colors = torch.tensor([(0, 0, 0),  # 0=background
                                         # 1=cloud_shadow, 2=double_plant, 3=planter_skip, 4=standing_water, 5=waterway
                                         (255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 0, 255), (255, 0, 255),
                                         # 6=weed_cluster
                                         (0, 255, 255)]).byte()
    elif year == '2021':
        if loss_type == 'bce':

            label_colors = torch.tensor([  # 0=background
                # 1=double_plant, 2=drydown, 3=end_row, 4=nutrient_deficiency, 5=planter_skip
                (255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 0, 255), (255, 0, 255),
                # 6=storm_damage, 7=water, 8=waterway, 9=weed_cluster
                (0, 255, 255), (155, 0, 0), (155, 155, 155), (70, 0, 155)]).byte()

        elif loss_type == "ce":
            label_colors = torch.tensor([(0, 0, 0),  # 0=background
                                         # 1=double_plant, 2=drydown, 3=end_row, 4=nutrient_deficiency, 5=planter_skip
                                         (255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 0, 255), (255, 0, 255),
                                         # 6=storm_damage, 7=water, 8=waterway, 9=weed_cluster
                                         (0, 255, 255), (155, 0, 0), (155, 155, 155), (70, 0, 155)]).byte()
    else:
        print("Error: Unimplemented loss type! Color coding interupted!")
        sys.exit(0)
    r = torch.zeros(size=(image.shape[-2], image.shape[-1]), device=device, dtype=torch.uint8)
    g = torch.zeros(size=(image.shape[-2], image.shape[-1]), device=device, dtype=torch.uint8)
    b = torch.zeros(size=(image.shape[-2], image.shape[-1]), device=device, dtype=torch.uint8)

    for class_idx in range(0, nc):
        if nc > 2:
            image = image.squeeze(0)
        if len(image.shape) > 2:
            idx = image[class_idx, :, :] == 1
        else:
            idx = image == class_idx

        r[idx] = label_colors[class_idx, 0]
        g[idx] = label_colors[class_idx, 1]
        b[idx] = label_colors[class_idx, 2]

    rgb = torch.stack([r, g, b], axis=2)

    return rgb


def range_norm(img):
    return (img - torch.min(img)) / (torch.max(img) - torch.min(img))


def load_raw_data(test_loader, name, k_iter, loss_type):
    image = np.load(os.path.join(test_loader.dataset.root_dir, name.iloc[k_iter]['filenames']) + ".npy",
                    allow_pickle=False).astype(np.float32)[:4]
    if loss_type == 'bce':
        # Posto zelimo da izbacimo background klasu kada koristimo BCE loss , indeksiramo kanali kao [5:], ako se radi sa background klasom odnosno koristi se CE loss, promeniti na [4:]
        target = np.load(os.path.join(test_loader.dataset.root_dir, name.iloc[k_iter]['filenames']) + ".npy",
                         allow_pickle=False).astype(np.float32)[5:-2]
    else:
        target = np.load(os.path.join(test_loader.dataset.root_dir, name.iloc[k_iter]['filenames']) + ".npy",
                         allow_pickle=False).astype(np.float32)[4:-2]

    return image, target






