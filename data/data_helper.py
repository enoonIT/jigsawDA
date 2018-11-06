from os.path import join, dirname

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data import StandardDataset
from data.JigsawLoader import JigsawDataset, SimpleDataset, get_split_dataset_info, _dataset_info, JigsawTestDatasetMultiple
from data.concat_dataset import ConcatDataset

mnist = 'mnist'
mnist_m = 'mnist_m'
svhn = 'svhn'
synth = 'synth'
usps = 'usps'

vlcs_datasets = ["CALTECH", "LABELME", "PASCAL", "SUN"]
pacs_datasets = ["art_painting", "cartoon", "photo", "sketch"]
office_datasets = ["amazon", "dslr", "webcam"]
digits_datasets = [mnist, mnist, svhn, usps]
available_datasets = office_datasets + pacs_datasets + vlcs_datasets + digits_datasets
office_paths = {dataset: "/home/enoon/data/images/office/%s" % dataset for dataset in office_datasets}
pacs_paths = {dataset: "/home/enoon/data/images/PACS/kfold/%s" % dataset for dataset in pacs_datasets}
vlcs_paths = {dataset: "/home/enoon/data/images/VLCS/%s/test" % dataset for dataset in pacs_datasets}
paths = {**office_paths, **pacs_paths, **vlcs_paths}

dataset_std = {mnist: (0.30280363, 0.30280363, 0.30280363),
               mnist_m: (0.2384788, 0.22375608, 0.24496263),
               svhn: (0.1951134, 0.19804622, 0.19481073),
               synth: (0.29410212, 0.2939651, 0.29404707),
               usps: (0.25887518, 0.25887518, 0.25887518),
               }

dataset_mean = {mnist: (0.13909429, 0.13909429, 0.13909429),
                mnist_m: (0.45920207, 0.46326601, 0.41085603),
                svhn: (0.43744073, 0.4437959, 0.4733686),
                synth: (0.46332872, 0.46316052, 0.46327512),
                usps: (0.17025368, 0.17025368, 0.17025368),
                }


class Subset(torch.utils.data.Dataset):
    def __init__(self, dataset, limit):
        indices = torch.randperm(len(dataset))[:limit]
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def get_train_dataloader(args, patches):
    dataset_list = args.source
    assert isinstance(dataset_list, list)
    datasets = []
    val_datasets = []
    img_transformer, tile_transformer = get_train_transformers(args)
    limit = args.limit_source
    for dname in dataset_list:
        name_train, name_val, labels_train, labels_val = get_split_dataset_info(join(dirname(__file__), 'txt_lists', '%s_train.txt' % dname), args.val_size)
        train_dataset = JigsawDataset(name_train, labels_train, patches=patches, img_transformer=img_transformer,
                                      tile_transformer=tile_transformer, jig_classes=args.jigsaw_n_classes, bias_whole_image=args.bias_whole_image)
        if limit:
            train_dataset = Subset(train_dataset, limit)
        datasets.append(train_dataset)
        val_datasets.append(
            SimpleDataset(name_val, labels_val, img_transformer=get_val_transformer(args),
                          patches=patches, jig_classes=args.jigsaw_n_classes))
    dataset = ConcatDataset(datasets)
    val_dataset = ConcatDataset(val_datasets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    return loader, val_loader


def get_val_dataloader(args, patches=False):
    names, labels = _dataset_info(join(dirname(__file__), 'txt_lists', '%s_test.txt' % args.target))
    img_tr = get_val_transformer(args)
    val_dataset = SimpleDataset(names, labels, patches=patches, img_transformer=img_tr, jig_classes=args.jigsaw_n_classes)
    if args.limit_target and len(val_dataset) > args.limit_target:
        val_dataset = Subset(val_dataset, args.limit_target)
        print("Using %d subset of val dataset" % args.limit_target)
    dataset = ConcatDataset([val_dataset])
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    return loader


def get_combo_dataloader(args, patches):
    dataset_list = args.source
    assert isinstance(dataset_list, list)
    jig_datasets = []
    img_datasets = []
    val_datasets = []
    jig_transformer, tile_transformer = get_train_transformers(args)
    img_transformer = transforms.Compose(
        jig_transformer.transforms + [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    limit = args.limit_source
    jig_classes = args.jigsaw_n_classes
    for dname in dataset_list:
        name_train, _, labels_train, _ = get_split_dataset_info(join(dirname(__file__), 'txt_lists', '%s_train.txt' % dname), args.val_size)
        jig_train_dataset = JigsawDataset(name_train, labels_train, patches=patches, img_transformer=img_transformer,
                                          tile_transformer=tile_transformer, jig_classes=jig_classes, bias_whole_image=args.bias_whole_image)
        if limit:
            raise "Not implemented yet"
        jig_datasets.append(jig_train_dataset)

    if args.target in args.source:
        k = args.source.index(args.target)
        classification_dataset_list = dataset_list[:k] + dataset_list[k + 1:]
    else:
        classification_dataset_list = dataset_list
    for dname in classification_dataset_list:
        name_train, name_val, labels_train, labels_val = get_split_dataset_info(join(dirname(__file__), 'txt_lists', '%s_train.txt' % dname), args.val_size)
        img_dataset = SimpleDataset(name_train, labels_train, img_transformer=img_transformer, patches=patches, jig_classes=jig_classes)
        if limit:
            raise "Not implemented yet"
        img_datasets.append(img_dataset)
        val_datasets.append(
            SimpleDataset(name_val, labels_val, img_transformer=get_val_transformer(args),
                          patches=patches, jig_classes=jig_classes))
    jig_loader = torch.utils.data.DataLoader(ConcatDataset(jig_datasets), batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,
                                             drop_last=True)
    img_loader = torch.utils.data.DataLoader(ConcatDataset(img_datasets), batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,
                                             drop_last=True)
    val_loader = torch.utils.data.DataLoader(ConcatDataset(val_datasets), batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True,
                                             drop_last=False)
    return img_loader, jig_loader, val_loader


def get_train_transformers(args):
    img_tr = [transforms.RandomResizedCrop(int(args.image_size), (args.min_scale, args.max_scale))]
    if args.random_horiz_flip > 0.0:
        img_tr.append(transforms.RandomHorizontalFlip(args.random_horiz_flip))
    if args.jitter > 0.0:
        img_tr.append(transforms.ColorJitter(brightness=args.jitter, contrast=args.jitter, saturation=args.jitter, hue=min(0.5, args.jitter)))

    tile_tr = []
    if args.tile_random_grayscale:
        tile_tr.append(transforms.RandomGrayscale(args.tile_random_grayscale))
    tile_tr = tile_tr + [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    return transforms.Compose(img_tr), transforms.Compose(tile_tr)


def get_val_transformer(args):
    img_tr = [transforms.Resize(args.image_size), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    return transforms.Compose(img_tr)
