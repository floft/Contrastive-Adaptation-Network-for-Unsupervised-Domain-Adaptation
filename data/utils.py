import torchvision.transforms as transforms
from PIL import Image
import torch
from config.config import cfg
import os
# import pickle
import pickle5 as pickle

# def get_transform(train=True):
#     transform_list = []
#     if cfg.DATA_TRANSFORM.RESIZE_OR_CROP == 'resize_and_crop':
#         osize = [cfg.DATA_TRANSFORM.LOADSIZE, cfg.DATA_TRANSFORM.LOADSIZE]
#         transform_list.append(transforms.Resize(osize, Image.BICUBIC))
#         if train:
#             transform_list.append(transforms.RandomCrop(cfg.DATA_TRANSFORM.FINESIZE))
#         else:
#             if cfg.DATA_TRANSFORM.WITH_FIVE_CROP:
#                 transform_list.append(transforms.FiveCrop(cfg.DATA_TRANSFORM.FINESIZE))
#             else:
#                 transform_list.append(transforms.CenterCrop(cfg.DATA_TRANSFORM.FINESIZE))

#     elif cfg.DATA_TRANSFORM.RESIZE_OR_CROP == 'crop':
#         if train:
#             transform_list.append(transforms.RandomCrop(cfg.DATA_TRANSFORM.FINESIZE))
#         else:
#             if cfg.DATA_TRANSFORM.WITH_FIVE_CROP:
#                 transform_list.append(transforms.FiveCrop(cfg.DATA_TRANSFORM.FINESIZE))
#             else:
#                 transform_list.append(transforms.CenterCrop(cfg.DATA_TRANSFORM.FINESIZE))

#     if train and cfg.DATA_TRANSFORM.FLIP:
#         transform_list.append(transforms.RandomHorizontalFlip())

#     to_normalized_tensor = [transforms.ToTensor(),
#                             transforms.Normalize(mean=cfg.DATA_TRANSFORM.NORMALIZE_MEAN,
#                                        std=cfg.DATA_TRANSFORM.NORMALIZE_STD)]

#     if not train and cfg.DATA_TRANSFORM.WITH_FIVE_CROP:
#         transform_list += [transforms.Lambda(lambda crops: torch.stack([
#                 transforms.Compose(to_normalized_tensor)(crop) for crop in crops]))]
#     else:
#         transform_list += to_normalized_tensor

#     return transforms.Compose(transform_list)


def load_pickle(filename):
    assert os.path.exists(filename), filename + " does not exist"

    with open(filename, 'rb') as f:
        data = pickle.load(f)

    return data


def load_all_from_pickle(filename):
    images = []
    labels = []
    data = load_pickle(filename)

    for label, examples in data.items():
        for i in range(len(examples)):
            # To be compatible with prior version, needs to be a string
            images.append("{}:{}:{}".format(filename, label, i))
            labels.append(int(label))

    return images, labels


def load_class_from_pickle(filename, category):
    images = []
    data = load_pickle(filename)

    assert category in data.keys(), "key " + str(category) + " not found"
    examples = data[category]

    for i in range(len(examples)):
        # To be compatible with prior version, needs to be a string
        images.append("{}:{}:{}".format(filename, category, i))

    return images


def split_into_parts(path):
    pickle_filename, label, ex_ind = path.split(":")
    return pickle_filename, label, int(ex_ind)


def dataloading_is_v2(filename_or_dir):
    """ if v2 then the filename will have exactly three parts """
    parts = filename_or_dir.split(":")
    is_v2 = len(parts) == 3

    # TODO for now we're assuming it's v2
    assert is_v2, "should be using data loading v2"

    return is_v2


def class_load_from_pickle(cls, path):
    """ Hacky way to load the possibly-multiple pickle files and get the images
    from the "path" pickle_filename:label:example_index string format"""
    pickle_filename, label, ex_ind = split_into_parts(path)
    if not hasattr(cls, "loaded_pickle_data"):
        cls.loaded_pickle_data = {}
    if pickle_filename not in cls.loaded_pickle_data:
        cls.loaded_pickle_data[pickle_filename] = load_pickle(pickle_filename)
    img = cls.loaded_pickle_data[pickle_filename][label][ex_ind]
    return img
