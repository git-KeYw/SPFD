import numpy as np
import torch
import copy
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

def _load_event_image_npz(npz_path, expect_hw=None, to_uint8=True):
    d = np.load(npz_path)
    if "arr_0" not in d:
        raise KeyError(f"{npz_path} missing 'arr_0'")
    img = d["arr_0"]  # float32, HxW

    if expect_hw is not None and (img.shape[0] != expect_hw[0] or img.shape[1] != expect_hw[1]):
        pass

    if to_uint8:
        m, M = float(img.min()), float(img.max())
        if M > m:
            img = (img - m) / (M - m)
        else:
            img = np.zeros_like(img)
        img = (img * 255.0).astype(np.uint8)  # HxW uint8
    else:
        img = img.astype(np.float32)

    if img.ndim == 2:
        img = img[:, :, None]
    return img

class DsecDetMapper:
    def __init__(self, augmentation, augmentation_with_crop, is_train=True, img_format="RGB", mask_on=False):
        self.is_train = is_train
        self.img_format = img_format
        self.mask_on = mask_on
        self.augmentation = augmentation
        self.augmentation_with_crop = augmentation_with_crop

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)

        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if self.augmentation_with_crop is None:
            image, transforms = T.apply_transform_gens(self.augmentation, image)
        else:
            if np.random.rand() > 0.5:
                image, transforms = T.apply_transform_gens(self.augmentation, image)
            else:
                image, transforms = T.apply_transform_gens(self.augmentation_with_crop, image)

        image_shape = image.shape[:2]
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if "event_file" in dataset_dict:
            ev_img = _load_event_image_npz(
                dataset_dict["event_file"],
                expect_hw=(dataset_dict["height"], dataset_dict["width"]),
                to_uint8=True,
            )
            ev_img = transforms.apply_image(ev_img)  # HxWx1 / HxWx3
            if ev_img.shape[2] == 1:
                ev_img = np.repeat(ev_img, 3, axis=2)
                pass
            dataset_dict["event"] = torch.as_tensor(np.ascontiguousarray(ev_img.transpose(2, 0, 1))).float()

        if self.is_train and "annotations" in dataset_dict:
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                anno.pop("keypoints", None)
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(annos, image_shape)
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict