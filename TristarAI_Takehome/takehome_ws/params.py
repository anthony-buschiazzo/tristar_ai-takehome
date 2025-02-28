import os

from torchvision import transforms as T
from sklearn.model_selection import train_test_split

### Script to gather lists of images, preprocessing functions, and other parameters to be used in creating the dataloaders

class GetParams():
    def __init__(self, cfg, training):

        if training:
            data_dir = cfg["train_data_dir"]
            train_split = cfg["train_split"]
            self.bs = cfg["bs"]
            self.aug = cfg["augmentations"]

            ben_img_list = [os.path.join(data_dir, "Benign", imgpath) for imgpath in os.listdir(os.path.join(data_dir, "Benign"))]
            ben_cls = [0 for i in range(len(ben_img_list))]
            mal_img_list = [os.path.join(data_dir, "Malignant", imgpath) for imgpath in os.listdir(os.path.join(data_dir, "Malignant"))]
            mal_cls = [1 for i in range(len(mal_img_list))]
            
            ben_train_x, ben_val_x, ben_train_y, ben_val_y = train_test_split(ben_img_list, ben_cls, train_size = train_split)
            mal_train_x, mal_val_x, mal_train_y, mal_val_y = train_test_split(mal_img_list, mal_cls, train_size = train_split)

            self.train_x = ben_train_x + mal_train_x
            self.train_y = ben_train_y + mal_train_y
            self.val_x = ben_val_x + mal_val_x
            self.val_y = ben_val_y + mal_val_y
        
        else:
            self.aug = False
            data_dir = cfg["test_data_dir"]
            ben_img_list = [os.path.join(data_dir, "Benign", imgpath) for imgpath in os.listdir(os.path.join(data_dir, "Benign"))]
            ben_cls = [0 for i in range(len(ben_img_list))]
            mal_img_list = [os.path.join(data_dir, "Malignant", imgpath) for imgpath in os.listdir(os.path.join(data_dir, "Malignant"))]
            mal_cls = [1 for i in range(len(mal_img_list))]

            self.eval_x = ben_img_list + mal_img_list
            self.eval_y = ben_cls + mal_cls

        self.get_preprocessing()
        
    def get_training_params(self):
        train_params = {
            "image_list": self.train_x,
            "labels": self.train_y,
            "bs": self.bs,
            "pre_fn": self.train_pre_fn
        }

        val_params = {
            "image_list": self.val_x,
            "labels": self.val_y,
            "bs": self.bs,
            "pre_fn": self.val_pre_fn
        }

        return train_params, val_params

    def get_eval_params(self):
        eval_params = {
            "image_list": self.eval_x,
            "labels": self.eval_y,
            "pre_fn": self.val_pre_fn
        }

        return eval_params

    def get_preprocessing(self):

        if self.aug:
            self.train_pre_fn = T.Compose([
                T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)], p=0.1),
                T.RandomApply([T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 3.0))], p=0.1),
                T.RandomApply([T.RandomRotation(degrees=(0, 180), interpolation=T.InterpolationMode.BILINEAR, fill=0)], p=0.25),
                T.RandomHorizontalFlip(p = 0.5),
                T.RandomVerticalFlip(p = 0.5),
                T.ToTensor(),
                T.Normalize(
                    mean = [0.485, 0.456, 0.406],
                    std = [0.229, 0.224, 0.225],
                )
            ])

        else:
            self.train_pre_fn = T.Compose([
                T.RandomHorizontalFlip(p = 0.5),
                T.RandomVerticalFlip(p = 0.5),
                T.ToTensor(),
                T.Normalize(
                    mean = [0.485, 0.456, 0.406],
                    std = [0.229, 0.224, 0.225],
                )
            ])

        self.val_pre_fn = T.Compose([
            T.ToTensor(),
            T.Normalize(
                mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225],
            )
        ])



