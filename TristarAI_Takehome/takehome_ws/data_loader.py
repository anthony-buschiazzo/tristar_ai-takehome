from torch.utils.data import Dataset, DataLoader
from PIL import Image

### Script to create dataset and dataloaders for training and evaluation purposed

def get_train_loaders(train_params, val_params):
    train_dataset = ClassificationDataset(train_params)
    train_loader = DataLoader(
        train_dataset,
        batch_size = train_params["bs"],
        shuffle = True,
        drop_last = True
    )

    val_dataset = ClassificationDataset(val_params)
    val_loader = DataLoader(
        val_dataset,
        batch_size = val_params["bs"],
        shuffle = True,
        drop_last = True
    )

    return train_loader, val_loader

def get_eval_loader(eval_params):
    eval_dataset = ClassificationDataset(eval_params)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size = 1,
        shuffle = False,
        drop_last = False
    )
    return eval_loader

class ClassificationDataset(Dataset):
    
    def __init__(self, params):
        self.image_list = params["image_list"]
        self.labels = params["labels"]
        self.pre_fn = params["pre_fn"]

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        image_path = self.image_list[index]
        image = Image.open(image_path)
        image = self.pre_fn(image)
        label = self.labels[index]
        return image, label
        