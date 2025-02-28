import argparse
import os
import yaml
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torchvision import models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from params import GetParams
from data_loader import get_train_loaders

def get_cfg():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config_file", type=str, default="/home/tristarAI/takehome_ws/configs/default.yaml", help=("File path to training config file."))
    args = parser.parse_args()
    arg = args.config_file
    return arg

def make_plots(train_loss, val_loss, train_rec, val_rec, epochs, save_dir):
    #function to plot train and validation loss and recall curves through out training. saves to where models are saved
    plt.figure(figsize=(10, 7))
    plt.plot(epochs, train_loss, color="green", linestyle="-", label="Train Loss")
    plt.plot(epochs, val_loss, color="blue", linestyle="-", label="Validation Loss")
    plt.xlim(1, max(epochs))
    plt.xticks(epochs)
    plt.ylim(0, max(max(train_loss, val_loss))+0.1)
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Train and Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "training_loss.png"))
    plt.close()

    plt.figure(figsize=(10, 7))
    plt.plot(epochs, train_rec, color="green", linestyle="-", label="Train Recall")
    plt.plot(epochs, val_rec, color="blue", linestyle="-", label="Validation Recall")
    plt.xlim(1, max(epochs))
    plt.xticks(epochs)
    plt.ylim(0, 1)
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.title("Train and Validation Recall")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "training_recall.png"))
    plt.close()



class Trainer():
    def __init__(self, cfg_path):

        with open(cfg_path) as stream:
            cfg = yaml.safe_load(stream)
        
        #initilize arguments to be used during training, pulled from the provided config file.
        self.epochs = cfg["epochs"]
        self.warmup = cfg["warmup"]
        self.lr0 = cfg["lr0"]
        self.lrf = cfg["lrf"]
        self.update_lr_freq = cfg["update_lr_freq"]
        self.momentum = cfg["momentum"]
        self.weight_decay = cfg["weight_decay"]
        self.bs = cfg["bs"]

        #creates directory to save models. if directory already exists it will create backup directories until it finds one that doesnt exist
        #done to prevent accidentally saving over previously trained model. also copies config file provided to the directory
        self.logdir = os.path.join(cfg["model_save_dir"], cfg["experiment"])
        backup = self.logdir
        bck_index = 1
        while os.path.isdir(backup):
            backup = self.logdir+"_backup_{}".format(bck_index)
            bck_index += 1
        if os.path.isdir(self.logdir):
            print("\nWarning! {} already exists. Saving to {} instead.".format(self.logdir, backup))
            self.logdir = backup
        else:
            print("\nSaving models to {}".format(self.logdir))
        os.makedirs(self.logdir, exist_ok=False)
        os.system("cp {} {}".format(cfg_path, os.path.join(self.logdir, "config.yaml")))

        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        print("\nTraining on {}.".format(self.device))

        #gather data parameters and create dataloaders
        train_params, val_params = GetParams(cfg, True).get_training_params()

        self.train_loader, self.val_loader = get_train_loaders(train_params, val_params)

        #initialize efficientnet model with pretrained weights provided by torchvision
        #classification layer is replaced with a new set of random weights
        self.model = models.efficientnet_b0(weights="IMAGENET1K_V1")
        self.model.classifier[1] = nn.Linear(in_features = 1280, out_features = 2)
        self.model.to(self.device)

        #define learning rate and step rate to be used for learning rate scheduler
        if self.warmup > 0:
            self.warmup_steps = self.warmup*self.train_loader.__len__()
            self.warmup_steprate = self.lr0/self.warmup_steps
            self.current_lr = 0
        else:
            self.warmup_steps = 0
            self.warmup_steprate = 0
            self.current_lr = self.lr0
        self.total_steps = (self.epochs + self.warmup)*self.train_loader.__len__()

        #initialize stochastic gradient descent optimizer and cross entropy loss function
        self.optimizer = optim.SGD(self.model.parameters(), lr = self.current_lr, momentum = self.momentum, weight_decay = self.weight_decay)

        self.CELoss = nn.CrossEntropyLoss()

        #initilize metrics to compare model performance against and create a pandas dataframe to store metrics to later save
        self.metricslog = pd.DataFrame(columns=["Epoch", "Current LR", "Train CE Loss", "Train Rec","Val CE Loss", "Val Rec", "New Best Val Loss", "New Best Val Rec"])
        self.bestLoss = np.inf
        self.bestRec = -np.inf
        self.all_train_loss = []
        self.all_val_loss = []
        self.all_train_rec = []
        self.all_val_rec = []
        self.all_epochs = []
        self.cur_step = 0
        self.train()
    
    def train(self):
        #start of training function, just loops through the rest of the process for each epoch
        print("\n Training session beginning. Current LR is {} and after {} warmup epochs will reach {}".format(self.current_lr, self.warmup, self.lr0))
        for epoch in range(self.epochs + self.warmup):
            self.cur_epoch = epoch + 1
            self.start_epoch()
        
        print("Training process has completed all epochs.\n\nBest validation loss was {} on epoch {}.\n\nBest validation recall was {} on epoch {}.".format(round(self.bestLoss, 4), self.bestLoss_epoch, round(self.bestRec, 4), self.bestRec_epoch))
    
    def start_epoch(self):
        #intializes loss metrics and cms for current epoch then loops through data for the training portion followed by the validation portion
        self.train_running_loss = 0.0
        self.train_cm = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
        self.model.train(True)
        for i, data in enumerate(tqdm(self.train_loader), 0):
            self.cur_step += 1
            inputs, labels = data
            inputs = inputs.to(self.device).float()
            labels = labels.to(self.device).long()
            self.optimizer.zero_grad()
            self.train_batch_eval(inputs, labels)
        
        self.val_running_loss = 0.0
        self.val_cm = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
        self.model.train(False)
        with torch.no_grad():
            for i, data in enumerate(tqdm(self.val_loader), 0):
                inputs, labels = data
                inputs = inputs.to(self.device).float()
                labels = labels.to(self.device).long()
                self.val_batch_eval(inputs, labels)
        self.end_epoch()
        
    def train_batch_eval(self, inputs, labels):
        #inferences on data, calculates loss, backwards propogates, steps the optimizer, then collects loss and cm metrics
        outputs = self.model(inputs)
        loss = self.CELoss(outputs, labels)
        loss.backward()
        self.optimizer.step()
        cls_preds = torch.argmax(outputs, dim=1)
        self.train_running_loss += loss.item()
        self.train_cm["TP"] += len(np.intersect1d(torch.where(cls_preds == 1)[0].cpu().detach(), torch.where(labels == 1)[0].cpu().detach()))
        self.train_cm["TN"] += len(np.intersect1d(torch.where(cls_preds == 0)[0].cpu().detach(), torch.where(labels == 0)[0].cpu().detach()))
        self.train_cm["FP"] += len(np.intersect1d(torch.where(cls_preds == 1)[0].cpu().detach(), torch.where(labels == 0)[0].cpu().detach()))
        self.train_cm["FN"] += len(np.intersect1d(torch.where(cls_preds == 0)[0].cpu().detach(), torch.where(labels == 1)[0].cpu().detach()))
        self.end_batch()

    def val_batch_eval(self, inputs, labels):
        #inferences on data and collects loss and cm metrics
        outputs = self.model(inputs)
        self.val_running_loss += self.CELoss(outputs, labels).item()
        cls_preds = torch.argmax(outputs, dim=1)
        self.val_cm["TP"] += len(np.intersect1d(torch.where(cls_preds == 1)[0].cpu().detach(), torch.where(labels == 1)[0].cpu().detach()))
        self.val_cm["TN"] += len(np.intersect1d(torch.where(cls_preds == 0)[0].cpu().detach(), torch.where(labels == 0)[0].cpu().detach()))
        self.val_cm["FP"] += len(np.intersect1d(torch.where(cls_preds == 1)[0].cpu().detach(), torch.where(labels == 0)[0].cpu().detach()))
        self.val_cm["FN"] += len(np.intersect1d(torch.where(cls_preds == 0)[0].cpu().detach(), torch.where(labels == 1)[0].cpu().detach()))

    def end_batch(self):
        #learning rate schedule. during warmup it is linear from 0 to lr0
        #during training it implements linear polynomial decay
        if self.cur_step <= self.warmup_steps:
            self.current_lr += self.warmup_steprate
            self.optimizer.param_groups[0]["lr"] = self.current_lr
        elif (self.cur_step-self.warmup_steps)%self.update_lr_freq == 0:
            self.current_lr = max(((1 - (self.cur_step - self.warmup_steps)/(self.total_steps - self.warmup_steps))**.9)*self.lr0, self.lrf)
            self.optimizer.param_groups[0]["lr"] = self.current_lr
    
    def end_epoch(self):
        #collects overall metrics of epoch and prints out some key metrics for the current epoch
        self.avg_train_loss = self.train_running_loss/self.train_loader.__len__()
        self.train_rec = self.train_cm["TP"]/(self.train_cm["TP"] + self.train_cm["FN"])
        self.avg_val_loss = self.val_running_loss/self.val_loader.__len__()
        self.val_rec = self.val_cm["TP"]/(self.val_cm["TP"] + self.val_cm["FN"])

        if self.cur_step <= self.warmup_steps:
            print("\nWarmup Epoch {} Complete. Current LR is {}\n   Training:\n         CE Loss: {}, Recall: {}\n\n  Validation:\n       CE Loss: {}, Recall: {}\n".format(self.cur_epoch, round(self.current_lr, 6), round(self.avg_train_loss, 4), round(self.train_rec, 4), round(self.avg_val_loss, 4), round(self.val_rec, 4)))
        else:
            self.all_epochs.append(self.cur_epoch - self.warmup)
            self.all_train_loss.append(self.avg_train_loss)
            self.all_train_rec.append(self.train_rec)
            self.all_val_loss.append(self.avg_val_loss)
            self.all_val_rec.append(self.val_rec)
            print("\nEpoch {} Complete. Current LR is {}\n   Training:\n         CE Loss: {}, Recall: {}\n\n  Validation:\n          CE Loss: {}, Recall: {}\n".format(self.cur_epoch-self.warmup, round(self.current_lr, 6), round(self.avg_train_loss, 4), round(self.train_rec, 4), round(self.avg_val_loss, 4), round(self.val_rec, 4)))

        self.model_metric_eval()
    
    def model_metric_eval(self):
        #checks to see if the model has a new best validation loss and or validation recall, saving models labeled as such
        #also saves the model every epoch regardless as "last.pth"
        #saves metrics to pandas dataframe and exports to model save directory, after one epoch will start saving off plots in make_plots function
        if self.cur_step > self.warmup_steps:
            if self.avg_val_loss < self.bestLoss:
                self.bestLoss = self.avg_val_loss
                self.bestLoss_epoch = self.cur_epoch - self.warmup
                print("New best validation CE Loss model with a validation loss of {}\n".format(round(self.bestLoss, 4)))
                torch.save(self.model.state_dict(), os.path.join(self.logdir, "bestLoss.pth"))
            
            if self.val_rec > self.bestRec:
                self.bestRec = self.val_rec
                self.bestRec_epoch = self.cur_epoch - self.warmup
                print("New best validation recall model with a validation recall of {}\n".format(round(self.bestRec, 4)))
                torch.save(self.model.state_dict(), os.path.join(self.logdir, "bestRecall.pth"))
            
            newlog = pd.DataFrame(data={"Epoch": self.cur_epoch - self.warmup, "Current LR": round(self.current_lr, 6), "Train CE Loss": round(self.avg_train_loss, 4), "Train Rec": round(self.train_rec, 4), "Val CE Loss": round(self.avg_val_loss, 4), "Val Rec": round(self.val_rec, 4), "New Best Val Loss": bool(self.avg_val_loss==self.bestLoss), "New Best Val Rec": bool(self.val_rec==self.bestRec)}, index=[0])
            self.metricslog = pd.concat([self.metricslog if not self.metricslog.empty else None, newlog], ignore_index=False)
            self.metricslog.to_csv(os.path.join(self.logdir, "training_metrics_log.csv"), index=False)
            if self.cur_epoch - self.warmup > 1:
                make_plots(self.all_train_loss, self.all_val_loss, self.all_train_rec, self.all_val_rec, self.all_epochs, self.logdir)
            torch.save(self.model.state_dict(), os.path.join(self.logdir, "last.pth"))

if __name__ == "__main__":
    cfg = get_cfg()
    _ = Trainer(cfg)