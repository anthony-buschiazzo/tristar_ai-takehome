import argparse
import os
import yaml
from tqdm import tqdm

import torch
from torch import nn
from torchvision import models
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

from params import GetParams
from data_loader import get_eval_loader

def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model_dir", type=str, default="/home/tristarAI/takehome_ws/trained_models/default_config", help=("File path to directory containing model to evaluate on."))
    parser.add_argument("--model", type=str, default="bestAccuracy.pth", help=("File name of weights to be used for evaluation."))
    args = parser.parse_args()
    args = {
        "model_dir": args.model_dir,
        "model": args.model
    }
    return args

class Evaluator():
    def __init__(self, model_dir, model):
        
        self.savefile = os.path.join(model_dir, os.path.splitext(model)[0])
        
        with open(os.path.join(model_dir, "config.yaml")) as stream:
            cfg = yaml.safe_load(stream)
        
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        print("\nInferencing on {}.\n".format(self.device))


        #gather data parameters and create dataloaders
        eval_params = GetParams(cfg, False).get_eval_params()

        self.eval_loader = get_eval_loader(eval_params)

        #initialize efficientnet model and load in weights from previous trainings
        self.model = models.efficientnet_b0(weights = None)
        self.model.classifier[1] = nn.Linear(in_features = 1280, out_features = 2)
        weights = torch.load(os.path.join(model_dir, model))
        if self.device == "cuda":
            self.model.load_state_dict(weights)
        else:
            self.model.load_state_dict(weights, map_location=torch.device("cpu"))
        self.model.train(False)

        #creates confusion matrices to collect statistics during inference
        self.cm = np.zeros((2, 2), dtype=np.int32)
        self.cm_postproc_conf = np.zeros((2, 2), dtype=np.int32)
        self.cm_postproc_rediagnose = np.zeros((2, 2), dtype=np.int32)
        self.sigmoid = nn.Sigmoid()

        self.infer()
        self.gather_metrics()
    
    def infer(self):
        #function for inferencing on data and filling in confusion matrix array
        with torch.no_grad():
            for i, data in enumerate(tqdm(self.eval_loader), 0):
                inputs, label = data
                label = label[0].item()

                outputs = self.model(inputs)

                pred = torch.argmax(outputs).item()
                
                self.cm[pred, label] += 1

                #post processing on output to determine "good" vs "bad" predictions
                if torch.max(self.sigmoid(outputs)).item() > 0.775:
                    self.cm_postproc_conf[pred, label] += 1
                else:
                    self.cm_postproc_rediagnose[pred, label] += 1
    
    def gather_metrics(self):
        #calculates true positives, true negatives, etc.. to be used to calculate accuracy, precision, recall, and f1-score
        #creates a confusion matrices and adds other previous metrics mentioned to the plot. saves to where trained model is located.
        tp = self.cm[1, 1]
        tn = self.cm[0, 0]
        fp = self.cm[1, 0]
        fn = self.cm[0, 1]

        acc = (tp+tn)/(tp+fp+fn+tn)
        prec = tp/(tp+fp)
        rec = tp/(tp+fn)
        f1 = 2*(prec*rec)/(prec+rec)

        plt.figure(figsize=(10, 7))
        heatmap = sn.heatmap(self.cm, annot=True, cbar=False, fmt="d")
        heatmap.set_xticklabels(["Benign", "Malignant"])
        heatmap.set_yticklabels(["Benign", "Malignant"], rotation=90)
        heatmap.set_ylabel("Ground Truth")
        heatmap.set_ylabel("Predictions")
        figure = heatmap.get_figure()
        plt.text(0, 0, "Accuracy: {}        Precision: {}        Recall: {}        F1-Score: {}".format(round(acc, 4), round(prec, 4), round(rec, 4), round(f1, 4)))
        plt.savefig(self.savefile+"_eval_results.png")
        plt.close()


        tp = self.cm_postproc_conf[1, 1]
        tn = self.cm_postproc_conf[0, 0]
        fp = self.cm_postproc_conf[1, 0]
        fn = self.cm_postproc_conf[0, 1]

        acc = (tp+tn)/(tp+fp+fn+tn)
        prec = tp/(tp+fp)
        rec = tp/(tp+fn)
        f1 = 2*(prec*rec)/(prec+rec)

        plt.figure(figsize=(10, 7))
        heatmap = sn.heatmap(self.cm_postproc_conf, annot=True, cbar=False, fmt="d")
        heatmap.set_xticklabels(["Benign", "Malignant"])
        heatmap.set_yticklabels(["Benign", "Malignant"], rotation=90)
        heatmap.set_ylabel("Ground Truth")
        heatmap.set_ylabel("Predictions")
        figure = heatmap.get_figure()
        plt.text(0, 0, "Accuracy: {}        Precision: {}        Recall: {}        F1-Score: {}".format(round(acc, 4), round(prec, 4), round(rec, 4), round(f1, 4)))
        plt.savefig(self.savefile+"_eval_post_processed_results_confident.png")
        plt.close()


        tp = self.cm_postproc_rediagnose[1, 1]
        tn = self.cm_postproc_rediagnose[0, 0]
        fp = self.cm_postproc_rediagnose[1, 0]
        fn = self.cm_postproc_rediagnose[0, 1]

        acc = (tp+tn)/(tp+fp+fn+tn)
        prec = tp/(tp+fp)
        rec = tp/(tp+fn)
        f1 = 2*(prec*rec)/(prec+rec)

        plt.figure(figsize=(10, 7))
        heatmap = sn.heatmap(self.cm_postproc_rediagnose, annot=True, cbar=False, fmt="d")
        heatmap.set_xticklabels(["Benign", "Malignant"])
        heatmap.set_yticklabels(["Benign", "Malignant"], rotation=90)
        heatmap.set_ylabel("Ground Truth")
        heatmap.set_ylabel("Predictions")
        figure = heatmap.get_figure()
        plt.text(0, 0, "Accuracy: {}        Precision: {}        Recall: {}        F1-Score: {}".format(round(acc, 4), round(prec, 4), round(rec, 4), round(f1, 4)))
        plt.savefig(self.savefile+"_eval_post_processed_results_rediagnoses.png")
        plt.close()

        

        


if __name__ == "__main__":
    args = get_args()
    _ = Evaluator(**args)
    