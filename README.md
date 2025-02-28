# Project Setup and Usage Guide

## 1. Clone the Repository

After pulling the repository, follow these steps:

### 2. Setup the Environment

1. **Navigate to the `TristarAI_Takehome` directory:**

   ```bash
   cd TristarAI_Takehome
   ```

2. **Create a `data` directory** (this is ignored in the `.gitignore` file to avoid pushing large amounts of data):

   ```bash
   mkdir data
   ```

3. **Run the `build_docker.sh` script:**

   ```bash
   ./build_docker.sh
   ```

   - **Note:** You may need to modify the `OS` and `CUDA_VERSION` variables, as well as the base `nvidia/cuda` Docker image, for compatibility with your system.

4. **Run the `run_docker.sh` script:**

   ```bash
   ./run_docker.sh
   ```

---

## 2. Using the Docker Container

Once you're inside the Docker container, there are two main scripts for training and evaluation with an EfficientNet-B0 model:

- **Training script**: `train.sh`
- **Evaluation script**: `evaluate.sh`

### 2.1 Training the Model

To train a model, run the `train.sh` script, which requires one argument: the path to a config file containing training arguments.

**Config file options**:

Two pre-made config files are available in the `config` directory.

### 2.2 Available Arguments

Here’s a list of available arguments you can specify in the config file:

- `train_data_dir`: Path to the parent directory containing the `Malignant` and `Benign` subdirectories for training.
- `model_save_dir`: Path to the parent directory where models are saved during training, as well as various metrics.
- `experiment`: Subdirectory within `model_save_dir` to save results for that specific training run.
- `train_split`: Percentage of data to use for training vs. validation. Valid values: float (0, 1).
- `epochs`: Number of epochs to run after any warm-up epochs. Valid values: integer > 0.
- `warmup`: Number of warm-up epochs to allow the model to slowly rise to the initial learning rate (`lr0`). Valid values: integer >= 0.
- `bs`: Batch size. Valid values: integer > 0.
- `lr0`: Initial learning rate. Valid values: float (0, 1).
- `lrf`: Minimum final learning rate. Valid values: float (0, 1).
- `update_lr_freq`: Number of batches before updating the learning rate after warm-up. Valid values: integer > 0.
- `momentum`: Momentum for the optimizer. Valid values: float (0, 1).
- `weight_decay`: Weight decay for the optimizer. Valid values: float (0, 1).
- `augmentations`: Whether to use additional augmentations. See `params.py` for available augmentations. Valid values: boolean (`True` / `False`).

### 2.3 Training Outputs

During training, several files are saved:

- The config file used.
- The model with the best recall score, the best loss score, and the model from the final epoch.
- A CSV file containing metrics from each epoch.
- Two plots showing train and validation loss and recall metrics throughout training.

---

## 3. Evaluating the Model

To evaluate the model, run the `evaluate.sh` script with two arguments:

```bash
./evaluate.sh --model_dir <path_to_saved_models> --model <model_filename>
```

Where:
- `--model_dir`: Path to the directory where models were saved during training.
- `--model`: The specific model to use, e.g., `bestRecall.pth` or `bestLoss.pth`.

In the config file (which was copied to the evaluation directory during training), make sure to specify:

- `test_data_dir`: Path to the parent directory containing the `Malignant` and `Benign` subdirectories for testing.

### 3.1 Evaluation Outputs

After evaluation, the following files are saved:

1. **`eval_results.png`**: Base confusion matrix from non-postprocessed inference. Includes accuracy, precision, recall, and F1 score.
2. **`..._post_processed_confident.png`**: Post-processing was applied to determine which inferences were most confident.
3. **`..._post_processed_rediagnoses.png`**: These are the inferences deemed unconfident and should seek a second opinion.

---

## 4. Notes on Training and Evaluation

### 4.1 Why Recall is Used for Model Selection

Recall was prioritized over accuracy because the use case involves cancer detection. The worst-case scenario in this case is false negatives (a patient with cancer not being diagnosed). Minimizing this case is crucial, which is why recall (which is sensitive to false negatives) is the most important metric.

### 4.2 Post-processing During Evaluation

During inference, a "confidence score" was derived to provide more meaningful insights than the raw model output. A sigmoid function was applied to the model's output, and the confidence values were analyzed for both correct and incorrect predictions.

- **Correct predictions**:
  - Min: 0.4990, Max: 1.0, Mean: 0.9650, Std: 0.0761
- **Incorrect predictions**:
  - Min: 0.4899, Max: 0.9987, Mean: 0.7747, Std: 0.1531

The mean value of incorrect scores was found to be approximately 2.5 standard deviations away from correct scores. A threshold of `0.775` was chosen, which effectively keeps around 98% of correct predictions while removing about half of the incorrect predictions.

In practice, rejected cases could be sent for a second opinion, thus improving recall.

### 4.3 Test Set Usage To Determine Post-processing

Although using the test data to improve performance is generally not recommended, the full test set was used for evaluation in this case. In practice, I would consider using a sectioned off portion of the training/validation data for testing to avoid bias.

---

## 5. TODO

- Reject invalid values in config arguments.
