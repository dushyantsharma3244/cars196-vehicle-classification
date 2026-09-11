# Cars196 Vehicle Classification

Fine-grained vehicle classification across 196 categories, comparing a CNN trained from scratch against a fine-tuned, pretrained ResNet50.

## Results

| Model | Approach | Accuracy |
|---|---|---:|
|  CNN | 	No pretrained weights | **2.75%** |
| ResNet50 | Transfer learning | **81.34%** |

Reported accuracy comes from a held-out split of the labeled training data (the official test set has no labels — it's meant for submission-based evaluation). Evaluation transforms are fully deterministic for reproducibility.

## Dataset

A public fine-grained vehicle dataset: 196 categories, ~16,000 images, ~30-40 images per class. Categories differ in subtle details (grille shape, headlight design, trim) rather than broad structural differences. Not included in this repo — only paths/labels are referenced; obtain the images from the original source and update paths before running the notebook.

## CNN

4 conv blocks (Conv → BatchNorm → ReLU, max pooling in the first three), adaptive max pooling, dropout-regularized classifier. Trained from random init at 128×128, horizontal flip augmentation, Adam (lr=1e-4), 25 epochs.

**2.75% accuracy.** The low accuracy highlights the difficulty of learning discriminative features from scratch for a fine-grained, 196-class classification problem. The custom CNN was trained from random initialization, without pretrained visual representations, making the task substantially harder. This result provides a useful baseline for comparison with the pretrained ResNet50, which achieved much higher accuracy through transfer learning.

## ResNet50 (Transfer Learning)

ImageNet-pretrained ResNet50, final layer replaced with a 196-class head, full network fine-tuned at 224×224. Random resized crop + horizontal flip augmentation, ImageNet normalization, Adam (lr=1e-4), 20 epochs.

**81.34% accuracy** — the gap versus the CNN comes down to pretrained features, not architecture complexity: starting from representations learned on millions of images beats learning vision from ~13,000 images across 196 classes.

## Tech stack

Python, PyTorch, torchvision, NumPy, Pandas, Matplotlib, scikit-learn

Download the dataset from its original source, update the image paths in the notebook, then run `Cars196_Vehicle_Classification.ipynb` top to bottom.

