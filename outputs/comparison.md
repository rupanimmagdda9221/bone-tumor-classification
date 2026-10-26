# Model comparison (test set)

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | Missed malignant | False alarms |
|---|---|---|---|---|---|---|---|
| ResNet50 + augmentation (proposed) | 0.890 | 0.692 | 0.706 | 0.699 | 0.945 | 15 / 51 | 16 / 230 |
| ResNet50, no augmentation | 0.875 | 0.674 | 0.608 | 0.639 | 0.901 | 20 / 51 | 15 / 230 |

Precision, recall and F1 are for the malignant class. All runs use the same stratified test set (281 images).
