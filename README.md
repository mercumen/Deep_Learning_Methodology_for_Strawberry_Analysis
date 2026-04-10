# Deep Learning Methodology for Strawberry Analysis

## Project Overview
This project aims to classify strawberry ripeness levels using deep learning techniques. The main goal is to compare different methods and analyze how they affect model performance. The project focuses especially on overfitting, generalization, and performance improvement across multiple experiments.

## Dataset
A multi-class strawberry ripeness dataset was used in this project. The images are grouped into three classes:
- Unripe
- Ripe
- Overripe

## Methods Used
The following methods were applied and compared:
- Baseline CNN
- L2 Regularization
- Dropout
- Early Stopping
- Data Augmentation

## Why These Methods?
These methods were selected to improve model performance and reduce overfitting.

- Baseline CNN was used as the reference model.
- L2 Regularization was applied to make training more stable and reduce excessively large weights.
- Dropout was tested to reduce memorization by randomly disabling neurons during training.
- Early Stopping was used to stop training when validation performance stopped improving.
- Data Augmentation was applied to increase data diversity and improve generalization.

## Results
The models were compared based on test accuracy:

- Baseline: 48.25%
- L2: 51.75%
- Dropout: 43.86%
- Early Stopping: 80.80%
- Augmentation: 69.93%

## Analysis
The results show that regularization methods affected the model in different ways.

- The baseline model provided limited performance.
- L2 Regularization slightly improved the results and gave a more stable training process.
- Dropout reduced model performance, which may indicate excessive regularization.
- Early Stopping achieved the best result by preventing overfitting and stopping training at the optimal point.
- Data Augmentation also improved performance by helping the model generalize better.

Validation loss analysis showed that loss started to increase after certain epochs for some experiments, which indicates overfitting. Early Stopping helped reduce this problem.

## Conclusion
Among all tested methods, Early Stopping gave the best performance. It provided the best balance between learning and generalization. Data Augmentation also produced strong results. Overall, the project showed that choosing the right regularization method can significantly improve classification performance.

## Project Structure
```text
experiments/
results/
src/
.gitignore
README.md
app.py
requirements.txt
