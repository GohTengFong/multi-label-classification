# Multi-Label Classification

This Jupyter Notebook solves a multi-label classification problem for real-world images. These images are a subset of the famous [YCB dataset](https://www.ycbbenchmarks.com/).

## Dataset Details
The dataset contains 21 everyday objects.

| Label                     | Label ID |
|---------------------------|----------|
| 002_master_chef_can       | 0        |
| 003_cracker_box           | 1        |
| 004_sugar_box             | 2        |
| 005_tomato_soup_can       | 3        |
| 006_mustard_bottle        | 4        |
| 007_tuna_fish_can         | 5        |
| 008_pudding_box           | 6        |
| 009_gelatin_box           | 7        |
| 010_potted_meat_can       | 8        |
| 011_banana                | 9        |
| 019_pitcher_base          | 10       |
| 021_bleach_cleanser       | 11       |
| 024_bowl                  | 12       |
| 025_mug                   | 13       |
| 035_power_drill           | 14       |
| 036_wood_block            | 15       |
| 037_scissors              | 16       |
| 040_large_marker          | 17       |
| 051_large_clamp           | 18       |
| 052_extra_large_clamp     | 19       |
| 061_foam_brick            | 20       |

## Evaluation Metric
The evaluation metric that is used is the F1-Score. In a simple Binary Classification problem, the F1-Score is denoted by the following formula:

$$
\frac{2 * \text{True Positive}}{2 * \text{True Positive} + \text{False Positive} + \text{False Negative}}
$$

However, in a Multi-Label Classification problem, the $\text{True Positive}$, $\text{False Positive}$ and $\text{False Negative}$ are computed per-label. Different averaging strategies (**Micro**, **Macro**, and **Weighted**) yield different F1-scores. For details, see [this guide on F1-score averaging methods](https://iamirmasoud.com/2022/06/19/understanding-micro-macro-and-weighted-averages-for-scikit-learn-metrics-in-multi-class-classification-with-example/).

## Previous Iterations
Before settling on the final model, 3 prior iterations were made. The first 2 explored pre-built models in PyTorch.

1. ResNet50 (0.68761)
    - a popular deep convolutional neural network (CNN) architecture
2. SwinTransformer (0.26534)
    - a hierarchical vision transformer architecture
    - introduces local window attention and shifted window mechanisms
3. Self-Coded CNN (0.39323)

## Model's Performance
The final model used a series of convolutional layers for feature extraction from images.

### Convolutional Layers
1. It performs convolution (`Conv2d`) and expands the feature depth
2. `LeakyReLU` is used to introduce non-linearity
3. `BatchNorm2d` is used to normalise the data
4. `MaxPool2d` is used to reduce the dimensionality of the while retaining the local patterns

The data is then flattened to be pushed through the linear layers.

### Linear Layers
1. At each stage, the number of output neurons decreases, until 21 (to classify into 1 of the 21 classes)
2. `ReLU` is used to allow the neural networks to sieve out parts of a curve to combine with other parts of other curves thus, forming a new complex curve
3. `Dropout(0.5)` is used to randomly drop 50% of the neurons during training to prevent overfitting

The model boasted an F1-Score of 0.99514. This indicates that it achieved high precision and recall with very few false positives (misclassifications) or or false negatives (missed detections).