Domain: Healthcare

## Part 1 findings

### R1.1 — Estimation and hypothesis testing

The estimated mean care-plan cost is INR 8.9382 thousand. The z-statistic is 2.8930, which exceeds the two-tailed critical value of 1.96. Therefore, we reject the null hypothesis that the population mean care-plan cost is INR 8,000 at the 5% significance level.

A Type I error would mean concluding that the average care-plan cost differs from INR 8,000 when the true population average is actually INR 8,000.

The estimated difference is 0.9382 thousand, approximately INR 938.18 above the planning assumption. This is about 11.7% of INR 8,000 and could be practically meaningful for budgeting.

### R1.2 — Required sample size

The required sample size is 1,293 records, compared with the 800 records currently available. This means 493 additional records would be needed to achieve the requested margin of error of INR 500 at 95% confidence, using the estimated sample standard deviation.

### R1.3 — Gradient descent

The gradient descent slope gap is approximately 0.0000 (actual value: 3.33 × 10⁻¹⁶). Gradient descent reached practically the same line as the closed-form solution because repeated updates using the specified learning rate minimized the cost function and converged to the same regression coefficients.

### R1.4 — Residual funnel and log transformation

The raw residual standard deviation ratio is 3.7525, indicating a clear funnel pattern because the upper group's residual spread is much larger than the lower group's.

After the log transformation, the ratio decreases to 1.0396. This indicates that the log-log model substantially reduces the funnel pattern and produces much more similar residual variation between the two groups.

### R1.5 — Multiple regression

The predictors with absolute coefficient values above 0.05 are:

- Average bill (log X4): 1.1576
- Missed visit count (X2): 0.0940

- # PharmEasy Healthcare Decision Intelligence

Domain: Healthcare

## Part 2 findings

### 2.1 Stratified Train-Test Split

The dataset was divided using stratified sampling to preserve the proportion of positive and negative outcomes.

- Training rows: 560
- Training positive cases: 119
- Testing rows: 240
- Testing positive cases: 51

### 2.2 Logistic Regression

Logistic Regression was implemented from scratch using gradient descent.

- Learning rate: 0.1
- Training iterations: 1000
- Loss at epoch 0: 0.6931
- Loss at epoch 1000: 0.3700

The decreasing loss indicates that the model learned from the training data.

### 2.3 Model Evaluation

At threshold 0.5:

- True Positives: 24
- True Negatives: 169
- False Positives: 20
- False Negatives: 27
- Accuracy: 0.8042
- Precision: 0.5455
- Recall: 0.4706
- F1-score: 0.5053

At threshold 0.3:

- True Positives: 38
- True Negatives: 145
- False Positives: 44
- False Negatives: 13
- Accuracy: 0.7625
- Precision: 0.4634
- Recall: 0.7451
- F1-score: 0.5714

Lowering the threshold increased recall and F1-score but also increased false positives.

### 2.4 Odds Ratios

The calculated odds ratios were:

- X1: 0.5739
- X2: 1.7341
- X3: 2.4629
- X4: 0.8642
- X5: 0.8056
- X6: 1.2646

An odds ratio greater than 1 indicates a positive association with the outcome, while a value below 1 indicates a negative association, holding the other model variables constant.

### 2.5 K-Nearest Neighbours

KNN was implemented from scratch.

- Best K: 5
- Cross-validation F1-score: 0.4541

Test performance:

- True Positives: 17
- False Positives: 24
- True Negatives: 165
- False Negatives: 34
- Accuracy: 0.7583
- Precision: 0.4146
- Recall: 0.3333
- F1-score: 0.3696

Logistic Regression performed better than KNN on the test F1-score.

### 2.6 Threshold and Cost Analysis

The cost-sensitive threshold selected was 0.15.

- Training cost at selected threshold: INR 12,150
- Test cost at selected threshold: INR 6,050
- Test cost at threshold 0.5: INR 11,800
- Cost when nobody is flagged: INR 20,400

This demonstrates that threshold selection should consider the operational cost of missed positive cases and unnecessary flags, rather than accuracy alone.

## Part 3 findings

### 3.1 K-Means Clustering

The elbow method was used to select the number of clusters.

| Number of clusters | WCSS |
|---|---:|
| K = 2 | 2262.6753 |
| K = 3 | 1602.8471 |
| K = 4 | 1494.7732 |
| K = 5 | 1323.7157 |
| K = 6 | 1278.1691 |

The selected elbow value was K = 3.

The resulting segment sizes were:

- Segment A: 322 records
- Segment B: 255 records
- Segment C: 223 records

### 3.2 Segment Analysis

| Segment | Size | Outcome rate | Mean value |
|---|---:|---:|---:|
| A | 322 | 0.0186 | 3.5976 |
| B | 255 | 0.2353 | 17.1248 |
| C | 223 | 0.4664 | 7.2883 |

Interpretation:

- Segment A has a low outcome rate.
- Segment B has a moderate outcome rate and the highest mean value.
- Segment C has the highest outcome rate and may require greater operational attention.

These segments can help identify different groups for further investigation and scheduling decisions.

### 3.3 Hierarchical Clustering

Three linkage methods were compared using the supplied observations.

| Linkage method | Clusters at selected cut |
|---|---:|
| Single | 4 |
| Complete | 2 |
| Average | 2 |

Single linkage forms clusters based on the minimum distance between observations. Complete linkage uses the maximum distance, while average linkage uses the average pairwise distance.

The results demonstrate that the selected linkage method affects the resulting cluster structure.

### 3.4 Principal Component Analysis (PCA)

PCA was performed on five standardized features.

| Component | Eigenvalue | Explained variance | Cumulative variance |
|---|---:|---:|---:|
| PC1 | 2.6136 | 52.27% | 52.27% |
| PC2 | 1.1828 | 23.66% | 75.93% |
| PC3 | 0.5114 | 10.23% | 86.16% |
| PC4 | 0.4055 | 8.11% | 94.27% |
| PC5 | 0.2867 | 5.73% | 100.00% |

Three principal components are sufficient to explain more than 80% of the total variance.

PC1 explains the largest proportion of variance, followed by PC2. PCA reduces dimensionality while retaining most of the information in the original features.

## Part 4 findings

### 4.1 Decision Tree Split Analysis

The root node entropy was 0.7462 and the root Gini impurity was 0.3347.

The four candidate splits were evaluated.

| Split | Information gain | Gini gain |
|---|---:|---:|
| S1 | 0.0110 | 0.0052 |
| S2 | 0.0033 | 0.0015 |
| S3 | 0.1046 | 0.0494 |
| S4 | 0.1252 | 0.0606 |

S4 (X1 <= 24) was selected as the best split using both information gain and Gini gain.

A fully grown decision tree can overfit the training data by creating very small or pure leaves. A maximum-depth limit or minimum-samples-per-leaf stopping rule can help control overfitting.

### 4.2 Model Voting

Five models were evaluated using the test data.

| Model | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Logistic Regression | 0.4634 | 0.7451 | 0.5714 |
| KNN | 0.4146 | 0.3333 | 0.3696 |
| Decision Stump | 0.4333 | 0.7647 | 0.5532 |
| Hard Voting | 0.4375 | 0.6863 | 0.5344 |
| Soft Voting | 0.4368 | 0.7451 | 0.5507 |

Logistic Regression achieved the highest F1-score of 0.5714.

The decision stump achieved the highest recall of 0.7647. However, both voting approaches achieved lower F1-scores than Logistic Regression. Therefore, combining the models did not improve the overall F1-score in this experiment.

### 4.3 AdaBoost

AdaBoost was implemented using two weak classifiers and eight observations.

| Metric | Result |
|---|---:|
| Round 1 weighted error | 0.2500 |
| Round 1 alpha | 0.5493 |
| Round 2 weighted error | 0.1667 |
| Round 2 alpha | 0.8047 |
| Ensemble accuracy | 0.7500 |

After the first round, points 3 and 7 were misclassified. Their combined normalized weight became 0.50.

The second weak classifier received a larger alpha because its weighted error was lower than the first classifier's error. This means AdaBoost assigned greater influence to the second classifier.

### 4.4 Out-of-Bag Fraction

The theoretical out-of-bag fraction was calculated as:

OOB fraction = (1 - 1/560)^560

Result: 0.3676, approximately 36.76%.

This represents the expected fraction of training observations not selected in a bootstrap sample of 560 observations.

### 4.5 Neural Network Architecture and Parameter Analysis

#### MLP and CNN parameter counts

| Layer | Parameters |
|---|---:|
| MLP hidden layer 1 | 12,560 |
| MLP hidden layer 2 | 272 |
| MLP output layer | 170 |
| CNN convolution 1 | 156 |
| CNN convolution 2 | 2,416 |
| CNN dense layer 1 | 12,832 |
| CNN dense layer 2 | 330 |

- Total MLP parameters: 13,002
- Total CNN parameters: 15,734

#### CNN Model A output shapes

| Layer | Output shape |
|---|---|
| Input | 28 × 28 × 1 |
| Convolution 1 | 28 × 28 × 6 |
| ReLU | 28 × 28 × 6 |
| Max pooling 1 | 14 × 14 × 6 |
| Convolution 2 | 10 × 10 × 16 |
| ReLU | 10 × 10 × 16 |
| Max pooling 2 | 5 × 5 × 16 |
| Flatten | 400 |
| Dense layer | 32 |
| Output layer | 10 |

#### Fully connected versus convolutional layer

For a 128 × 128 × 3 input:

- Fully connected layer with 1024 neurons: 50,332,672 parameters.
- Convolutional layer with 64 filters of size 3 × 3: 1,792 parameters.

Convolution uses local connectivity and shared filter weights, resulting in substantially fewer parameters than a fully connected layer.

#### RNN architecture

For an RNN with input size 64, hidden size 128, and output size 16:

- W_xh: 64 × 128
- W_hh: 128 × 128
- W_hy: 128 × 16
- b_h: 128
- b_y: 16
- Total parameters: 26,768

The RNN parameter count does not increase with sequence length because the same weights are reused at every time step.

Vanishing gradients occur when gradients become very small during backpropagation through long sequences. LSTM and GRU architectures help address this issue through gating mechanisms. Skip connections can also improve gradient flow in suitable network architectures.

For a 10-class classification problem, a softmax output activation with categorical cross-entropy loss is an appropriate choice.

## Model-choice memo

To: Head of Scheduling Operations Team

Subject: Recommended model for scheduling decision support

I recommend Logistic Regression as the initial structured-data model for scheduling decision support.

On the test set, Logistic Regression achieved an F1-score of 0.5714 and recall of 0.7451 at a threshold of 0.3. It performed better in F1-score than KNN (0.3696), the decision stump (0.5532), hard voting (0.5344), and soft voting (0.5507).

For cost-sensitive operation, the Task 2.8 analysis selected a threshold of 0.15, with a test cost of INR 6,050. This compares with INR 11,800 at threshold 0.5 and INR 20,400 when nobody is flagged. The alternative models' costs have not been calculated using the same cost rule, so a direct cost comparison is not available.

One important risk is that the model was evaluated on a limited synthetic dataset. Its performance may not generalize to real scheduling data.

Model A, the CNN architecture, is worth exploring as an offline prototype only if relevant image data becomes available. It should not delay deployment testing of the structured Logistic Regression model.

The model should undergo validation on real operational data and human review before being used for actual scheduling decisions.

## Limitations

1. The project uses a synthetic dataset, so its results may not represent real PharmEasy operational patterns.
2. The dataset contains a limited number of observations, which may affect model generalization.
3. The clustering results depend on feature selection, scaling, and the chosen number of clusters.
4. Model performance may change when applied to new data or different operating conditions.
5. The cost analysis depends on the assumed cost values and selected decision threshold.
6. Neural network architectures were analyzed theoretically; they were not trained or evaluated on a real image or text dataset.
- Copay burden ratio (X3): 0.1067
- Chronic condition flag (X6): 0.1518

The multiple regression adjusted R-squared is 0.7403, compared with 0.6561 for the simple log-log regression. The higher adjusted R-squared indicates that the additional predictors improve the model's explanatory performance, even after accounting for the number of predictors.# Healthcare
