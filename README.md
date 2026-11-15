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
- Copay burden ratio (X3): 0.1067
- Chronic condition flag (X6): 0.1518

The multiple regression adjusted R-squared is 0.7403, compared with 0.6561 for the simple log-log regression. The higher adjusted R-squared indicates that the additional predictors improve the model's explanatory performance, even after accounting for the number of predictors.# Healthcare
