### California Housing — Linear & Polynomial Regression Analysis###
## Baseline Linear Regression

**Model**: Linear Regression (SGD)
**Features**: Original numeric features (no feature engineering)
```zsh
Training MSE : 4,859,165,344.50
Test MSE     : 5,082,316,308.53
Training RMSE: 69,707.71
Test RMSE    : 71,290.36
```

This model serves as the baseline for all further experiments.

# Ridge Regression (No Validation)

**Model**: Ridge Regression
**Alpha**: Default (α = 1.0)
```
Training MSE : 4,811,134,409.29
Test MSE     : 5,068,545,657.22
Training RMSE: 69,362.34
Test RMSE    : 71,193.72
```

A small improvement over baseline was observed, but this result was based on a single train–test split.

# Ridge Regression (With Validation)
```
Best α (validation): 1000
Test MSE           : 7,086,006,227.02
```
# Observations

* Introduced a validation split and tuned α over multiple values
* The earlier improvement disappeared after proper validation
* Very large α values sometimes appeared optimal due to random split noise
* Decreasing α consistently reduced MSE

# Interpretation

* The model is bias-limited, not variance-limited
* The dataset is mostly linear and not strongly overfitting
* Ridge adds bias without sufficiently reducing variance
* Improvements from Ridge were not stable across splits

**Conclusion:** Ridge does not reliably improve generalization for this dataset.

## Feature Engineering + Ridge Regression
**Engineered Features**
```
rooms_per_household

bedrooms_per_rooms

density

Validated α = 10.0

Training RMSE: 68,210.03
Test RMSE    : 74,183.35

```
Training error decreased, but test error increased — indicating overfitting.

# Polynomial Regression Experiments
**Engineered Features + All Polynomial Features**
```
Validated α = 100

Training RMSE: 65,181.66
Test RMSE    : 71,895.32

```
Large reduction in training error, but *no improvement over baseline test RMSE.*

**Selected Polynomial Features Only**
```
(median_income², latitude × longitude)

Validated α = 10.0

Training RMSE: 67,178.07
Test RMSE    : 75,748.13

```
Reducing polynomial scope increased bias and worsened generalization.

**Selected + All Polynomial Features**
```
Validated α = 10.0

Training RMSE: 63,050.96
Test RMSE    : 72,178.24
```

Lowest training error observed, but *test error remained similar* — indicating high variance.

**Final Conclusion**
I tested polynomial regression with selective feature expansion and ridge regularization.
While training error consistently decreased, validation and test error did not improve, indicating increased variance and limited nonlinear signal in the dataset.
This suggests that linear models are close to optimal for the California Housing dataset, and that tree-based models are better suited.

**Key Takeaways**

* Linear regression already performs near optimally
* Ridge regularization provides limited benefit
* Polynomial features reduce bias but increase variance
* Tree-based models are more appropriate for this dataset

