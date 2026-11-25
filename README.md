
# ✈️ Flight Delay Prediction

## About the Project

I'm working on a machine learning model to predict whether a flight will depart at least 15 minutes late.

The goal is to make predictions **two hours before the scheduled departure**, using only information available at that moment.

I'm using historical U.S. flight data from January 2025, provided by the [Bureau of Transportation Statistics (BTS)](https://www.transtats.bts.gov/).

Through this project, I want to understand the factors associated with flight delays and build a complete machine learning workflow.

## 📊 Dataset

The original dataset contains **539,747 flights and 15 columns**.

After excluding cancelled flights and observations with missing target values, I kept **523,435 flights**.

The target variable is `DEP_DEL15`:

- `0` — departure delay below 15 minutes
- `1` — departure delay of 15 minutes or more

| Class | Flights | Percentage |
|---|---:|---:|
| Not delayed | 428,003 | 81.77% |
| Delayed | 95,432 | 18.23% |

The dataset is imbalanced, so accuracy alone is not enough to evaluate model performance.

## 🔍 Exploratory Data Analysis

I explored the dataset to understand missing values, potential data leakage, and the relationship between flight characteristics and departure delays.

### Key Findings

| Feature | Observation |
|---|---|
| Origin and destination | Extreme delay rates often come from airports with small sample sizes. |
| Departure time | Delay rates increase from 8.07% at 06:00 to 25.01% at 20:00. |
| Distance | The relationship between distance and delays is not clearly linear. |
| Day of week | Sunday and Monday have the highest delay rates, while Wednesday has the lowest. |
| Airline | Delay rates vary across the 14 airlines. |
| Evening flights | 13 out of 14 airlines have higher delay rates for evening flights. |

One of the most interesting patterns I found was the difference between evening and non-evening flights.

| Departure period | Flights | Delay rate |
|---|---:|---:|
| Before 18:00 | 414,245 | 16.80% |
| 18:00–23:59 | 109,190 | 23.68% |

These findings show associations, not necessarily causal relationships.

## Data Leakage Prevention

Since predictions are made before departure, I excluded features that would not be available at the prediction point:

- `DEP_TIME`
- `DEP_DELAY`
- `CANCELLED`
- `CANCELLATION_CODE`
- `DIVERTED`

I also avoided replacing missing target values with zero.

The current model focuses on operated flights rather than predicting cancellations.

## ⚙️ Data Preprocessing

### Chronological Train-Validation Split

Since flight data is time-dependent, I used a chronological split instead of a random split.

| Dataset | Period | Flights | Delay rate |
|---|---|---:|---:|
| Training | January 1–24 | 404,662 | 20.22% |
| Validation | January 25–31 | 118,773 | 11.46% |

This allows the model to learn from historical data and be evaluated on a later period.

### Feature Engineering

I validated scheduled departure and arrival times, converted HHMM values into minutes since midnight, and applied cyclical encoding using a 1,440-minute cycle.

The original time columns were replaced with:

- `DEP_TIME_SIN`
- `DEP_TIME_COS`
- `ARR_TIME_SIN`
- `ARR_TIME_COS`

This preserves the cyclical nature of time without losing minute-level information.

### Feature Transformation

| Feature group | Features | Transformation |
|---|---|---|
| Categorical | ORIGIN, DEST, OP_UNIQUE_CARRIER, DAY_OF_WEEK | OneHotEncoder |
| Numerical | DISTANCE | StandardScaler |
| Cyclical | Departure and arrival sine/cosine features | Passthrough |

I fitted the encoder and scaler exclusively on training data and reused the learned parameters to transform the validation set.

Using `ColumnTransformer`, I combined all transformations into a single representation containing **684 features**.

| Dataset | Before | After |
|---|---|---|
| Training | (404662, 9) | (404662, 684) |
| Validation | (118773, 9) | (118773, 684) |

## 🤖 Baseline Model — Logistic Regression

I chose Logistic Regression as my first model because it is relatively simple, interpretable, and provides a reference for future experiments.

I combined the ColumnTransformer and classifier into a Scikit-learn Pipeline.

### Model Configuration

| Hyperparameter | Value |
|---|---|
| C | 1.0 |
| Solver | lbfgs |
| Max iterations | 1000 |
| Class weight | None |

The model completed optimization in 80 iterations and learned 684 coefficients and one intercept.

### Baseline Results

Using the default classification threshold of 0.5:

| Metric | Score |
|---|---:|
| Accuracy | 88.41% |
| Precision | 20.85% |
| Recall | 0.40% |
| F1-score | 0.0078 |

Although the accuracy was high, the model detected only **54 out of 13,609 delayed flights**.

A simple majority-class baseline would achieve approximately 88.54% accuracy, showing why accuracy can be misleading in imbalanced classification problems.

## 📈 Classification Threshold Analysis

Before changing the model, I wanted to understand whether the default classification threshold was contributing to the low recall.

I analyzed the predicted probabilities and found that most values were below 0.5. The median predicted probability was approximately 0.195.

I tested classification thresholds from 0.05 to 0.50 without retraining the model.

### Threshold Comparison

| Threshold | Accuracy | Precision | Recall | F1-score |
|---:|---:|---:|---:|---:|
| 0.05 | 12.02% | 11.47% | 99.40% | 0.2057 |
| 0.10 | 21.18% | 11.97% | 92.49% | 0.2119 |
| 0.15 | 38.03% | 13.05% | 77.85% | 0.2235 |
| **0.20** | **54.18%** | **14.18%** | **59.34%** | **0.2288** |
| 0.25 | 67.34% | 15.26% | 40.66% | 0.2220 |
| 0.30 | 76.87% | 16.24% | 24.51% | 0.1954 |
| 0.50 | 88.41% | 20.85% | 0.40% | 0.0078 |

The best tested threshold was **0.20**, based on the validation F1-score.

### Confusion Matrix Comparison

| Metric | Threshold 0.5 | Threshold 0.2 |
|---|---:|---:|
| True Negatives | 104,959 | 56,275 |
| False Positives | 205 | 48,889 |
| False Negatives | 13,555 | 5,534 |
| True Positives | 54 | 8,075 |

Reducing the threshold improved recall and F1-score significantly, but also generated many more false positives.

I visualized the predicted probability distribution and compared the evaluation metrics across different thresholds to better understand the precision-recall trade-off.

The results suggest that threshold adjustment alone is not enough to achieve reliable predictions.

## 🛠️ Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## 📁 Repository Structure

```
flight-delay-prediction/
├── .gitignore
├── 01_data_exploration.ipynb
└── README.md
```

## Project Progress

- [x] Problem definition
- [x] Data cleaning and missing value analysis
- [x] Exploratory Data Analysis
- [x] Data leakage prevention
- [x] Chronological train-validation split
- [x] Feature selection
- [x] Cyclical Encoding
- [x] One-Hot Encoding
- [x] Numerical feature scaling
- [x] ColumnTransformer
- [x] Machine Learning Pipeline
- [x] Baseline Logistic Regression
- [x] Confusion Matrix
- [x] Accuracy, Precision, Recall and F1-score
- [x] Predicted probability analysis
- [x] Classification threshold tuning
- [x] Threshold performance visualization
- [ ] ROC-AUC and Precision-Recall analysis
- [ ] Class imbalance experiments
- [ ] Model comparison
- [ ] Hyperparameter tuning
- [ ] Final test set evaluation

## Running the Notebook

Install the required packages:

```bash
pip install pandas numpy matplotlib scikit-learn jupyter
```

Download the January 2025 flight data from the [BTS website](https://www.transtats.bts.gov/) and save it as `flightinfo.csv` in the project directory.

The dataset is not included in the repository.

Open the notebook:

```bash
jupyter notebook 01_data_exploration.ipynb
```

## What's Next?

My next step is to evaluate the model using ROC-AUC and Precision-Recall metrics to better understand its ability to distinguish delayed from non-delayed flights.

I also want to experiment with class weighting, additional features, and more complex classification models.

Since the exploratory analysis covered the entire January dataset and the validation set was used for threshold selection, I plan to evaluate the final model on an independent future period.

This project is still in progress.
