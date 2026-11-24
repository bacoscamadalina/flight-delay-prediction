# ✈️ Flight Delay Prediction

## About the Project

I'm working on a machine learning model to predict whether a flight will depart at least 15 minutes late.

The goal is to make predictions **two hours before the scheduled departure**, using only information available at that moment.

I'm using historical U.S. flight data from January 2025, provided by the [Bureau of Transportation Statistics (BTS)](https://www.transtats.bts.gov/).

Through this project, I want to understand the factors associated with flight delays and build a complete machine learning workflow, from data exploration to model evaluation.

## 📊 Dataset

The original dataset contains **539,747 flights and 15 columns**.

After excluding cancelled flights and observations with missing target values, I kept **523,435 flights**.

The target variable is `DEP_DEL15`:

- `0` — departure delay below 15 minutes
    
- `1` — departure delay of 15 minutes or more
    

### Target Distribution

|Class|Flights|Percentage|
|---|---|---|
|Not delayed|428,003|81.77%|
|Delayed|95,432|18.23%|

The dataset is imbalanced, so accuracy alone will not be enough to evaluate the model.

## 🔍 Exploratory Data Analysis

I started by exploring the dataset, checking missing values, identifying potential data leakage, and analyzing the relationship between different features and departure delays.

### Key Findings

|Feature|Observation|
|---|---|
|Origin and destination|Delay rates vary across airports, but extreme values often come from small sample sizes.|
|Departure time|The delay rate increases from 8.07% at 06:00 to 25.01% at 20:00.|
|Distance|The relationship between distance and departure delays does not appear to be linear.|
|Day of week|Sunday and Monday have the highest delay rates, while Wednesday has the lowest.|
|Airline|Delay rates vary across the 14 airlines.|
|Evening flights|13 out of 14 airlines have higher delay rates for flights scheduled after 18:00, including 18:00.|

One of the most interesting patterns I found was the difference between evening and non-evening flights.

|Departure period|Flights|Delay rate|
|---|---|---|
|Before 18:00|414,245|16.80%|
|18:00–23:59|109,190|23.68%|

These findings show associations, not necessarily causal relationships. Other factors, such as airport traffic, weather, and airline operations, may also influence delays.

## Data Leakage Prevention

Since predictions are made before departure, I excluded features that would not be available at the prediction point:

- `DEP_TIME`
    
- `DEP_DELAY`
    
- `CANCELLED`
    
- `CANCELLATION_CODE`
    
- `DIVERTED`
    

I also avoided replacing missing target values with zero, as this would incorrectly label some flights as not delayed.

The current model focuses on operated flights rather than predicting cancellations.

## ⚙️ Data Preprocessing

### Chronological Train-Validation Split

Since flight data is time-dependent, I decided to split the dataset chronologically instead of using a random split.

|Dataset|Period|Flights|Delay rate|
|---|---|---|---|
|Training|January 1–24|404,662|20.22%|
|Validation|January 25–31|118,773|11.46%|

This approach allows the model to learn from historical data and be evaluated on a later period.

The difference in delay rates between the two sets is something I will consider during model evaluation.

### Feature Engineering

I validated the scheduled departure and arrival times and converted their HHMM representation into minutes since midnight.

I then applied **Cyclical Encoding** using sine and cosine transformations with a 1,440-minute cycle.

This preserves the cyclical nature of time, so that times close to midnight remain close in their numerical representation.

The original time columns were replaced with four new features:

- `DEP_TIME_SIN`
    
- `DEP_TIME_COS`
    
- `ARR_TIME_SIN`
    
- `ARR_TIME_COS`
    

### Feature Transformation

After feature engineering, I divided the nine input features into three groups:

|Feature group|Features|Transformation|
|---|---|---|
|Categorical|ORIGIN, DEST, OP_UNIQUE_CARRIER, DAY_OF_WEEK|OneHotEncoder|
|Numerical|DISTANCE|StandardScaler|
|Cyclical|Departure and arrival sine/cosine features|Passthrough|

I applied One-Hot Encoding to the categorical features, generating **679 binary features**.

I also standardized the distance feature using the mean and standard deviation calculated exclusively from the training set.

Both transformations were fitted on training data only and then applied to validation data to prevent data leakage.

For the first model, I decided to keep the cyclical features unchanged.

Once combined, the preprocessing transformations are expected to produce **684 input features**.

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
- [ ] ColumnTransformer
- [ ] Preprocessing pipeline
- [ ] Baseline Logistic Regression
- [ ] Model evaluation
- [ ] Model comparison and hyperparameter tuning
- [ ] Final test set evaluation
## Running the Notebook

Install the required packages:

```
pip install pandas numpy matplotlib scikit-learn jupyter
```

Download the January 2025 flight data from the [BTS website](https://www.transtats.bts.gov/) and save it as `flightinfo.csv` in the project directory.

The dataset is not included in the repository.

Open the notebook:

```
jupyter notebook 01_data_exploration.ipynb
```

## What's Next?

My next step is to combine the preprocessing transformations using ColumnTransformer and build a machine learning pipeline.

I will start with Logistic Regression as a baseline model, evaluate its performance, and then experiment with more complex models.

Since the dataset is imbalanced, I plan to focus on precision, recall, F1-score, and other relevant classification metrics rather than accuracy alone.

The current dataset covers only one month, so I would also like to evaluate the final model on an independent future period to better understand its generalization ability.

This project is still in progress.