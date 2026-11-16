# ✈️ Flight Delay Prediction

## About the Project

I'm working on a machine learning model to predict whether a flight will depart at least 15 minutes late.

The idea is to make the prediction **two hours before the scheduled departure**, using only information that would be available at that moment.

I'm using historical flight data from the [Bureau of Transportation Statistics (BTS)](https://www.transtats.bts.gov/).

This is an ongoing project. I'm currently exploring the data, looking for patterns related to departure delays, and deciding which features could be useful for the model.

## 📊 Dataset

The original dataset contains **539,747 flights and 15 columns**.

The target is `DEP_DEL15`:

- `0` — departure delay below 15 minutes
    
- `1` — departure delay of 15 minutes or more
    

I excluded cancelled flights and rows with missing target values, leaving **523,435 flights** for the current analysis.

### Target Distribution

|Class|Flights|Percentage|
|---|---|---|
|Not delayed|428,003|81.77%|
|Delayed|95,432|18.23%|

The dataset is imbalanced, so accuracy alone won't be enough to evaluate the model. For example, predicting every flight as not delayed would give around 81.77% accuracy without correctly identifying any delayed flights.

## 🔍 Exploratory Data Analysis

### Missing Values

I started by checking missing values and understanding why they occur.

|Column|Missing values|
|---|---|
|`DEP_TIME`|15,886|
|`DEP_DELAY`|15,923|
|`DEP_DEL15`|15,923|
|`CANCELLATION_CODE`|523,435|

Most missing departure delay values are related to cancelled flights. I also found that `CANCELLATION_CODE` is usually missing when a flight wasn't cancelled.

I decided not to fill missing target values with zero, since that would incorrectly label some flights as not delayed.

### Data Leakage

Since I want to predict delays two hours before departure, I need to be careful about which features I use.

For example:

|Feature|Use for prediction?|
|---|---|
|`ORIGIN`|Yes, candidate|
|`CRS_DEP_TIME`|Yes, candidate|
|`DISTANCE`|Yes, candidate|
|`DEP_TIME`|No|
|`DEP_DELAY`|No|

Actual departure time and departure delay are only known after the prediction point, so using them would introduce data leakage.

### 🛫 Origin Airport

I compared departure delay rates across origin airports.

Some airports had very high delay rates:

|Airport|Flights|Delay rate|
|---|---|---|
|CKB|9|77.78%|
|HGR|19|57.89%|
|USA|49|44.90%|
|TOL|30|43.33%|
|SCE|16|37.50%|

Others had much lower rates:

|Airport|Flights|Delay rate|
|---|---|---|
|PQI|31|0.00%|
|OTH|17|0.00%|
|EKO|31|3.23%|
|PSG|55|3.64%|
|OTZ|25|4.00%|

One thing I noticed is that all these airports have relatively few flights in the dataset.

For example, CKB has a delay rate of 77.78%, but that comes from only 9 flights. A single additional on-time flight would change the percentage considerably.

Because of this, I don't think it's reasonable to rank airports based only on these percentages without considering the number of observations.

### 🕒 Scheduled Departure Time

I extracted the departure hour from `CRS_DEP_TIME` to check whether delays are more common at certain times of the day.

For example, `1430` becomes hour `14`.

Here are some results:

|Hour|Flights|Delay rate|
|---|---|---|
|06:00|35,288|8.07%|
|09:00|31,417|14.46%|
|12:00|32,072|19.62%|
|15:00|30,865|22.15%|
|18:00|32,518|23.16%|
|20:00|22,380|25.01%|

This was one of the more interesting patterns I found.

Flights scheduled early in the morning have lower observed delay rates, while flights later in the day tend to have higher rates.

At 06:00, the delay rate is around 8%, compared with 25% at 20:00.

This doesn't necessarily mean that departing later causes delays. There could be other factors involved, such as previous flight delays, airport traffic, or weather.

Still, I think departure hour is worth testing as a feature when I start training models.

## 🛠️ Tools

- **Python** — programming
    
- **Pandas** — data cleaning and analysis
    
- **NumPy** — numerical operations
    
- **Matplotlib** — visualizations
    
- **Jupyter Notebook** — experiments and documentation
    
- **Scikit-learn** — planned model development
    

## 📁 Repository Structure

```
flight-delay-prediction/
├── 01_data_exploration.ipynb
├── flightinfo.csv
└── README.md
```

The structure will change as I add preprocessing and modeling notebooks.

## 🚀 Progress

- Explore the dataset
    
- Investigate missing values and cancellations
    
- Analyze target distribution
    
- Identify potential data leakage
    
- Analyze delay rates by origin airport
    
- Analyze delays by scheduled departure hour
    
- Finish EDA visualizations
    
- Explore additional features
    
- Prepare training and validation data
    
- Train a baseline model
    
- Compare different models
    
- Evaluate results and improve the model
    

## ▶️ Running the Notebook

Install the packages used for the exploratory analysis:

```
pip install pandas numpy matplotlib jupyter
```

Open the notebook:

```
jupyter notebook 01_data_exploration.ipynb
```

The notebook reads the BTS data from `flightinfo.csv`. The dataset can be obtained from the [BTS website](https://www.transtats.bts.gov/).

## What's Next?

I'm still working on the exploratory analysis.

Once I finish investigating the features, I'll move on to preprocessing, model training, and evaluation.

I'll update this README with model results and anything else I find interesting along the way.