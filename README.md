# ✈️ Flight Delay Prediction

## About the Project

I'm working on a machine learning model to predict whether a flight will depart at least 15 minutes late.

The idea is to make the prediction **two hours before the scheduled departure**, using only information that would be available at that moment.

I'm using historical U.S. flight data from the [Bureau of Transportation Statistics (BTS)](https://www.transtats.bts.gov/).

I've been exploring the data to understand what might be related to departure delays. So far, I've looked at airports, airlines, departure times, days of the week, and flight distances.

The next step is to prepare the data and start training models.

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

The dataset is imbalanced, so accuracy alone won't be enough to evaluate the model.

For example, predicting every flight as not delayed would give around 81.77% accuracy without correctly identifying any delayed flights.

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

Some examples:

|Feature|Use for prediction?|
|---|---|
|`ORIGIN`|Candidate|
|`DEST`|Candidate|
|`CRS_DEP_TIME`|Candidate|
|`DISTANCE`|Candidate|
|`DAY_OF_WEEK`|Candidate|
|`OP_UNIQUE_CARRIER`|Candidate|
|`DEP_TIME`|No|
|`DEP_DELAY`|No|
|`CANCELLED`|No|
|`CANCELLATION_CODE`|No|
|`DIVERTED`|No|

Actual departure time, departure delay, and final flight status are not known at the prediction point.

Using these values would introduce data leakage.

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

For example, CKB has a delay rate of 77.78%, but that comes from only 9 flights.

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

### 📏 Flight Distance

I checked the distribution of flight distances to see whether longer flights tend to have more departure delays.

The average distance is around **848 miles**, while the median is **715 miles**.

The distribution is right-skewed, with most flights covering shorter distances and relatively few flights above 3,000 miles.

I grouped the flights into four distance categories:

|Distance category|Flights|Delay rate|
|---|---|---|
|Short|175,856|17.01%|
|Medium|186,753|18.81%|
|Long|128,250|19.59%|
|Very Long|32,576|16.16%|

I expected longer flights might have higher delay rates, but the results weren't that straightforward.

Delay rates increase from Short to Long flights, then decrease for Very Long flights.

This suggests that the relationship isn't simply linear. Other factors, such as airlines, routes, or departure times, might also play a role.

For now, I'm keeping `DISTANCE` as a candidate feature.

### 📅 Day of Week

I also wanted to see whether delays are more common on certain days of the week.

|Day|Flights|Delay rate|
|---|---|---|
|Monday|70,812|21.80%|
|Tuesday|61,560|14.92%|
|Wednesday|79,861|12.92%|
|Thursday|89,192|16.78%|
|Friday|88,487|18.87%|
|Saturday|62,298|21.31%|
|Sunday|71,225|21.84%|

Thursday and Friday have the most flights, but they don't have the highest delay rates.

Sunday and Monday have the highest rates, while Wednesday has the lowest.

I found this interesting because it shows that having more flights doesn't automatically mean having a higher percentage of delays.

One possible explanation is that flight schedules differ across the week. For example, some days might have more evening flights, which already seem to have higher delay rates.

I haven't tested all these explanations yet.

### ✈️ Airline Carrier

Next, I looked at `OP_UNIQUE_CARRIER` to compare departure delays across airlines.

The dataset contains **14 airlines**.

Some results:

|Airline|Flights|Delay rate|
|---|---|---|
|WN|102,291|17.47%|
|G4|9,226|23.57%|
|F9|15,134|23.20%|
|OH|19,195|23.13%|
|HA|6,562|11.37%|
|YX|26,714|12.25%|

WN operates the most flights, but it doesn't have the highest delay rate.

G4 and F9 have higher delay rates despite operating fewer flights.

I don't think these differences can be explained by the airline alone. Airlines may operate at different airports, use different schedules, or have different operational conditions.

#### Airline vs. Departure Time

Since I had already found higher delay rates later in the day, I wanted to check whether airlines with more evening flights also tend to have more delays.

I defined evening flights as those scheduled between **18:00 and 23:59**.

Across the dataset:

|Departure period|Flights|Delay rate|
|---|---|---|
|Before 18:00|414,245|16.80%|
|18:00–23:59|109,190|23.68%|

Evening flights have a higher observed delay rate.

I then compared evening and non-evening flights separately for each airline.

|Airline|Before 18:00|Evening|
|---|---|---|
|G4|20.72%|35.37%|
|NK|14.84%|27.08%|
|WN|15.34%|26.63%|
|B6|19.79%|30.52%|
|OO|18.92%|17.81%|

I found that **13 out of 14 airlines** have higher delay rates for evening flights. OO is the only exception.

This made me think that airline and departure hour might be useful together.

However, the proportion of evening flights doesn't fully explain the differences between airlines. Other factors could still be involved.

### 🛬 Destination Airport

I also analyzed `DEST` to see whether the destination airport is associated with departure delays.

There are **329 destination airports**, with no missing values.

The destinations with the most flights are:

|Destination|Flights|
|---|---|
|DEN|24,372|
|DFW|23,520|
|ATL|22,600|
|ORD|21,220|
|CLT|16,108|

I calculated departure delay rates for each destination.

The highest observed rates were:

|Destination|Flights|Delay rate|
|---|---|---|
|CKB|9|55.56%|
|USA|49|44.90%|
|COD|61|37.70%|
|PRC|61|36.07%|
|JMS|58|34.48%|

The lowest were:

|Destination|Flights|Delay rate|
|---|---|---|
|EKO|31|0.00%|
|OTH|18|0.00%|
|PBG|59|5.08%|
|LWS|93|5.38%|
|PIH|61|6.56%|

Just like in the origin airport analysis, the most extreme rates come from airports with relatively few flights.

I also need to remember that `DEP_DEL15` measures departure delays, not arrival delays.

For now, `DEST` is another feature I want to test during modeling.

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
├── .gitignore
├── 01_data_exploration.ipynb
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
    
- Explore flight distance and distance categories
    
- Analyze delay rates by day of the week
    
- Compare airline delay rates
    
- Investigate airline and departure time patterns
    
- Analyze destination airports
    
- Complete the main EDA visualizations
    
- Prepare training and validation data
    
- Build a preprocessing pipeline
    
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

I've finished the main exploratory analysis and identified several features that might help predict departure delays.

Next, I want to prepare the data for machine learning, decide how to split the dataset into training and validation sets, and build a first baseline model.

Since flight data is time-dependent, I also want to make sure the validation approach reflects how predictions would work in practice.

I'll update this README as I train and evaluate the models.