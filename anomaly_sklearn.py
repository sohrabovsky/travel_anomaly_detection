# Importing Packages

import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

# Data:
intflight = pd.read_csv(
    r"/home/sohrab-salehin/Documents/python_scripts/ARIMA-test/intflight_hourly.csv",
    index_col="Booking Date",
    parse_dates=True,
)

# Defining hour and day_num for labeling: 

intflight["hour"] = [i.hour for i in intflight.index]
intflight["day_num"] = [i.dayofweek for i in intflight.index]

# Creating Model:

model = LocalOutlierFactor(contamination=0.03)
intflight["anomaly"] = model.fit_predict(intflight) # -1 for anomalies determines outliers


# if you want to use trained data and then apply it to new data you can use:

# model = LocalOutlierFactor(contamination=0.03)
# model.fit(trainData)
# model.predict(testData)

