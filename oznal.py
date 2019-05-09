# -*- coding: utf-8 -*-

# Import libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing

# > Util functions


def boolToBinary(row):
    if row == "t":
        return 1
    if row == "f":
        return 0
    return row


def transformCleaningFee(row):
    if type(row) == float:
        return row
    row = row.replace("$", "").replace(",", "").replace(" ", "")
    return float(row)


def reject_outliers(sr, iq_range=0.5):
    pcnt = (1 - iq_range) / 2
    qlow, median, qhigh = sr.dropna().quantile([pcnt, 0.50, 1 - pcnt])
    iqr = qhigh - qlow
    return sr[(sr - median).abs() <= iqr]


def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3 - q1  # Interquartile range
    fence_low = q1 - 1.5 * iqr
    fence_high = q3 + 1.5 * iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true[y_true == 0] = 1

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# < Util functions

# Vybratie stlpcov z datasetu
df = pd.DataFrame(
    data=pd.read_csv("listings.csv"),
    columns=[
        "price",
        "availability_365",
        "minimum_nights",
        "number_of_reviews",
        "room_type",
        "neighbourhood",
    ],
)

df2 = pd.DataFrame(
    data=pd.read_csv("listings_details.csv"),
    columns=[
        "accommodates",
        "cleaning_fee",
        "guests_included",
        "host_identity_verified",
        "property_type",
        "bathrooms",
        "bedrooms",
        "beds",
        "maximum_nights",
        "review_scores_rating",
        "review_scores_accuracy",
        "review_scores_cleanliness",
        "review_scores_checkin",
        "review_scores_communication",
        "review_scores_location",
        "review_scores_value",
        "reviews_per_month",
    ],
)

data = df.join(df2)

features = [
    "availability_365",
    "minimum_nights",
    "number_of_reviews",
    "room_type",
    "neighbourhood",
    "accommodates",
    "cleaning_fee",
    "maximum_nights",
    "guests_included",
    "host_identity_verified",
    "property_type",
    "bathrooms",
    "bedrooms",
    "beds",
    "review_scores_rating",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value",
    "reviews_per_month",
]
target = ["price"]

data = remove_outlier(data, "availability_365")
data = remove_outlier(data, "minimum_nights")
data = remove_outlier(data, "number_of_reviews")
data = remove_outlier(data, "price")

# Drop NAN values
# print(data.isna().sum())
# data = data.dropna()
data["cleaning_fee"] = data["cleaning_fee"].apply(transformCleaningFee)
data["host_identity_verified"] = data["host_identity_verified"].apply(boolToBinary)
imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")

# columns_with_NaN_values = data[data.columns[data.isna().any()].tolist()]
# filled_columns = imputer.fit_transform(columns_with_NaN_values)

# # Vypis tabulku korelacie
# print(data.corr())

X = data.iloc[:][features]
y = data.iloc[:][target]

# X = X.fillna(0)

# get_dummies method
X = pd.get_dummies(X, prefix_sep="", drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=9
)

X_train = imputer.fit_transform(X_train)
X_test = imputer.fit_transform(X_test)

# # normalization of data
# train_norm = X_train[X_train.columns]
# test_norm = X_test[X_test.columns]

# std_scale = preprocessing.StandardScaler().fit(train_norm)
# x_train_norm = std_scale.transform(train_norm)

# training_norm_col = pd.DataFrame(
#     x_train_norm, index=train_norm.index, columns=train_norm.columns
# )
# X_train.update(training_norm_col)

# x_test_norm = std_scale.transform(test_norm)
# testing_norm_col = pd.DataFrame(
#     x_test_norm, index=test_norm.index, columns=test_norm.columns
# )
# X_test.update(testing_norm_col)

# Backward Elimination
# cols = list(X.columns)
# pmax = 1
# while len(cols) > 0:
#     p = []
#     X_1 = X[cols]
#     X_1 = sm.add_constant(X_1)
#     model = sm.OLS(y, X_1).fit()
#     p = pd.Series(model.pvalues.values[1:], index=cols)
#     pmax = max(p)
#     feature_with_p_max = p.idxmax()
#     if pmax > 0.05:
#         cols.remove(feature_with_p_max)
#     else:
#         break
# selected_features_BE = cols
# print(selected_features_BE)
# print(len(selected_features_BE))
# print(X_train.head())
# X_train = X_train[np.asarray(selected_features_BE)]
# X_test = X_test[np.asarray(selected_features_BE)]
# print(X_train.head())

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
plt.scatter(y_test, predictions, color="blue")
plt.xlabel("True Values")
plt.ylabel("Predictions")

print(model.score(X_test, y_test))
print("Mean squared error:", math.sqrt(mean_squared_error(y_test, predictions)))
print(
    "Mean average percentage error:",
    mean_absolute_percentage_error(y_test, predictions),
)
print("Variance score: %.2f" % r2_score(y_test, predictions))
plt.show()
