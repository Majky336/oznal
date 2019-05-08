# -*- coding: utf-8 -*-
# Data preprocessing

# Import libraries
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import pandas as pd
import numpy as np

from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.feature_selection import RFE
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import precision_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.utils import check_array


# > Util functions


def reject_outliers(sr, iq_range=0.5):
    pcnt = (1 - iq_range) / 2
    qlow, median, qhigh = sr.dropna().quantile([pcnt, 0.50, 1-pcnt])
    iqr = qhigh - qlow
    return sr[(sr - median).abs() <= iqr]


def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1  # Interquartile range
    fence_low = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) &
                       (df_in[col_name] < fence_high)]
    return df_out


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true[y_true == 0] = 1

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# < Util functions

# Vybratie stlpcov z datasetu
df = pd.DataFrame(data=pd.read_csv('listings.csv'), columns=[
    'price',
    'availability_365',
    'minimum_nights',
    'number_of_reviews',
    'room_type',
    'neighbourhood'
])

df2 = pd.DataFrame(data=pd.read_csv('listings_details.csv'), columns=[
    'property_type',
    'bathrooms',
    'bedrooms',
    'beds',
])

# print(df.columns.tolist())
# print(df2.columns.tolist())
# print("--------------------------------------------")
data = df.join(df2)

features = ['availability_365', 'minimum_nights', 'number_of_reviews',
            'room_type', 'neighbourhood', 'property_type', 'bathrooms', 'bedrooms', 'beds']
target = ['price']


data = remove_outlier(data, 'availability_365')
data = remove_outlier(data, 'minimum_nights')
data = remove_outlier(data, 'number_of_reviews')
data = remove_outlier(data, 'price')

# # Vypis tabulku korelacie
print(data.corr())

X = data.iloc[:][features]
y = data.iloc[:][target]

X = X.fillna(0)

# X_dict = X.to_dict(orient='record')
# dv_X = DictVectorizer(sparse=False)
# X_encoded = dv_X.fit_transform(X_dict)

# get_dummies method
X = pd.get_dummies(X, prefix_sep='', drop_first=True)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=9)

# print(X_train)
# print(X_train.dropna(axis=0, how='any'))

# # bp = plt.boxplot(X_train)
# # plt.show()

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

predictions = linear_model.predict(X_test)
plt.scatter(y_test, predictions, color='blue')
plt.xlabel("True Values")
plt.ylabel("Predictions")

print(linear_model.score(X_test, y_test))
print("Mean squared error:", mean_squared_error(y_test, predictions))
print("Mean average percentage error:",
      mean_absolute_percentage_error(y_test, predictions))
print('Variance score: %.2f' % r2_score(y_test, predictions))
plt.show()

# print('K fold')

# scores = []
# best_svr = SVR(kernel='rbf', gamma='auto')
# cv = KFold(n_splits=5, shuffle=True, random_state=9)
# for train_index, test_index in cv.split(X):
#     print("Train Index: ", train_index)
#     print("Test Index: ", test_index)

#     X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
#     best_svr.fit(X_train, y_train)
#     scores.append(best_svr.score(X_test, y_test))


# print(cross_val_score(best_svr, X, y, cv=5))
# print(cross_val_predict(best_svr, X, y, cv=5))

# # Definicia mriezky pre jednotlive grafy
# plt.subplot(1, 2, 1)
# bp = plt.boxplot(data["price"], showfliers=False)
# plt.ylabel('Cena')
# plt.title('Boxplot pre atribút cena')

# # Zistime hornu hranicu ceny z boxplotu pre cenu
# max_whisker = [item.get_ydata()[1] for item in bp['whiskers']][1]

# # Odfiltruejeme data, ktore presahuju hornu hranicu ceny
# filtered_df = df[df.price < max_whisker]

# # Mriezka a umiestnenie noveho grafu
# plt.subplot(1, 2, 2)
# # arguments are passed to np.histogram
# plt.hist(filtered_df['price'], bins='auto')
# plt.title("Histogram pre atribút cena")
# plt.ylabel('Počet záznamov')
# plt.xlabel('Cena')
# plt.show()

# # Vyberieme si unikatne mena susedstiev
# neighbourhoods = filtered_df.neighbourhood.unique().tolist()
# # Namapujeme si mena susedstiev na cisla
# mapper = {neighbourhoods[i]: i for i in range(0, len(neighbourhoods))}
# # Vo filtrovanom datasete vymenime slovne nazvy susedstiev na cisla, pomocou mapovaca
# filtered_df = filtered_df.replace({'neighbourhood': mapper})

# # Vytvorime farby pre jednotlive susedstva -- mostly Googled stuff
# norm = clrs.Normalize(vmin=0, vmax=21, clip=True)
# mapper = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis)
# filtered_df['color'] = filtered_df['neighbourhood'].apply(
#     lambda x: clrs.to_hex(mapper.to_rgba(x)))

# patches = []
# colors = filtered_df.color.unique().tolist()
# index = 0
# # Vytvorime patches pre farby, tak aby sa dali aplikovat na scatter plot
# for i in neighbourhoods:
#     patches.append(mpatches.Patch(color=colors[index], label=i))
#     index += 1

# # Zobrazime vysledny scatter plot aj s farbami
# plt.figure()
# sc = plt.scatter(filtered_df['latitude'],
#                  filtered_df['longitude'], c=filtered_df['color'])
# plt.xlabel('Zemepisná šírka')
# plt.ylabel('Zemepisná dĺžka')
# plt.legend(handles=patches, loc='lower-left')
# plt.show()
