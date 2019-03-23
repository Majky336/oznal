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
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

# Import dataset

# Nacitanie datasetu
data = pd.DataFrame(pd.read_csv('listings.csv'))
# Vybratie stlpcov z datasetu
df = pd.DataFrame(data=pd.read_csv('listings.csv'), columns=[
    'price',
    'availability_365',
    'minimum_nights',
    'number_of_reviews',
    ])

df2 = pd.DataFrame(data=pd.read_csv('listings.csv'), columns=[
    'room_type',
    'neighbourhood'
    ])
    
df = df[(np.abs(stats.zscore(df))< 3).all(axis=1)]
df = df.join(df2)

mapper = { 'Private room': 0, 'Entire home/apt': 1, 'Shared room': 2}
df = df.replace({ 'room_type': mapper })
neighbourhoods = df.neighbourhood.unique().tolist()
neighbour_mapper = {neighbourhoods[i]: i for i in range(0, len(neighbourhoods))}
df = df.replace({'neighbourhood': neighbour_mapper})

# Vypis tabulku korelacie
print(df.corr())

pd.plotting.scatter_matrix(df, figsize=(7, 7))
plt.show()


X = df.iloc[:, [1, 2, 3, 4, 5]].values
y = df.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=9)

bp = plt.boxplot(X_train)
plt.show()

# avail_365_upper_outlier = bp['fliers'][0].get_data()[1]
# min_nights_upper_outlier =  bp['fliers'][1].get_data()[1]
# no_reviews_upper_outlier = bp['fliers'][2].get_data()[1]

# print(len(X_train))
# X_train = [x for x in X_train if x[0] not in avail_365_upper_outlier]
# X_train = [x for x in X_train if x[1] not in min_nights_upper_outlier]
# X_train = [x for x in X_train if x[2] not in no_reviews_upper_outlier]
# print(X_train)

linear_model = LinearRegression(n_jobs=-1)
linear_model.fit(X_train,y_train)

predictions = linear_model.predict(X_test)
plt.scatter(y_test,predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")

print(linear_model.score(X_test, y_test))
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
