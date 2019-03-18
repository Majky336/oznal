# -*- coding: utf-8 -*-
### Data preprocessing

### Import libraries
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import pandas as pd

### Import dataset

data = pd.DataFrame(pd.read_csv('listings.csv'))
df = pd.DataFrame(data = pd.read_csv('listings.csv'), columns=[
        'price', 
        'number_of_reviews', 
        'latitude', 
        'longitude', 
        'neighbourhood'])

plt.subplot(1,2,1)
bp = plt.boxplot(data["price"], showfliers=False)
plt.ylabel('Cena')
plt.title('Boxplot pre atribút cena')

max_whisker = [item.get_ydata()[1] for item in bp['whiskers']][1]

filtered_df = df[df.price < max_whisker]

print(max_whisker)

plt.subplot(1,2,2)
plt.hist(filtered_df['price'], bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram pre atribút cena")
plt.ylabel('Počet záznamov')
plt.xlabel('Cena')
plt.show()


neighbourhoods = filtered_df.neighbourhood.unique().tolist()
mapper = { neighbourhoods[i]:i  for i in range(0, len(neighbourhoods))}
filtered_df = filtered_df.replace({ 'neighbourhood': mapper })

norm = clrs.Normalize(vmin=0, vmax=21, clip=True)
mapper = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis)
filtered_df['color'] = filtered_df['neighbourhood'].apply(lambda x: clrs.to_hex(mapper.to_rgba(x)))

patches = []
colors = filtered_df.color.unique().tolist()
index = 0

for i in neighbourhoods:
    patches.append(mpatches.Patch(color=colors[index], label=i))
    index += 1


plt.figure()
sc = plt.scatter(filtered_df['latitude'], filtered_df['longitude'], c=filtered_df['color'])
plt.xlabel('Zemepisná šírka')
plt.ylabel('Zemepisná dĺžka')
plt.legend(handles=patches, loc='lower-left')
plt.show()