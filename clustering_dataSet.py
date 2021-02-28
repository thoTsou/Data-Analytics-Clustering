import pandas as pd
from numpy import *

data = pd.read_csv( 'dataSet.csv' , encoding='utf-8') #original data

df_rep = data.copy() #make copy of original data

replace_map1 = {'Ranking': {'not_recom': 1, 'recommend': 2, 'very_recom': 3, 'priority': 4,'spec_prior': 5} }

df_rep.replace(replace_map1, inplace=True)

replace_map2 = {'Parents_Occupation': {'usual': 1, 'pretentious': 2, 'great_pret': 3 } }

df_rep.replace(replace_map2, inplace=True)


replace_map5 = {'Housing_Conditions': {'critical': 1, 'less_conv': 3, 'convenient': 3} }

df_rep.replace(replace_map5, inplace=True)


replace_map6 = {'Finance_Standing': {'inconv': 1, 'convenient': 2} }
df_rep.replace(replace_map6, inplace=True)

replace_map7 = {'Social_picture': {'problematic': 1, 'slightly_prob': 2, 'nonprob': 3} }

df_rep.replace(replace_map7, inplace=True)

replace_map11 = {' Family_Structure': {'foster': 1, 'incomplete': 2, 'completed': 3, 'complete': 4} }

df_rep.replace(replace_map11, inplace=True)

replace_map8 = {'Health_picture': {'not_recom': 1, 'priority': 2, 'recommended': 3} }

df_rep.replace(replace_map8, inplace=True)

replace_map9 = {'Number_of_Children': {'more': 5} }

df_rep.replace(replace_map9, inplace=True)


replace_map10 = {'Childs_Nursery': {'very_crit': 1, 'critical': 2, 'improper': 3, 'less_proper': 4,'proper': 5} }

df_rep.replace(replace_map10, inplace=True)


df_rep = df_rep.sample(n = 400)


from sklearn.preprocessing import StandardScaler

features = ['Parents_Occupation', 'Childs_Nursery', ' Family_Structure', 'Number_of_Children' , 'Housing_Conditions' , 'Finance_Standing' , 'Social_picture' , 'Health_picture' ]


x = df_rep.loc[:, features].values


x = StandardScaler().fit_transform(x)

x # array with values of df_rep after standarization and without Ranking column

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['component 1', 'component 2'])

principalDf #dataframe with only two components , after pca on x array

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5).fit(principalDf)
centroids = kmeans.cluster_centers_
print(centroids)



plt.scatter(principalDf['component 1'], principalDf['component 2'] ,c= kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1],c='red' )


from sklearn.metrics import silhouette_score

range_n_clusters = [2, 3, 4, 5]


for n_clusters in range_n_clusters:
  
    clusterer = KMeans(n_clusters=n_clusters).fit(principalDf)
    cluster_labels = clusterer.fit_predict(principalDf)
    
    silhouette_avg = silhouette_score(principalDf, cluster_labels)
    print("For n_clusters =", n_clusters,"The average silhouette_score is :", silhouette_avg)