# Importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import plotly.graph_objs as go
import plotly .offline as offline
import plotly.figure_factory as ff
import chart_studio.plotly as py


# Importing dataset and examining it
dataset=pd.read_csv('Clients.csv')
dataset.head()
dataset.shape
dataset.isnull().sum()
dataset.describe()

#To see how the numeric data is distributed.
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
dataset['balance'].hist( figsize=(10,5), color='#E14906')
plt.show()

# Converting Categorical features into Numerical features(Encoding)
dataset['job'] = dataset['job'].map({'management':0,'technician':1,'entrepreneur':2,'retired':3,'admin.':4,'services':5,
                                     'blue-collar':6,'self-employed':7,'unemployed':8,'housemaid':9,'student':10})
dataset['marital'] = dataset['marital'].map({'single':0, 'married':1, 'divorced':2})
dataset['education'] = dataset['education'].map({'primary':0, 'secondary':1, 'tertiary':2})
dataset['default'] = dataset['default'].map({'no':0, 'yes':1})
dataset['housing'] = dataset['housing'].map({'no':0, 'yes':1})
dataset['personal'] = dataset['personal'].map({'no':0, 'yes':1})
dataset['term'] = dataset['term'].map({'no':0, 'yes':1})

# Plotting correlation matrix
corrs = dataset.corr()
figure = ff.create_annotated_heatmap(
    z=corrs.values,
    x=list(corrs.columns),
    y=list(corrs.index),
    annotation_text=corrs.round(2).values,
    showscale=True)
offline.plot(figure,filename='corrheatmap_20200420.html')

# Subsetting Data
dataset_test1=dataset[['age','job','balance','housing']]
dataset_test1.head()

dataset_test2 = dataset.loc[:,['age','job','default', 'housing','personal', 'term']]
dataset_test2.head()

# Scaling variables 
feature_scaler = StandardScaler()
dataset_test1=feature_scaler.fit_transform(dataset_test1)

# First Subset of Clients
# Finding the number of clusters (K) - Elbow Plot Method
inertia = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 100)
    kmeans.fit(dataset_test1)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia)
plt.title('The Elbow Plot')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

#Fit Model K mean Cluster
kmeans1 = KMeans(n_clusters = 2)
kmeans1.fit(dataset_test1)

# Summary stats for Clusters
print('Size of Cluster 1: ', (dataset[kmeans1.labels_==0].shape[0]/dataset.shape[0])*100,'\n'
      'Size of Cluster 2: ', (dataset[kmeans1.labels_==1].shape[0]/dataset.shape[0])*100,'\n'
     )

print('Cluster 1 Avg.age: ',dataset[kmeans1.labels_==0].age.mean(),'\n',
      'Cluster 2 Avg.age: ',dataset[kmeans1.labels_==1].age.mean(),'\n')

print('Cluster 1 Avg.balance: ',dataset[kmeans1.labels_==0].balance.mean(),'\n',
      'Cluster 2 Avg.balance: ',dataset[kmeans1.labels_==1].balance.mean(),'\n')

print('Cluster 1 Job Profile Distribution: ',dataset[kmeans1.labels_==0].job.value_counts(),'\n',
      'Cluster 2 Job Profile Distribution: ',dataset[kmeans1.labels_==1].job.value_counts(),'\n')

print('Cluster 1 Housing Loan Distribution: ',dataset[kmeans1.labels_==0].housing.value_counts(),'\n',
      'Cluster 2 Housing Loan Distribution: ',dataset[kmeans1.labels_==1].housing.value_counts(),'\n')
      
# Plotting t-sne visualization to examine and interpret clusters
# HTML File: http://localhost:8888/view/Visualization_CA2/House_Loan_Status_2020042049.html
tsne = TSNE(n_components = 2, perplexity =49,n_iter=2000)
x_tsne = tsne.fit_transform(dataset_test1)

age = list(dataset['age'])
job = list(dataset['job'])
balance = list(dataset['balance'])
housing = list(dataset['housing'])
data = [go.Scatter(x=x_tsne[:,0], y=x_tsne[:,1], mode='markers',
                    marker = dict(color=kmeans1.labels_, colorscale='Rainbow', opacity=0.5),
                                text=[f'age: {a}; job: {b}; balance: {c};housing: {d}' for a,b,c,d in list(zip(age,job,balance,housing))],
                                hoverinfo='text')]

layout = go.Layout(title = 't-SNE Dimensionality Reduction', width = 700, height = 700,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='House_Loan_Final.html')


# Second Subset of Clients
# Plotting Elbow-graph to choose optimal value for K( Number of clusters)
inertia = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 100)
    kmeans.fit(dataset_test2)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia)
plt.title('The Elbow Plot')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# Fitting K-means to the dataset
kmeans2 = KMeans(n_clusters = 2)
kmeans2.fit(dataset_test2)

# Summary stats for Clusters
# Summary stats for Clusters
print('Size of Cluster 1: ', (dataset[kmeans2.labels_==0].shape[0]/dataset.shape[0])*100,'\n'
      'Size of Cluster 2: ', (dataset[kmeans2.labels_==1].shape[0]/dataset.shape[0])*100,'\n'
     )

print('Cluster 1 Avg.age: ',dataset[kmeans2.labels_==0].age.mean(),'\n',
      'Cluster 2 Avg.age: ',dataset[kmeans2.labels_==1].age.mean(),'\n')

print('Cluster 1 Job Profile Distribution: ',dataset[kmeans2.labels_==0].job.value_counts(),'\n',
      'Cluster 2 Job Profile Distribution: ',dataset[kmeans2.labels_==1].job.value_counts(),'\n')

print('Cluster 1 Housing Loan Distribution: ',dataset[kmeans2.labels_==0].housing.value_counts(),'\n',
      'Cluster 2 Housing Loan Distribution: ',dataset[kmeans2.labels_==1].housing.value_counts(),'\n')

print('Cluster 1 Default Status Distribution: ',dataset[kmeans2.labels_==0].default.value_counts(),'\n',
      'Cluster 2 Default Status Distribution: ',dataset[kmeans2.labels_==1].default.value_counts(),'\n')

print('Cluster 1 Personal Loan Distribution: ',dataset[kmeans2.labels_==0].personal.value_counts(),'\n',
      'Cluster 2 Personal Loan Distribution: ',dataset[kmeans2.labels_==1].personal.value_counts(),'\n')
 
print('Cluster 1 Term Deposit Distribution: ',dataset[kmeans2.labels_==0].term.value_counts(),'\n',
      'Cluster 2 Term Deposit Distribution: ',dataset[kmeans2.labels_==1].term.value_counts(),'\n')
      
# Fitting t-sne with the data to derive two dimensions(components) which expalin the maximum variance      
tsne1 = TSNE(n_components = 2, perplexity =30,n_iter=3000)
x_tsne1 = tsne1.fit_transform(dataset_test2) 
 
 
# Plotting t-sne visualization to examine and interpret clusters
# HTML: http://localhost:8888/view/ML_Daily_Learning/VIZ_CA2/Client_subsett_1.html
job = list(dataset['job'])
age = list(dataset['age'])
default = list(dataset['default'])
#balance = list(dataset['balance'])
housing = list(dataset['housing'])
personal = list(dataset['personal'])
term = list(dataset['term'])
data = [go.Scatter(x=x_tsne1[:,0], y=x_tsne1[:,1], mode='markers',
                    marker = dict(color=kmeans2.labels_, colorscale='Rainbow', opacity=0.5),
                                text=[f'age:{a},job: {c}, default:{e}, housing:{g}, personal:{h}, term:{i}' for a,c,e,g,h,i in list(zip(age,job,default,housing,personal,term))],
                                hoverinfo='text')]

layout = go.Layout(title = 't-SNE Dimensionality Reduction', width = 700, height = 700,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='Client_subset_1.html') 








# DATASET - Patches
# Importing dataset and examining it
dataset=pd.read_csv('Patches.csv')

# View top 5 rows
print(dataset.head())

# look at the shape of the dataset
print(dataset.shape)

# check the datatypes of columns and if data is missing for any variable
print(dataset.info())

# Describe your data set to look at the summary metrics of numeric variables
print(dataset.describe())   

# Converting Categorical features into Numerical features(Encoding)
dataset['Tree'] = dataset['Tree'].map({'Other':0, 'Spruce':1})

# After encoding the categorical variables
print(dataset.head())
print(dataset.info())

# Inspect correlation matrix to check for correlation between variables
corrs = dataset.corr()
figure = ff.create_annotated_heatmap(
    z=corrs.values,
    x=list(corrs.columns),
    y=list(corrs.index),
    annotation_text=corrs.round(2).values,
    showscale=True)
offline.plot(figure,filename='corrheatmap_patches.html')

#Dropping Horizontal_Distance_To_Hydrology as it is highly corelated with Vertical_Distance_To_Hydrology
dataset=dataset.drop(['Horizontal_Distance_To_Hydrology'], axis = 1)

dataset.head()

# Subset1
### Trying to arrive at two clusters, one with patches where 'Spruce' trees grow, and second includes patches where 'Other' trees grow

''' Note: We couldn't find any natural clusters in the data, considering all the variables.
- We considered 'Tree' variable to distinguish the clusters, bcz it is the only categorical variable( which can be considered to be the target variable in supervised learning algorithms - Classification)<br>
- After trying multiple combinations of variables, we found 'Slope' and 'Elevation' to be significant in predicting 'Tree', hence it froms 2 clusters'''

# Creating a subset and scaling the variables 
Subset1=dataset[['Slope','Elevation','Vertical_Distance_To_Hydrology','Tree']]
feature_scaler = StandardScaler()
Subset1=feature_scaler.fit_transform(Subset1)


# Elbow plot to find the ideal count of K- number of clusters
inertia = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 100)
    kmeans.fit(Subset1)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia)
plt.title('The Elbow Plot')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# Fitting k-means with 2 clusters( k chosen from Elbow Plot)
kmeans = KMeans(n_clusters = 2)
kmeans.fit(Subset1)

# Summary stats for Clusters
print('Size of Cluster 1: ', (dataset[kmeans.labels_==0].shape[0]/dataset.shape[0])*100,'\n'
      'Size of Cluster 2: ', (dataset[kmeans.labels_==1].shape[0]/dataset.shape[0])*100,'\n'
     )

print('Cluster 1 Avg.Elevation: ',dataset[kmeans.labels_==0].Elevation.mean(),'\n',
      'Cluster 2 Avg.Elevation: ',dataset[kmeans.labels_==1].Elevation.mean(),'\n')

print('Cluster 1 Avg.Slope: ',dataset[kmeans.labels_==0].Slope.mean(),'\n',
      'Cluster 2 Avg.Slope: ',dataset[kmeans.labels_==1].Slope.mean(),'\n')

print('Cluster 1 Avg.Vertical_Distance_To_Hydrology: ',dataset[kmeans.labels_==0].Vertical_Distance_To_Hydrology.mean(),'\n',
      'Cluster 2 Avg.Vertical_Distance_To_Hydrology: ',dataset[kmeans.labels_==1].Vertical_Distance_To_Hydrology.mean(),'\n')


# Fitting t-sne with the data to derive two dimensions(components) which expalin the maximum variance
tsne = TSNE(n_components = 2, perplexity =28,n_iter=2000)
x_tsne = tsne.fit_transform(Subset1)

# Plotting t-sne visualization to examine and interpret clusters 
# HTML: file:///C:/Users/manum/ML_Daily_Learning/VIZ_CA2/tsne_patches1_2clstr.html
Elevation = list(dataset['Elevation'])
V_to_H=list(dataset['Vertical_Distance_To_Hydrology'])
Slope = list(dataset['Slope'])
Tree = list(dataset['Tree'])
data = [go.Scatter(x=x_tsne[:,0], y=x_tsne[:,1], mode='markers',
                    marker = dict(color=kmeans.labels_, colorscale='Rainbow', opacity=0.5),
                                text=[f'Elevation: {a}; Slope:{b};V_to_H{c}; Tree:{d}' for a,b,c,d in list(zip(Elevation,Slope,V_to_H,Tree))],
                                hoverinfo='text')]

layout = go.Layout(title = 't-SNE Dimensionality Reduction 2 clusters', width = 700, height = 700,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='tsne_patches1_2clstr.html')


# Subset2
# This subset results in two clusters, one with elevation grater than 2900 and  other with elevation<2900 
# We can say that we can predict Elevation with variables 'Horizontal_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways'

# Subsetting and scaling the varibles
feature_scaler=StandardScaler()
Subset2=dataset[['Elevation', 'Vertical_Distance_To_Hydrology',   
       'Horizontal_Distance_To_Roadways']]
Subset2=feature_scaler.fit_transform(Subset2)

# Finding the number of clusters (K) - Elbow Plot Method

inertia = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 100)
    kmeans.fit(Subset2)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia)
plt.title('The Elbow Plot')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# Fitting K-means with 2 clusters
kmeans = KMeans(n_clusters = 2)
kmeans.fit(Subset2)

# Summary Statistics for Clusters
print('Size of Cluster 1: ', (dataset[kmeans.labels_==0].shape[0]/dataset.shape[0])*100,'\n'
      'Size of Cluster 2: ', (dataset[kmeans.labels_==1].shape[0]/dataset.shape[0])*100,'\n'
     )

print('Cluster 1 Avg.Elevation: ',dataset[kmeans.labels_==0].Elevation.mean(),'\n',
      'Cluster 2 Avg.Elevation: ',dataset[kmeans.labels_==1].Elevation.mean(),'\n')

print('Cluster 1 Avg.Horizontal_Distance_To_Roadways: ',dataset[kmeans.labels_==0].Horizontal_Distance_To_Roadways.mean(),'\n',
      'Cluster 2 Avg.Horizontal_Distance_To_Roadways: ',dataset[kmeans.labels_==1].Horizontal_Distance_To_Roadways.mean(),'\n')

print('Cluster 1 Avg.Vertical_Distance_To_Hydrology: ',dataset[kmeans.labels_==0].Vertical_Distance_To_Hydrology.mean(),'\n',
      'Cluster 2 Avg.Vertical_Distance_To_Hydrology: ',dataset[kmeans.labels_==1].Vertical_Distance_To_Hydrology.mean(),'\n')


# Fitting t-sne with the data to derive two dimensions(components) which expalin the maximum variance
tsne = TSNE(n_components = 2, perplexity =24,n_iter=2000)
x_tsne = tsne.fit_transform(Subset2)

# Plotting t-sne visualization to examine and interpret clusters
# HTML: file:///C:/Users/manum/ML_Daily_Learning/VIZ_CA2/tsne_patches2_2clstr_V.html
Elevation = list(dataset['Elevation'])
V_D_H = list(dataset['Vertical_Distance_To_Hydrology'])
H_D_R = list(dataset['Horizontal_Distance_To_Roadways'])
data = [go.Scatter(x=x_tsne[:,0], y=x_tsne[:,1], mode='markers',
                    marker = dict(color=kmeans.labels_, colorscale='Rainbow', opacity=0.5),
                                text=[f'Elevation: {a}; V_D_H: {c}; H_D_R: {d}' for a,c,d in list(zip(Elevation,V_D_H,H_D_R))],
                                hoverinfo='text')]

layout = go.Layout(title = 't-SNE Dimensionality Reduction for 2 clusters', width = 700, height = 700,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='tsne_patches2_2clstr_V.html')


# Subset 3
# Trying to form clusters based on value od slope, slope less than sample mean and slope greater than sample mean

## Subsetting and scaling the varibles
Subset3=dataset[['Elevation','Slope','Horizontal_Distance_To_Hydrology']

# Encoding slope by checking if the value is lesser than the sample mean or not 
def converter(column):
    if column >= 16.5:
        return 0 # Low
    else:
        return 1 # High

Subset3['Slope'] = Subset3['Slope'].apply(converter)
Subset33=Subset3
Subset3=feature_scaler.fit_transform(Subset3)

# Finding the number of clusters (K) - Elbow Plot Method

inertia = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 100)
    kmeans.fit(Subset3)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia)
plt.title('The Elbow Plot')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()


# Fitting K-means with k=2
kmeans = KMeans(n_clusters = 2)
kmeans.fit(Subset3)

# Summary stats for Clusters
print('Size of Cluster 1: ', (dataset[kmeans.labels_==0].shape[0]/dataset.shape[0])*100,'\n'
      'Size of Cluster 2: ', (dataset[kmeans.labels_==1].shape[0]/dataset.shape[0])*100,'\n'
     )

print('Cluster 1 Avg.Elevation: ',dataset[kmeans.labels_==0].Elevation.mean(),'\n',
      'Cluster 2 Avg.Elevation: ',dataset[kmeans.labels_==1].Elevation.mean(),'\n')

print('Cluster 1 Avg.Slope: ',dataset[kmeans.labels_==0].Slope.mean(),'\n',
      'Cluster 2 Avg.Slope: ',dataset[kmeans.labels_==1].Slope.mean(),'\n')

print('Cluster 1 Avg.Vertical_Distance_To_Hydrology: ',dataset[kmeans.labels_==0].Vertical_Distance_To_Hydrology.mean(),'\n',
      'Cluster 2 Avg.Vertical_Distance_To_Hydrology: ',dataset[kmeans.labels_==1].Vertical_Distance_To_Hydrology.mean(),'\n')


# Fitting t-sne with the data to derive two dimensions(components) which expalin the maximum variance
# HTML: file:///C:/Users/manum/ML_Daily_Learning/VIZ_CA2/tsne_patches3_2clstr_V.html
tsne = TSNE(n_components = 2, perplexity =50,n_iter=2000)
x_tsne = tsne.fit_transform(Subset3)

# Plotting t-sne visualization to examine and interpret clusters
Slope = list(Subset33['Slope'])
Elevation = list(dataset['Elevation'])
V_D_H = list(dataset['Vertical_Distance_To_Hydrology'])
data = [go.Scatter(x=x_tsne[:,0], y=x_tsne[:,1], mode='markers',
                    marker = dict(color=kmeans.labels_, colorscale='Rainbow', opacity=0.5),
                                text=[f'Slope: {a}; Elevation: {b}; V_D_H: {c}' for a,b,c in list(zip(Slope,Elevation,V_D_H))],
                                hoverinfo='text')]

layout = go.Layout(title = 't-SNE Dimensionality Reduction', width = 700, height = 700,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='tsne_patches3_2clstr_V.html')
