"""
Your Name: Tianyi Fan
Class: CS677 - Summer 2
Date: 8/25/2019
Homework Problem # Final Project
Description of Problem: statistical analyses for parking citations 
in Los Angeles during 2017 and 2018
"""

#Final Project
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix

#import file
ticker='LA_Parking_Citations'
input_dir = r''
ticker_file = os.path.join(input_dir, ticker + '.csv')
df = pd.read_csv(ticker_file)

"""Data Processing"""
#contain only relevant columns
new_df = df[['Ticket number', 'Issue Date', 'Issue time', 'RP State Plate',
             'Make', 'Body Style', 'Location', 'Violation Description',
             'Fine amount']]

#separate date to year and month
new_df['Year'] = [d.split('-')[0] for d in new_df.loc[:,'Issue Date']]
new_df['Month'] = [d.split('-')[1] for d in new_df.loc[:,'Issue Date']]

#rearrange and re-filter dataframe
new_df = new_df[['Ticket number', 'Year', 'Month', 'RP State Plate',
             'Make', 'Body Style', 'Location', 'Violation Description',
             'Fine amount']]

#replace NaN to 0
new_df['Fine amount'] = new_df['Fine amount'].replace(np.nan, 0)

"""Main Project"""
#Training Data for year 2017
df_2017 = new_df[new_df['Year'] == '2017']

#find the most frequent ticketed car body style and make
body_style = df_2017.groupby('Body Style')['Fine amount'].count(
        ).sort_values(ascending = False)

make = df_2017.groupby('Make')['Fine amount'].count(
        ).sort_values(ascending = False)

df_2017 = df_2017[(df_2017['Make'] == 'TOYT')]

#generate and scale x value
x_2017 = df_2017[['Fine amount']].values
scaler = StandardScaler()
scaler.fit(x_2017)
x_2017 = scaler.transform(x_2017)

#Body Style: PA(Panel) = 1, Other = 0
df_2017['Body Style'] = df_2017['Body Style'].apply(
        lambda x:1 if x == 'PA' else 0)
y_2017 = df_2017['Body Style'].values

#Testing Data for year 2018
df_2018 = new_df[new_df['Year'] == '2018']
df_2018 = df_2018[(df_2018['Make'] == 'TOYT')]

x_2018 = df_2018[['Fine amount']].values
scaler.fit(x_2018)
x_2018 = scaler.transform(x_2018)

#Body Style: PA(Panel) = 1, Other = 0
df_2018['Body Style'] = df_2018['Body Style'].apply(
        lambda x:1 if x == 'PA' else 0)
y_2018 = df_2018['Body Style'].values

"""Logistic Regression"""
log_reg = LogisticRegression()
log_reg = log_reg.fit(x_2017, y_2017)

predicted_lr = log_reg.predict(x_2018)
accuracy_lr = np.mean(predicted_lr == y_2018)

print("The accuracy for year 2018 by implementing "
      "logistic regression is {}".format("%.6f" % accuracy_lr))

#Confusion matrix
y_actual = df_2018['Body Style']
y_actual = y_actual.to_numpy()
cm_lr = confusion_matrix(y_actual, predicted_lr)
print(cm_lr)

#True postive rate and True negative rate
TPR_lr = round(cm_lr[1,1]/(cm_lr[1,1]+cm_lr[1,0]),8) #TPR = TP/(TP+FN)
TNR_lr = round(cm_lr[0,0]/(cm_lr[0,0]+cm_lr[0,1]),8) #TNR = TN/(TN+FP)
print('True positive rate is {}'.format(str(TPR_lr)))
print('True negative rate is {}'.format(str(TNR_lr)))

"""Decision Tree"""
clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(x_2017, y_2017)

predicted_dt = clf.predict(x_2018)
accuracy_dt = np.mean(predicted_dt == y_2018)

print("The accuracy for year 2018 by implementing "
      "decision tree is {}".format("%.6f" % accuracy_dt))

#Confusion matrix
y_actual = df_2018['Body Style']
y_actual = y_actual.to_numpy()
cm_dt = confusion_matrix(y_actual, predicted_dt)
print(cm_dt)

#True postive rate and True negative rate
TPR_dt = round(cm_dt[1,1]/(cm_dt[1,1]+cm_dt[1,0]),8) #TPR = TP/(TP+FN)
TNR_dt = round(cm_dt[0,0]/(cm_dt[0,0]+cm_dt[0,1]),8) #TNR = TN/(TN+FP)
print('True positive rate is {}'.format(str(TPR_dt)))
print('True negative rate is {}'.format(str(TNR_dt)))

"""Naive Bayesian"""
NB_classifier = GaussianNB()
NB_classifier = NB_classifier.fit(x_2017, y_2017)

predicted_nb = NB_classifier.predict(x_2018)
accuracy_nb = np.mean(predicted_nb == y_2018)

print("The accuracy for year 2018 by implementing "
      "naive bayesian is {}".format("%.6f" % accuracy_nb))

#Confusion matrix
y_actual = df_2018['Body Style']
y_actual = y_actual.to_numpy()
cm_nb = confusion_matrix(y_actual, predicted_nb)
print(cm_nb)

#True postive rate and True negative rate
TPR_nb = round(cm_nb[1,1]/(cm_nb[1,1]+cm_nb[1,0]),8) #TPR = TP/(TP+FN)
TNR_nb = round(cm_nb[0,0]/(cm_nb[0,0]+cm_nb[0,1]),8) #TNR = TN/(TN+FP)
print('True positive rate is {}'.format(str(TPR_nb)))
print('True negative rate is {}'.format(str(TNR_nb)))

"""K-nearest neighbors"""
"""Due to the large amount of data, the program may run a while 
(approx. 30 minutes). Please see the attachment in word file for the best k 
value, accuracy, confusion matrix, TPR and TNP"""
k = [3,5,7,9,11]
accuracy = []
for i in range(len(k)):
    knn_classifier = KNeighborsClassifier(n_neighbors = k[i])
    knn_classifier.fit(x_2017,y_2017)
    predicted_knn = knn_classifier.predict(x_2018)
    accuracy.append(np.mean(predicted_knn == y_2018))
    
plt.figure(figsize=(10,4))
ax = plt.gca()
plt.plot(range(3,13,2),accuracy,color ='red',linestyle='dashed',marker='o',
         markerfacecolor='black',markersize =10)
plt.xlabel('values of k')
plt.ylabel('accuracy')

knn_classifier = KNeighborsClassifier(n_neighbors = 3)
knn_classifier.fit(x_2017,y_2017)

predicted_knn = knn_classifier.predict(x_2018)
accuracy_knn = np.mean(predicted_knn == y_2018)

print("The accuracy for year 2018 by implementing "
      "K-nearest neighbors is {}".format("%.6f" % accuracy_knn))

#Confusion matrix
y_actual = df_2018['Body Style']
y_actual = y_actual.to_numpy()
cm_knn = confusion_matrix(y_actual, predicted_knn)
print(cm_knn)

#True postive rate and True negative rate
TPR_knn = round(cm_knn[1,1]/(cm_knn[1,1]+cm_knn[1,0]),8) #TPR = TP/(TP+FN)
TNR_knn = round(cm_knn[0,0]/(cm_knn[0,0]+cm_knn[0,1]),8) #TNR = TN/(TN+FP)
print('True positive rate is {}'.format(str(TPR_knn)))
print('True negative rate is {}'.format(str(TNR_knn)))

"""K-means clustering"""
df_kmeans = new_df[new_df['Year'] >= '2017']
df_kmeans = df_kmeans[(df_kmeans['Make'] == 'TOYT')]

df_kmeans = df_kmeans.reset_index(drop = True)

x_kmeans = df_kmeans[['Fine amount']].values
scaler = StandardScaler()
scaler.fit(x_kmeans)
x_kmeans = scaler.transform(x_kmeans)

inertia_list = []
for k in range(1,9):
    kmeans_classifier = KMeans(n_clusters=k)
    y_kmeans = kmeans_classifier.fit_predict(x_kmeans)
    inertia = kmeans_classifier.inertia_
    inertia_list.append(inertia)

fig,ax = plt.subplots(1,figsize =(7,5))
plt.plot(range(1,9), inertia_list, marker='o',
        color='green')

plt.legend()
plt.xlabel('number of clusters: k')
plt.ylabel('inertia')
plt.tight_layout()

plt.show()

kmeans_classifier = KMeans(n_clusters=2)
y_kmeans = kmeans_classifier.fit_predict(x_kmeans)
centroids = kmeans_classifier.cluster_centers_

y_kmeans_Q2 = pd.DataFrame(y_kmeans)
df_kmeans['Cluster'] = y_kmeans_Q2

cluster_0_pa = 0
cluster_0_other = 0
cluster_1_pa = 0
cluster_1_other = 0

for i in range(len(df_kmeans)):
    if df_kmeans.loc[i,'Body Style']=='PA' and df_kmeans.loc[i,'Cluster'] == 0:
        cluster_0_pa += 1
    elif df_kmeans.loc[i,'Body Style']!='PA' and df_kmeans.loc[i,'Cluster']==0:
        cluster_0_other += 1
    elif df_kmeans.loc[i,'Body Style']=='PA' and df_kmeans.loc[i,'Cluster']==1:
        cluster_1_pa += 1
    elif df_kmeans.loc[i,'Body Style']!='PA' and df_kmeans.loc[i,'Cluster']==1:
        cluster_1_other += 1
        
cluster_0 = df_kmeans[(df_kmeans.Cluster == 0)].count()['Body Style']
cluster_1 = df_kmeans[(df_kmeans.Cluster == 1)].count()['Body Style']

print("In the first cluster, the percentage of PA" 
      " body style is {}".format(round(cluster_0_pa/cluster_0,2)),
      "and the percentage of other"
      " body style is {}".format(round(cluster_0_other/cluster_0,2)))
print("In the second cluster, the percentage of PA" 
      " body style is {}".format(round(cluster_1_pa/cluster_1,2)),
      "and the percentage of other"
      " body style is {}".format(round(cluster_1_other/cluster_1,2)))

#determine whether there is a pure cluster
if cluster_0_pa/cluster_0 > 0.9:
    print("My first clustering for PA body style is a pure cluster")
elif cluster_0_other/cluster_0 > 0.9:
    print("My first clustering for other body style is a pure cluster")
elif cluster_1_pa/cluster_1 > 0.9:
    print("My second clustering for PA body styleis a pure cluster")
elif cluster_1_other/cluster_1 > 0.9:
    print("My second clustering for other body style is a pure cluster")
else:
    print("My clustering does not find any pure clusters.")
    