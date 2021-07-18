# -*- coding: utf-8 -*-
"""
Spyder Editor

US Accidents
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
import gc
#from imblearn.under_sampling import RandomUnderSampler

dataset=pd.read_csv("/home/gnaps/Scrivania/DataMining/US_Accidents_Dec20_Updated.csv")

#Data PreProcessing
#Check one class data 
columns=dataset.columns
for i in columns:
  print(i,dataset[i].unique().size)

      
#Getting missing values percentage
percentage_of_missing_values=dataset.isna().sum()/len(dataset)*100
percentage_of_missing_values[percentage_of_missing_values!=0].plot(kind="bar")

#Removing useless data
colsToDelete=["ID","Number", "Turning_Loop", "End_Lat", "End_Lng", "Airport_Code", "Zipcode", "Country", "Side", "Timezone", "Nautical_Twilight", "Astronomical_Twilight", "Civil_Twilight", "Wind_Chill(F)", "Precipitation(in)", "Description", "Weather_Timestamp"]
dataset.drop(colsToDelete,axis=1,inplace=True) 

#removing element with na weatherCond
dataset.dropna(subset=["Weather_Condition"], inplace=True)
#Could be a function
#Add wheater cond to dataset (if contains word then true)
dataset["snowCond"]=np.where(dataset["Weather_Condition"].str.contains("Snow|Wintry|Freezing|Hail|Sleet|Ice"), True, False)
print(dataset.snowCond.value_counts())
dataset["rainyCond"]=np.where(dataset["Weather_Condition"].str.contains("Rain|Drizzle|Thunderstorm|Thunder|T-Storm|Squalls"), True, False)
print(dataset.rainyCond.value_counts())
dataset["fogCond"]=np.where(dataset["Weather_Condition"].str.contains("Fog|Haze|Smoke|Mist|Dust|Sand|Volcanic"), True, False)
print(dataset.fogCond.value_counts())
dataset["windyCond"]=np.where(dataset["Weather_Condition"].str.contains("Windy|Tornado"), True, False)
print(dataset.windyCond.value_counts())

#set wind_Speed to windyCond mean value if windyCond true else to non windyCond mean value
windyAcc=dataset[dataset["windyCond"] == True]
notWindyAcc=dataset[dataset["windyCond"] == False]

windyMean = windyAcc["Wind_Speed(mph)"].mean()
notWindyMean = notWindyAcc["Wind_Speed(mph)"].mean()
dataset["Wind_Speed(mph)"]=np.where((dataset["Wind_Speed(mph)"].isna()) & (dataset["windyCond"]==False), notWindyMean, windyMean)

#describe=dataset.describe()

#Getting new missing values percentage
percentage_of_missing_values=dataset.isna().sum()/len(dataset)*100
percentage_of_missing_values[percentage_of_missing_values!=0].plot(kind="bar")

#Cleaning/Normalization
#Removing na elements
print(len(dataset))
print(dataset.isna().sum())
#removing element with na city
dataset.dropna(subset=["City"], inplace=True)
#removing element with na Wind_Direction
dataset.dropna(subset=["Wind_Direction"], inplace=True)
#removing element with na Sunrise_Sunset
dataset.dropna(subset=["Sunrise_Sunset"], inplace=True)

#removing/replacing with mean value element with na Temperature
dataset["Temperature(F)"]=dataset["Temperature(F)"].fillna(dataset["Temperature(F)"].median())
#removing/replacing with mean value element with na Humidity
dataset["Humidity(%)"]=dataset["Humidity(%)"].fillna(dataset["Humidity(%)"].median())
#removing/replacing with mean value element with na Visibility
dataset["Visibility(mi)"]=dataset["Visibility(mi)"].fillna(dataset["Visibility(mi)"].median())
#removing/replacing with mean value element with na Pressure
dataset["Pressure(in)"]=dataset["Pressure(in)"].fillna(dataset["Pressure(in)"].median())

#Getting new missing values percentage (should be 0)
print(len(dataset))
print(dataset.isna().sum())

#Normalization
#Normalizing WindDirection
dataset.loc[dataset['Wind_Direction']=='Calm','Wind_Direction'] = 'CALM'
dataset.loc[(dataset['Wind_Direction']=='West')|(dataset['Wind_Direction']=='WSW')|(dataset['Wind_Direction']=='WNW'),'Wind_Direction'] = 'W'
dataset.loc[(dataset['Wind_Direction']=='South')|(dataset['Wind_Direction']=='SSW')|(dataset['Wind_Direction']=='SSE'),'Wind_Direction'] = 'S'
dataset.loc[(dataset['Wind_Direction']=='North')|(dataset['Wind_Direction']=='NNW')|(dataset['Wind_Direction']=='NNE'),'Wind_Direction'] = 'N'
dataset.loc[(dataset['Wind_Direction']=='East')|(dataset['Wind_Direction']=='ESE')|(dataset['Wind_Direction']=='ENE'),'Wind_Direction'] = 'E'
dataset.loc[dataset['Wind_Direction']=='Variable','Wind_Direction'] = 'VAR'
print("Wind Direction after simplification: ", dataset['Wind_Direction'].unique())

print(dataset.dtypes)

#From date to day, month,year
dataset["year"]=(pd.DatetimeIndex(dataset["Start_Time"]).year)
dataset["hour"]=(pd.DatetimeIndex(dataset["Start_Time"]).hour)
dataset["month"]=(pd.DatetimeIndex(dataset["Start_Time"]).month)
dataset["dayofweek"]=(pd.DatetimeIndex(dataset["Start_Time"]).dayofweek)

#Duration in minutes
dataset["duration"]=(((pd.DatetimeIndex(dataset["End_Time"])- (pd.DatetimeIndex(dataset["Start_Time"]))).days*24*60+(((pd.DatetimeIndex(dataset["End_Time"]) - (pd.DatetimeIndex(dataset["Start_Time"]))).seconds//60)%60 )))

#Convert object type to int
dataset["Street"]=dataset["Street"].astype("category")
dataset["Street_Codes"]=dataset["Street"].cat.codes

dataset["City"]=dataset["City"].astype("category")
dataset["City_Codes"]=dataset["City"].cat.codes

dataset["County"]=dataset["County"].astype("category")
dataset["County_Codes"]=dataset["County"].cat.codes

dataset["State"]=dataset["State"].astype("category")
dataset["State_Codes"]=dataset["State"].cat.codes

dataset["Wind_Direction"]=dataset["Wind_Direction"].astype("category")
dataset["Wind_Direction_Codes"]=dataset["Wind_Direction"].cat.codes

dataset["Sunrise_Sunset"]=dataset["Sunrise_Sunset"].astype("category")
dataset["Sunrise_Sunset_Codes"]=dataset["Sunrise_Sunset"].cat.codes

#Removing useless data
dataset.drop(labels=["Weather_Condition","Start_Time","End_Time"], inplace=True, axis=1)
head=dataset.head()
print(dataset.dtypes)

#Optimize for memory usage
dataset["Street_Codes"].max()
dataset["Severity"]=dataset["Severity"].astype("int8")
dataset["Distance(mi)"]=dataset["Distance(mi)"].astype("float32")
dataset["Temperature(F)"]=dataset["Temperature(F)"].astype("float16")
dataset["Humidity(%)"]=dataset["Humidity(%)"].astype("float16")
dataset["Pressure(in)"]=dataset["Pressure(in)"].astype("float16")
dataset["Visibility(mi)"]=dataset["Visibility(mi)"].astype("float16")
dataset["Wind_Speed(mph)"]=dataset["Wind_Speed(mph)"].astype("float32")
dataset["hour"]=dataset["hour"].astype("int8")
dataset["month"]=dataset["month"].astype("int8")
dataset["dayofweek"]=dataset["dayofweek"].astype("int8")
dataset["duration"]=dataset["duration"].astype("int32")
dataset["Street_Codes"]=dataset["Street_Codes"].astype("int32")

#Let's do some analysis

#Duration mean
print("Duration mean: ")
print(dataset["duration"].mean())

#Resampling
print((dataset["Severity"]==1).sum())
print((dataset["Severity"]==2).sum())
print((dataset["Severity"]==3).sum())
print((dataset["Severity"]==4).sum())
def resample(dat, col, n):
    return pd.concat([dat[dat[col]==1].sample(n, replace = True),
                      dat[dat[col]==2].sample(n, replace = False),
                       dat[dat[col]==3].sample(n, replace = False),
                        dat[dat[col]==4].sample(n, replace = True)])
                   
dataset_res=resample(dataset,"Severity", 114000)

sns.set_theme(style="whitegrid")
#Getting duration by Severity
fig, axs = plt.subplots(ncols=2, figsize=(10, 4))
sns.boxplot(x="Severity", y="duration", data=dataset_res[['duration',"Severity"]], ax=axs[0])
#sns.stripplot(x="Severity", y="duration", data=dataset[['duration',"Severity"]], size=4, color=".26", ax=axs[0])
fig.suptitle('Duration & Distance by Accidents Severity', fontsize=16)
sns.boxplot(x="Severity", y="Distance(mi)", data=dataset_res[['Distance(mi)',"Severity"]],ax=axs[1])
plt.show()

#Getting Severityies by year
sns.countplot(x='year', hue='Severity', data=dataset_res ,palette="Set2")
plt.title('Count of Accidents by Year', size=15, y=1.05)
plt.show()

#Getting Incidents by month
plt.figure(figsize=(10,5))
sns.countplot(x='month', hue='Severity', data=dataset_res ,palette="Set2")
plt.title('Count of Accidents by Month', size=15, y=1.05)
plt.show()

#Getting Incidents by day
plt.figure(figsize=(10,5))
sns.countplot(x='dayofweek', hue='Severity', data=dataset_res ,palette="Set2")
plt.title('Count of Accidents by Weekday', size=15, y=1.05)
plt.show()

#Getting Incidents by hour
plt.figure(figsize=(10,5))
sns.countplot(x='hour', hue='Severity', data=dataset_res ,palette="Set2")
plt.title('Accident Severity by hour', size=15, y=1.05)
plt.show()

#Getting street types
# create a list of top 40 most common words in street name
st_type =' '.join(dataset['Street'].unique().tolist()) # flat the array of street name
st_type = re.split(" |-", st_type) # split the long string by space and hyphen
st_type = [x[0] for x in Counter(st_type).most_common(40)] # select the 40 most common words
print('the 40 most common words')
print(*st_type, sep = ", ") 

# Remove some irrelevant words and add spaces and hyphen back
st_type= [' Rd', ' St', ' Dr', ' Ave', ' Blvd', ' Ln', ' Highway', ' Pkwy', ' Hwy', 
          ' Way', ' Ct', 'Pl', ' Road', 'US-', 'Creek', ' Cir',  'Route', 
          'I-', 'Trl', 'Pike', ' Fwy']
print(*st_type, sep = ", ")  

# for each word create a boolean column
for i in st_type:
  dataset[i.strip()] = np.where(dataset['Street'].str.contains(i, case=True, na = False), True, False)
dataset.loc[dataset['Road']==1,'Rd'] = True
dataset.loc[dataset['Highway']==1,'Hwy'] = True

# resample again
dataset_res_str=resample(dataset,"Severity", 114000)

# plot correlation
dataset_res_str['Severity'] = dataset_res_str['Severity'].astype(int)
street_corr  = dataset_res_str.loc[:,['Severity']+[x.strip() for x in st_type]].corr()
plt.figure(figsize=(20,15))
cmap = sns.diverging_palette(220, 20, sep=20, as_cmap=True)
sns.heatmap(street_corr, annot=True, cmap=cmap, center=0).set_title("Correlation (resampled data)", fontsize=16)
plt.show()


plt.figure(figsize=(15,10))

plt.plot( 'Start_Lng', 'Start_Lat', data=dataset, linestyle='', marker='o', markersize=2, color="gray", alpha=0.2, label='All Accidents')
plt.plot( 'Start_Lng', 'Start_Lat', data=dataset[dataset['Severity']==3], linestyle='', marker='o', markersize=1.5, color="blue", alpha=0.15, label='Accidents with Severity Level 3')
plt.plot( 'Start_Lng', 'Start_Lat', data=dataset[dataset['Severity']==4], linestyle='', marker='o', markersize=0.5, color="red", alpha=0.1, label='Accidents with Severity Level 4')
plt.legend(markerscale=8)
plt.xlabel('Longitude', size=12, labelpad=3)
plt.ylabel('Latitude', size=12, labelpad=3)
plt.title('Map of Accidents', size=16, y=1.05)
plt.show()

#Accidents per state
plt.figure(figsize=(20,7))
sns.countplot(x="State",data=dataset)
plt.yscale("log")
plt.title("STATES WITH NUMBER OF ACCIDENTS",fontsize=20)
plt.show()

'''
#Accidents per city
top_cities=dataset["City"].value_counts().sort_values()[-20:].reset_index()
top_cities.columns=["city","number_of_accidents"]
plt.figure(figsize=(10,7))
sns.barplot(x="city",y="number_of_accidents",data=top_cities)
plt.title("TOP 10 CITIES WITH HIGHEST NUMBER OF ACCIDENTS",fontsize=20)
plt.xticks(rotation=40)
plt.show()
'''
#dataset_res['Severity'] = dataset_res['Severity'].astype('category')
num_features = ['Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)']
fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(15, 10))
plt.subplots_adjust(hspace=0.4,wspace = 0.2)
for i, feature in enumerate(num_features, 1):    
    plt.subplot(3, 2, i)
    sns.violinplot(x=feature, y="Severity", data=dataset_res, palette="Set2")
    
    plt.xlabel('{}'.format(feature), size=12, labelpad=3)
    plt.ylabel('Severity', size=12, labelpad=3)    
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)

    plt.title('{} Feature by Severity'.format(feature), size=14, y=1.05)
fig.suptitle('Density of Accidents by Weather Features (resampled data)', fontsize=18)
plt.show()

fig, axs = plt.subplots(ncols=1, nrows=4, figsize=(15, 10))
plt.subplots_adjust(hspace=0.4,wspace = 0.6)
weather=["snowCond","rainyCond","fogCond","windyCond"]
for i, feature in enumerate(weather, 1):    
    plt.subplot(1, 4, i)
    sns.countplot(x=feature, hue='Severity', data=dataset_res ,palette="Set2")
    
    plt.xlabel('{}'.format(feature), size=12, labelpad=3)
    plt.ylabel('Accident Count', size=12, labelpad=3)    
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)
    
    plt.legend(['1', '2', '3', '4'], loc='upper right', prop={'size': 10})
    plt.title('Count of Severity in \n {} Feature'.format(feature), size=14, y=1.05)
fig.suptitle('Count of Accidents by Weather Features (resampled data)', fontsize=18)
plt.show()

#Corr Matrix
# plot correlation
dataset_res['Severity'] = dataset_res['Severity'].astype(int)
plt.figure(figsize=(30,30))
cmap = sns.diverging_palette(220, 20, sep=20, as_cmap=True)
sns.heatmap(dataset_res.corr(), annot=True,cmap=cmap, center=0).set_title("Correlation Heatmap", fontsize=14)
plt.show()

#Removing not related data
dataset=dataset.drop(["year","Distance(mi)","duration","Street", "City","County","State","Wind_Direction","Sunrise_Sunset","County_Codes", "Crossing", "Traffic_Signal"],axis=1)

dataset.info(verbose=True, memory_usage="deep")



#Cleaning vars
names=["axs","colsToDelete","columns","dataset_res","dataset_res_str","feature","fig","head","i","notWindyAcc","notWindyMean","num_features","percentage_of_missing_values","st_type","street_corr","weather","windyAcc","windyMean"]
for name in names:
    del globals()[name]
del globals()["names"]
del globals()["name"]
gc.collect()

def evaluate(y_test,y_pred, alg):
    print (classification_report(y_test, y_pred))
        
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    
    conf_matrix = pd.DataFrame(data=confmat,columns=['Predicted:1','Predicted:2','Predicted:3','Predicted:4'],index=['1','2','3','4'])
    plt.figure(figsize = (8,5))
    sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu").set_title("Confusion Matrix \n"+alg, fontsize=16)
    plt.show()
    
def getFactors(X_train, alg):
    importances = pd.DataFrame(np.zeros((X_train.shape[1], 1)), columns=['importance'], index=dataset.drop('Severity',axis=1).columns)
    
    importances.iloc[:,0] = alg.feature_importances_
    
    importances.sort_values(by='importance', inplace=True, ascending=False)
    importances30 = importances.head(30)
    
    plt.figure(figsize=(15, 10))
    sns.barplot(x='importance', y=importances30.index, data=importances30)
    
    plt.xlabel('')
    plt.tick_params(axis='x', labelsize=10)
    plt.tick_params(axis='y', labelsize=10)
    if alg == "rf":
        plt.title('Random Forest Classifier Feature Importance', size=15)
    elif alg == "gbc":
        plt.title('Gradient Boost Classifier Feature Importance', size=15)

    plt.show()
    

#Data Mining with stratify
X=dataset.drop("Severity", axis=1)
y=dataset["Severity"]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3,test_size=0.1, random_state=0, stratify=y)


#NaiveBayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)   
y_pred=gnb.predict(X_test)

evaluate(y_test, y_pred, "NaiveBayes")

#Random Forest
rf=RandomForestClassifier(n_jobs=8)
rf.fit(X_train, y_train)
y_pred=rf.predict(X_test)
    
evaluate(y_test, y_pred, "Random Forest")
getFactors(X_train, rf)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.03,test_size=0.1, random_state=0, stratify=y)

#Gradient Boost
from sklearn.ensemble import GradientBoostingClassifier
gbc=GradientBoostingClassifier(verbose=True, max_depth=50, n_estimators=10)
gbc.fit(X_train, y_train)
y_pred=gbc.predict(X_test)

evaluate(y_test, y_pred, "Gradient Boost")
getFactors(X_train, gbc)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.03,test_size=0.01, random_state=0, stratify=y)
#KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_jobs=8, weights="distance", leaf_size=10)
knn.fit(X_train, y_train)
y_pred=knn.predict(X_test)

evaluate(y_test, y_pred, "KNN")

gc.collect()
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.15,test_size=0.1, random_state=0, stratify=y)
#AdaBoost
from sklearn.ensemble import AdaBoostClassifier
rf=RandomForestClassifier(n_estimators=100, n_jobs=8, verbose=True, warm_start=True)
clf = AdaBoostClassifier(base_estimator=rf, n_estimators=5)
clf.fit(X_train, y_train)

y_pred=clf.predict(X_test)

evaluate(y_test, y_pred, "AdaBoost")


#Data mining with resampled data
gc.collect()
dataset_res = resample(dataset, 'Severity',400000)
X=dataset_res.drop("Severity", axis=1)
y=dataset_res["Severity"]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5,test_size=0.2, random_state=0, stratify=y)


#NaiveBayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)   
y_pred=gnb.predict(X_test)

evaluate(y_test, y_pred,"NaiveBayes")

#Random Forest
rf=RandomForestClassifier(n_jobs=8, verbose=True)
rf.fit(X_train, y_train)
y_pred=rf.predict(X_test)
    
evaluate(y_test, y_pred,"Random Forest")
getFactors(X_train, rf)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.1,test_size=0.2, random_state=0, stratify=y)
#Gradient Boost
from sklearn.ensemble import GradientBoostingClassifier
gbc=GradientBoostingClassifier(n_estimators=200, max_depth=10, verbose=True, warm_start=True)
gbc.fit(X_train, y_train)
y_pred=gbc.predict(X_test)

evaluate(y_test, y_pred, "Gradient Boost")
getFactors(X_train, gbc)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.05,test_size=0.05, random_state=0, stratify=y)
#KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_jobs=8, weights="distance", leaf_size=10)
knn.fit(X_train, y_train)
y_pred=knn.predict(X_test)

evaluate(y_test, y_pred,"KNN")

#AdaBoost
from sklearn.ensemble import AdaBoostClassifier
rf=RandomForestClassifier(n_estimators=100, n_jobs=8, verbose=True, warm_start=True)
clf = AdaBoostClassifier(base_estimator=rf, n_estimators=5)
clf.fit(X_train, y_train)

y_pred=clf.predict(X_test)

evaluate(y_test, y_pred, "AdaBoost")

#GridSearch rf
clf_base = RandomForestClassifier(verbose=True, warm_start=True)
grid = {'n_estimators': [100,200, 300], 'max_depth': [10,20,30,50], 'max_features': [0.1,0.2,0.3]}        
clf_rf = GridSearchCV(clf_base, grid, n_jobs=8, verbose=3, scoring="f1_weighted")

clf_rf.fit(X_train, y_train)
clf_rf.best_params_
clf_rf.best_score_

y_pred = clf_rf.predict(X_test)

evaluate(y_test, y_pred, "Random Forest with gridSearch")

#GridSearch gbc
from sklearn.ensemble import GradientBoostingClassifier
gbc=GradientBoostingClassifier()
grid={'n_estimators': [100,200,400], 'max_depth': [10,15], 'warm_start': ['True'], 'loss': ['deviance','exponential'], 'criterion': ['friedman_mse','mse'], 'max_features': ['sqrt']}
clf_gbc = GridSearchCV(gbc, grid, n_jobs=8, verbose=3, scoring="f1_weighted")
clf_gbc.fit(X_train, y_train)
clf_gbc.best_params_
clf_gbc.best_score_

y_pred=gbc.predict(X_test)

#GridSearch knn
clf_base = RandomForestClassifier(verbose=True)
grid = {'n_estimators': [100,200, 300], 'warm_start': ['True'], 'max_depth': [10,20,30,50], 'max_features': [0.1,0.2,0.3]}        
clf_rf = GridSearchCV(clf_base, grid, n_jobs=8, verbose=3, scoring="f1_weighted")

clf_rf.fit(X_train, y_train)
clf_rf.best_params_
clf_rf.best_score_

y_pred = clf_rf.predict(X_test)

evaluate(y_test, y_pred, "KNN with gridSearch")

#Random Forest
rf=RandomForestClassifier(n_estimators=400, n_jobs=8, verbose=True, warm_start=True)
rf.fit(X_train, y_train)
y_pred=rf.predict(X_test)
    
evaluate(y_test, y_pred, "Random Forest")
getFactors(X_train, rf)

#Gradient Boost
from sklearn.ensemble import GradientBoostingClassifier
gbc=GradientBoostingClassifier(n_estimators=200, max_depth=10, verbose=True, warm_start=True)
gbc.fit(X_train, y_train)
y_pred=gbc.predict(X_test)

evaluate(y_test, y_pred, "Gradient Boost")
getFactors(X_train, gbc)
