#%%
#import required packages for the following functions 
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import re
import ast 
from sklearn.inspection import permutation_importance
import geopy.distance as dist
from sklearn.metrics import r2_score

# Calculates the distance from each listing from the city center
# (city center coordinates obtained from google)
def getKilometersFromCenter(data, city):
    kilometers = np.zeros(data.shape[0])
    if city == 'copenhagen':
        center =  [55.6761, 12.5683]
    elif city == 'stockholm':
        center =  [59.3293, 18.0686]
    elif city == 'london':
        center = [51.5074, -0.1272]
    elif city == 'paris':
        center = [48.8566, 2.3522]
    else:
        center = [52.5373, 13.3603]
    for i in range(data.shape[0]):
        # data[i, 0] = (data[i, 0]- center[0])*111
        # data[i, 1] = (data[i, 1] - center[1])*111
        # kilometers[i] = np.linalg.norm(data[i, 0]-data[i, 1])
        kilometers[i] = dist.geodesic(data[i], center).km
        if(data[i, 0] <  center[0]): 
            data[i, 0] = -dist.geodesic((data[i, 0], center[1]), center).km
        else: 
            data[i, 0] = dist.geodesic((data[i, 0], center[1]), center).km
        if(data[i, 1] <  center[1]):
            data[i, 1] = -dist.geodesic((center[0], data[i, 1]), center).km
        else:
            data[i, 1] = dist.geodesic((center[0], data[i, 1]), center).km
    return kilometers, data, center

def string_features_to_num(array):
    dictOfWords = { i : np.unique(array)[i] for i in range(0, len(np.unique(array)) ) }
    dictOfWords = {v: k for k, v in dictOfWords.items()}
    
    values = np.zeros(len(array))
    for i in range(len(array)):
        values[i] = dictOfWords.get(array[i])
        
    return values, dictOfWords

def selectPropertyType(type, X, y):
    xlist = []
    ylist = []
    for i in range(X.shape[0]):
        if(X[i,-5] == type):
            xlist.append(X[i,:])
            ylist.append(y[i])
    return np.array(xlist), np.array(ylist)

# One hot encoding for the type of property 
def oneHotEncodingPropertyType(type, X):
    isType = np.zeros(X.shape[0])
    print(X.shape)
    for i in range(X.shape[0]):
        if(X[i] == type):
            isType[i] = 1.0
    return isType

# Scatter diagram showing predicted price vs real price
# y = x axis added for reference
def plotPredVsReal(yreal, ypred, limit=750):
    plt.scatter(yreal, ypred, label="pred vs true")
    plt.plot(yreal, yreal, label="y=x", c='red')
    plt.xlabel("y-real")
    plt.ylabel("y-pred")
    plt.xlim((0,limit))
    plt.ylim((0,limit))
    plt.legend()
    plt.show

def host_Since_fix(X):
    yearsSince = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        yearsSince[i] = 2022 - int(X[i].split('-')[0])
    return yearsSince

# Create a bar chart showing the importance of features in decreasing order
def plot_feature_importance(best_model, X, y, featureDict):
    permut = permutation_importance(best_model, X, y, scoring='r2')
    importance = permut.importances_mean
    

    features = [featureDict.get(key) for key in np.argsort(importance)]
    plt.figure(figsize=(6, 14))
    plt.barh(range(len(features)), importance[np.argsort(importance)])
    plt.yticks(range(len(features)),features)
    # plt.bar(range(len(importance)), importance)
    # plt.gca().set(xticks=range(len(importance)), xticklabels = featureNames)
    # plt.xticks(rotation=90)
    plt.show()

# Heat map that showd the location of the city, its center, and the price
def plotCity(longAndLat, y):
    heatmapColors = np.clip(y, 0, 800)
    indexes = np.argsort(y)
    # c=scaler.fit_transform(y.reshape(-1,1))
    plt.scatter(longAndLat[indexes,1], longAndLat[indexes,0], c=heatmapColors[indexes], cmap=mpl.colormaps['hot'])
    plt.colorbar()
    plt.scatter(0, 0, label='center')
    plt.legend()
    plt.show()

# One hot encoding for all amenities (1 if present, 0 if not)
def amenitiesOneHot(X, amenityName):
    amenities = []
    for i in range(X.shape[0]):
        amenities.append(ast.literal_eval(X[i,-1]))

    amenity = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        for j in range(len(amenities[i])):
            if(amenities[i][j]==amenityName):
                amenity[i] = 1
    return amenity


def cleanBathroomFeature(X, y, featureNum):
    bathroomErr = []
    for i in range(X.shape[0]):
        try:
            X[i,featureNum] = float(X[i,featureNum].split()[0])
        except ValueError:
            bathroomErr.append(i)
    bathroomErr = np.array(bathroomErr)
    X = np.delete(X, bathroomErr, axis=0)
    y = np.delete(y, bathroomErr)
    return X, y

# Convert local currency to USD
def cleanCurrency(y, city):
    exchange =1
    if city=='copenhagen': exchange = 0.1579
    elif city=='stockholm': exchange = 0.088507
    elif city=='london': exchange = 1.1594
    else: exchange = 1.0151

    for i in range(len(y)):
        y[i] = float(re.sub(",", "", (y[i][1:])))*exchange
    return y

# One hot encoding for the neighbourhood that the listing is in
def neighbourhood_onehot(X):
    unique = np.unique(X)
    hoods = np.zeros((X.shape[0], len(unique)))
    dictOfHoods = { i : unique[i] for i in range(0, len(unique) ) }
    dictOfHoods = {v: k for k, v in dictOfHoods.items()}
    for i in range(X.shape[0]):
        hoods[i, dictOfHoods.get(X[i])] = 1
    indexes = []
    for i in range(hoods.shape[1]):
        if np.sum(hoods[:,i])/hoods.shape[0] > 0.01:
            indexes.append(i)
    return hoods[:,indexes], unique[indexes]

def getFeatureNames(neighbourhood_names, roomtype_names, property_names):
    featureNames = ['accommodates', 'bathrooms', 'bedrooms', 'beds', 'review_scores_rating', 'review_scores_location',
    'review_scores_value', 'review_scores_cleanliness', 'reviews_per_month', 'number_of_reviews', 'calculated_host_listings_count', 
    'minimum_nights', 'availability_30', 'availability_60', 'availability_90', 'availability_365', #'Entire rental unit', 'Entire condo', 'Private room in rental unit', #'Private room in condo', 'Entire villa', 'Entire loft', 
    'Private patio or balcony', 'TV', 'Wifi', 'Bed linens', 'Breakfast', 'Bathtub', 'Washer', 'Elevator', 'Pool', 'Km_from_center', 'km_from_latitude', 
    'km_from_longitude', 'Host_years']

    for name in neighbourhood_names:
        featureNames.append('NBHD: ' + name)
    for name in roomtype_names:
        featureNames.append('Room: ' +name)
    for name in property_names:
        featureNames.append('Prop: ' +name)

    featureDict = { i : featureNames[i] for i in range(len(featureNames)) }

    return featureNames, featureDict


# %%
def getCityData(city):
    data = pd.read_csv(('listings/listings_' + city + '.csv.gz'), compression='gzip')
    data = data[['number_of_reviews_ltm', 'accommodates','bathrooms_text','bedrooms', 'beds', 'host_since', 'neighbourhood_cleansed', 'review_scores_rating','review_scores_location','review_scores_value','review_scores_cleanliness','reviews_per_month','number_of_reviews','calculated_host_listings_count','minimum_nights','availability_30', 'availability_60', 'availability_90', 'availability_365', 'property_type','room_type','amenities','latitude','longitude','price']]
    
    data = data.dropna()

    datanp = np.array(data)
    

    datanp = datanp[datanp[:,0] != 0]
    datanp = datanp[:, 1:]

    

    X = datanp[:,:-1]
    y = datanp[:,-1]

    X, y = cleanBathroomFeature(X, y, 1)
    
    y = cleanCurrency(y, city)
    
    X = X[np.argsort(y)[10:-10], :]
    y = y[np.argsort(y)[10:-10]]
    #re-shuffling the data
    con_xy = np.hstack((X,y.reshape(-1,1)))
    np.random.shuffle(con_xy)
    X = con_xy[:,:-1]
    y = con_xy[:,-1]

    neighbourhoods, neighbourhood_names = neighbourhood_onehot(X[:,5])
    X = np.hstack((X[:,:5], X[:,6:]))

    # X[:,-5], propertytype_dict = string_features_to_num(X[:, -5])
    # X[:,-4], roomtype_dict = string_features_to_num(X[:, -4])

    # print(propertytype_dict)
    # print(list(enumerate(np.bincount(np.array(X[:, -5],dtype='int64')))))
    # X, y = selectPropertyType(propertytype_dict.get('Entire home'), X, y)
    
    latitudeAndLongitude = X[:,-2:]
    kilometers, longAndLat, center = getKilometersFromCenter(latitudeAndLongitude, city)
    X = X[:,:X.shape[1]-2]

    onehotSource = X[:,-3:]
    X = X[:,:X.shape[1]-3]

    roomtype, roomtype_names = neighbourhood_onehot(onehotSource[:, 1])
    propertytype, propertytype_names = neighbourhood_onehot(onehotSource[:, 0])
    # isType = oneHotEncodingPropertyType(propertytype_dict.get('Entire rental unit'), onehotSource[:,0])
    # X = np.hstack((X, isType.reshape(-1,1)))
    # isType = oneHotEncodingPropertyType(propertytype_dict.get('Entire condo'), onehotSource[:, 0])
    # X = np.hstack((X, isType.reshape(-1,1)))
    # isType = oneHotEncodingPropertyType(propertytype_dict.get('Private room in rental unit'), onehotSource[:,0])
    # X = np.hstack((X, isType.reshape(-1,1)))
    # isType = oneHotEncodingPropertyType(propertytype_dict.get('Private room in condo'), onehotSource)
    # X = np.hstack((X, isType.reshape(-1,1)))
    # isType = oneHotEncodingPropertyType(propertytype_dict.get('Entire villa'), onehotSource)
    # X = np.hstack((X, isType.reshape(-1,1)))
    # isType = oneHotEncodingPropertyType(propertytype_dict.get('Entire loft'), onehotSource)
    # X = np.hstack((X, isType.reshape(-1,1)))

    patio = amenitiesOneHot(onehotSource, 'Private patio or balcony')
    tv = amenitiesOneHot(onehotSource, 'TV')
    wifi = amenitiesOneHot(onehotSource, 'Wifi')
    bedlinens = amenitiesOneHot(onehotSource, 'Bed linens')
    breakfast = amenitiesOneHot(onehotSource, 'Breakfast')
    bathtub = amenitiesOneHot(onehotSource, 'Bathtub')
    washer = amenitiesOneHot(onehotSource, 'Washer')
    elevator = amenitiesOneHot(onehotSource, 'Elevator')
    pool = amenitiesOneHot(onehotSource, 'Pool')
    hostsince = host_Since_fix(X[:,4])

    onehot = np.hstack((tv.reshape(-1,1), patio.reshape(-1,1)))
    onehot = np.hstack((onehot, wifi.reshape(-1,1)))
    onehot = np.hstack((onehot, bedlinens.reshape(-1,1)))
    onehot = np.hstack((onehot, breakfast.reshape(-1,1)))
    onehot = np.hstack((onehot, bathtub.reshape(-1,1)))
    onehot = np.hstack((onehot, washer.reshape(-1,1)))
    onehot = np.hstack((onehot, elevator.reshape(-1,1)))
    onehot = np.hstack((onehot, pool.reshape(-1,1)))
    onehot = np.hstack((onehot, kilometers.reshape(-1,1)))
    onehot = np.hstack((onehot, longAndLat))
    onehot = np.hstack((onehot, hostsince.reshape(-1,1)))
    

    X = np.hstack((X[:,:4], X[:,5:]))
    X = np.hstack((X, onehot))
    X = np.hstack((X, neighbourhoods))
    X = np.hstack((X, roomtype))
    X = np.hstack((X, propertytype))

    featureNames, featureDict = getFeatureNames(neighbourhood_names, roomtype_names, propertytype_names)

    
    return X, y, featureNames, featureDict
# %%
city = 'copenhagen'
X, y, featureNames, featureDict = getCityData(city)


# %%
# SVM implemented with leave-one-out cross-validation
# from sklearn.model_selection import LeaveOneOut
# from sklearn.model_selection import cross_val_score
# from sklearn.svm import SVR


# X_data = np.asarray(X).astype('float32')
# y_data = np.asarray(y).astype('float32')
# # adding the price column back to feature array before shuffling
# all_data = np.append(X_data, y_data.reshape(-1,1), axis=1)
# np.random.shuffle(all_data)
# X_data = all_data[:,:-1]
# y_data = all_data[:,-1]

# svr_model = SVR(C=60, epsilon=0.95)
# loo = LeaveOneOut()
# scores = cross_val_score(estimator=svr_model, 
#                         X=X_data, 
#                         y=y_data, 
#                         scoring='r2',
#                         cv=loo,
#                         n_jobs=-1)

# mean_score = np.mean(scores)
# print(mean_score)

# %%
# Linear regression implemented with leave-one-out cross-validation (dataset too large for leave-one-out)
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR, NuSVR
from sklearn import metrics
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


X_data = np.asarray(X).astype('float32')
y_data = np.asarray(y).astype('float32')

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2) 
# adding the price column back to feature array before shuffling

lin_reg = make_pipeline(StandardScaler(), LinearRegression())
# linear_pipeline.fit(X_train, y_train)
# lin_reg = LinearRegression()
cv = LeaveOneOut()
scores = cross_val_score(estimator=lin_reg, 
                        X=X_train, 
                        y=y_train, 
                        scoring='neg_mean_squared_error',
                        cv=cv,
                        n_jobs=-1)
print(scores)

#%%
mean_score = np.mean(np.sqrt(np.absolute(scores)))
print(mean_score)


#%%
# Neural network on chosen city

X = np.asarray(X).astype('float32')
y = np.asarray(y).astype('float32')



X_train, X_testval, y_train, y_testval = train_test_split(X, y, test_size=0.20)
X_test, X_val, y_test, y_val = train_test_split(X_testval, y_testval, test_size=0.5)

print(X_train.shape, X_test.shape, X_val.shape)  


# %%
import time
def LeaveOneOut(model, X, y, n_points):
    start = time.time()
    X_ = X[:n_points,:]
    y_ = y[:n_points]
    predictions = []
    print(X.shape[0])
    for i in range(X_.shape[0]):
        if (i % 5 == 0):
            print("iteraton: ", i, ". Time since start:", "%.2f" % (time.time()-start))
        X_train = np.delete(X_, i, axis=0)
        y_train = np.delete(y_, i)
        X_test = X_[i, :].reshape(1, -1)
        model.fit(X_train, y_train)
        predictions.append(model.predict(X_test)) 

    return predictions

#%%
# Applying leave-one-out cross validation using linear regression model
from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
n_points = 500
predictions_LR = LeaveOneOut(linear_model, X, y, n_points)
print(city, "Linear regression leave-one-out r2:", round(r2_score(y[:n_points], predictions_LR),4))
plotPredVsReal(yreal=y[:n_points], ypred=predictions_LR, limit=750)

#%%
# Applying leave-one-out cross validation using NuSVR model
# Takes 21 hours and 20 minutes
from sklearn.svm import NuSVR
nusvr = NuSVR(C=60, nu=0.8)
n_points = 1000
predictions_NuSVR = LeaveOneOut(nusvr, X, y, n_points)
print(city, "NuSVR leave-one-out r2:", round(r2_score(y[:n_points], predictions_NuSVR),4))
plotPredVsReal(yreal=y[:n_points], ypred=predictions_NuSVR, limit=750)

#%%
# Applying leave-one-out cross validation using Gradient boosting model
# takes 10 hours and 37 minutes
from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor()
n_points = 500
predictions_gbr = LeaveOneOut(gbr, X, y, n_points=n_points)
print(city, "Linear regression leave-one-out r2:", round(r2_score(y[:n_points], predictions_gbr),4))
plotPredVsReal(yreal=y[:n_points], ypred=predictions_gbr, limit=750)
# %%
#normalize our data
from sklearn.preprocessing import StandardScaler
X_scaler = StandardScaler()
X_train_sc = X_scaler.fit_transform(X_train)
X_val_sc = X_scaler.transform(X_val)
X_test_sc = X_scaler.transform(X_test)

X_train_sc.shape
# %%

from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, SGD, RMSprop, Adagrad

input_dim = X_train_sc.shape[1]
# SGD: larger learning rate (0.06), Adam: smaller learning rate
lr = 0.0001
p = 0.2
model = Sequential()
#found 16 to be optimal
model.add(Dense(units=64, input_dim=input_dim, kernel_initializer='normal', activation='relu'))
# p = 0.2 optimal
model.add(Dropout(p))
model.add(Dense(units=256, kernel_initializer='normal', activation='relu'))
# model.add(Dropout(p))
# model.add(Dense(units=128, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, activation='linear', kernel_initializer='normal'))
print(model.summary())
from tensorflow.keras.callbacks import EarlyStopping
callback = EarlyStopping(monitor = 'val_msle'
                        , patience = 3
                        , verbose = 1
                        , mode = 'auto'
                        , min_delta = 0.00001)

num_epochs = 400

#larger --> increase stockholm score
#smaller --> increase copen score
# too large / too small, predictions converge to same value
b_size = 16
#optimizers: adam, sgd, 
model.compile(loss="msle", optimizer=Adam(learning_rate=lr), metrics = ['msle'])

history = model.fit(X_train_sc, y_train,
                   batch_size = b_size,
                   epochs = num_epochs,
                   verbose = 1,
                   validation_data=(X_val_sc, y_val),
                   callbacks = [callback])

import matplotlib.pyplot as plt
plt.plot(history.history['msle'])
plt.plot(history.history['val_msle'])
plt.ylabel('msle')
plt.xlabel('epoch')
plt.title('Neural Network')
plt.legend(['train', 'validation'])
plt.show()

c_pred = model.predict(X_val_sc)
plotPredVsReal(y_val, c_pred, 750)
print("Val r2:", round(r2_score(y_val, c_pred),4))
# %%

c_pred = model.predict(X_test_sc)
plotPredVsReal(y_test, c_pred, 750)
print("Test r2:", round(r2_score(y_test, c_pred),4))

c_pred = model.predict(X_train_sc)
plotPredVsReal(y_train, c_pred, 750)
print("Train r2:", round(r2_score(y_train, c_pred),4))

# %%
plot_feature_importance(model, X_val_sc, y_val, featureDict)




# %%
from sklearn.ensemble import RandomForestRegressor
best_score = 0
best_model = RandomForestRegressor()
for i in range(8):
    regr_model = RandomForestRegressor(max_depth=int(i*0.5+4), random_state=0, n_estimators=12*i+20)
    regr_model.fit(X_train, y_train)
    score = regr_model.score(X_val, y_val)
    if(score >best_score):
        best_score = score
        best_model = regr_model


print(best_model)
regr_train_score = best_model.score(X_train, y_train)
regr_test_score = best_model.score(X_test, y_test)
regr_val_score = best_model.score(X_val, y_val)

print("Training score: ", regr_train_score)
print("Test score: ", regr_test_score)
print("Validation score: ", regr_val_score)


plotPredVsReal(y_test, best_model.predict(X_test), 750)


# %%
from sklearn.svm import NuSVR
best_score = 0
best_model = NuSVR()
for i in range(20):
    svr_model = NuSVR(C=4*i+1, nu=0.04*i+0.005)
    svr_model.fit(X_train, y_train)
    score = svr_model.score(X_val, y_val)
    if(score >best_score):
        best_score = score
        best_model = svr_model



svr_train_score = best_model.score(X_train, y_train)
svr_test_score = best_model.score(X_test, y_test)
svr_val_score = best_model.score(X_val, y_val)
print("Training score:", svr_train_score)
print("Test score:", svr_test_score)
print("Validation score:", svr_val_score)

plotPredVsReal(y_test, best_model.predict(X_test), 750)


# %%





import math
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

import xgboost as xgb



X_train = pd.DataFrame(X_train, dtype='float')
X_test = pd.DataFrame(X_test, dtype='float')
X_val = pd.DataFrame(X_val, dtype='float')


y_train = pd.DataFrame(y_train, dtype='float')
y_test = pd.DataFrame(y_test, dtype='float')
y_val = pd.DataFrame(y_val, dtype='float')

# Fitting the model
# learning_rate=0.3, gamma =0.3 ,max_depth=3, n_estimators=17,


eval_set = [(X_val, y_val)]
# 
xgb_reg = xgb.XGBRegressor(learning_rate=0.22, max_depth=3, subsample=0.9, colsample_bytree = 0.7, n_estimators=100, eval_metric="rmse")
# , verbose=True
xgb_reg.fit(X_train, y_train, eval_set=eval_set)
training_preds_xgb_reg = xgb_reg.predict(X_train)
val_preds_xgb_reg = xgb_reg.predict(X_val)
test_preds_xgb_reg = xgb_reg.predict(X_test)

# Printing the results

print("\nTraining r2:", round(r2_score(y_train, training_preds_xgb_reg),4))
print("Test r2:", round(r2_score(y_test, test_preds_xgb_reg),4))
print("Validation r2:", round(r2_score(y_val, val_preds_xgb_reg),4))

# Producing a dataframe of feature importances
ft_weights_xgb_reg = pd.DataFrame(xgb_reg.feature_importances_, columns=['weight'], index=X_train.columns)
ft_weights_xgb_reg.sort_values('weight', inplace=True)

# Plotting feature importances
plt.figure(figsize=(8,10))
plt.barh(list(ft_weights_xgb_reg.index), list(ft_weights_xgb_reg.weight), align='center') 
plt.title("Feature importances in the XGBoost model", fontsize=14)
# plt.gca().set(range(len(featureNames)), yticklabels = featureNames)
plt.xlabel("Feature importance")
plt.margins(y=0.01)
plt.show()


# %%
