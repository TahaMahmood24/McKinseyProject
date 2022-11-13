import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

import warnings

warnings.filterwarnings('ignore')

print(pd.set_option('display.max_columns', None))

# training data
train = pd.read_csv('train.csv')

# test data
test = pd.read_csv('test.csv')
df = pd.concat([train, test], sort=False)
train.head()
print("Train vector: " + str(train.shape))
print("Test vector: " + str(test.shape))
print("DF Vector" + str(df.shape))
df["galaxy"] = df["galaxy"].astype('category')
# print(df["galaxy"])
df["galaxy"] = df["galaxy"].cat.codes
# print(df["galaxy"])
train = df[:3865]
test = df[3865:]
test = test.drop("y", axis=1)
test_res = test.copy()

train_gal = set(train["galaxy"])

s = 0
for x in train_gal:
    f = 0
    f = f + len(train.loc[train['galaxy'] == x])
    s = s + len(train.loc[train['galaxy'] == x])
    # print("The number of samples for galaxy {} is {}".format(x,f))
print(s)
print("Total distinct galaxies: {}".format(len(train_gal)))
print("Average samples per galaxy: {}".format(s / len(train_gal)))

test_gal = set(test["galaxy"])
s = 0
for x in test_gal:
    s = s + len(test.loc[test['galaxy'] == x])
print("Total distinct galaxies: {}".format(len(test_gal)))
print("Average samples per galaxy: {}".format(s / len(test_gal)))

"""
def cross_validation_loop(data, cor):
    labels = data['y']
    # print(len(labels))
    data = data.drop('galaxy', axis=1)
    data = data.drop('y', axis=1)
    # print(data)
    correlation = abs(data.corrwith(labels))
    # print(correlation)
    columns = correlation.nlargest(cor).index
    data = data[columns]

    imp = SimpleImputer(missing_values=np.nan, strategy='mean').fit(data)
    data = imp.transform(data)

    scaler = StandardScaler().fit(data)
    data = scaler.transform(data)

    estimator = XGBRegressor(n_estimators=500)

    cv_results = cross_validate(estimator, data, labels, cv=4, scoring='neg_root_mean_squared_error')
    # print(cv_results)
    error = np.mean(cv_results['test_score'])

    return error


train_gal = set(train["galaxy"])
# print(train_gal)
train_gal.remove(126)


def loop_train(cor):
    errors = []
    for gal in train_gal:
        index = train.index[train['galaxy'] == gal]
        data = train.loc[index]
        # print(data.head())
        errors.append(cross_validation_loop(data, cor))
    return np.mean(errors)


cor = [20, 21, 25, 30, 40, 50, 60, 70, 80]
# print(cor)
errors = []
for x in cor:
    print("The error of {} is {}".format(x, loop_train(x)))
    errors.append(loop_train(x))

print(errors)

"""

def test_loop(data, test_data):
    labels = data['y']
    data = data.drop('galaxy', axis=1)
    data = data.drop('y', axis=1)
    correlation = abs(data.corrwith(labels))
    columns = correlation.nlargest(21).index

    train_labels = labels
    train_data = data[columns]
    test_data = test_data[columns]

    imp = SimpleImputer(missing_values=np.nan, strategy='mean').fit(train_data)
    train_data = imp.transform(train_data)
    test_data = imp.transform(test_data)

    scaler = StandardScaler().fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)

    train_data_train, train_data_valid, train_label_train, train_label_valid = train_test_split(train_data, train_labels, train_size=0.8, test_size=0.2, random_state=0)
    model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
    model.fit(train_data_train, train_label_train, early_stopping_rounds = 5, eval_set=[(train_data_valid, train_label_valid)], verbose=False)

    predictions = model.predict(test_data)
    return predictions

test=test_res
test=test.sort_values(by=['galaxy'])
test_pred = pd.DataFrame(0, index=np.arange(len(test)), columns=["predicted_y"])
#print(test_pred)
i=0
for gal in test_gal:
    #print(test.loc[test['galaxy'] == gal ])
    count = len(test.loc[test['galaxy'] == gal])
    print(count, gal)
    index = train.index[train['galaxy'] == gal]
    data = train.loc[index]
    #print(data)
    pred = test_loop(data,test.loc[test['galaxy']==gal])
    test_pred.loc[i:i+count-1,'predicted_y'] = pred
    i = i + count

test["predicted_y"]=test_pred.to_numpy()
test.sort_index(inplace=True)
predictions = test["predicted_y"]
print(predictions)

index = predictions
pot_inc = -np.log(index+0.01)+3
p2= pot_inc**2
print(p2)
ss = pd.DataFrame({
    'Index':test.index,
    'pred': predictions,
    'opt_pred':0,
    'eei':test['existence expectancy index'], # So we can split into low and high EEI galaxies
})

ss.loc[p2.nlargest(400).index, 'opt_pred']=100
pp = ss.iloc[:400]
print(pp.shape)
ss=ss.sort_values('pred')
ss = ss.assign(New_Index = range(len(ss)))
ss.iloc[400:600].opt_pred = 50
ss=ss.sort_index()
increase = (ss['opt_pred']*p2)/1000
#plt.plot(ss.New_Index, ss.opt_pred,  color='blue')
plt.scatter(ss.New_Index, ss.opt_pred,  color='blue')
plt.xlabel("Index")
plt.ylabel("Optional Predictions")
plt.show()
print(sum(increase), ss.loc[ss.eei < 0.7, 'opt_pred'].sum(), "{:.10f}".format(ss['opt_pred'].sum()))
ss[['Index', 'pred', 'opt_pred']].to_csv('submission_XGBoost.csv', index=False)

