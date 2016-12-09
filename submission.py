import pandas as pd
import numpy as np
import time
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from tqdm import tqdm

data = pd.read_csv('calls_Irina/train_2011_2012_2013.csv',sep=';',usecols=['DATE','ASS_ASSIGNMENT','CSPL_RECEIVED_CALLS'])
subm_data = pd.read_table('calls_Irina/submission.txt')

data['time'] = data['DATE'].apply(lambda x: time.strptime(x, '%Y-%m-%d %H:%M:%S.000'))
subm_data['time'] = subm_data['DATE'].apply(lambda x: time.strptime(x, '%Y-%m-%d %H:%M:%S.000'))

data['day_week'] = data['time'].apply(lambda x: int(time.strftime("%w", x)))
data['day_month'] = data['time'].apply(lambda x: int(time.strftime("%d", x)))
data['month'] = data['time'].apply(lambda x: int(time.strftime("%m", x)))
data['time_absolut'] = data['time'].apply(time.mktime)
#data['hours'] = data['time'].apply(lambda x: float(time.strftime("%H", x)))
data['year'] = data['time'].apply(lambda x: float(time.strftime("%Y", x)))
data['if_weekend'] = data['day_week'].apply(lambda x: 1 if x == 0 or x == 6 else 0)

data['hours'] = data['time'].apply(lambda x: float(time.strftime("%H", x))+np.sign(float(time.strftime("%M", x)))*0.5)

data['if_night'] = data['hours'].apply(lambda x: 1 if x >= 7.5 and x <= 23.5 else 0)

### New feature: hours since the start of the day

subm_data['time_absolut'] = subm_data['time'].apply(time.mktime)
subm_data['day_week'] = subm_data['time'].apply(lambda x: int(time.strftime("%w", x)))
subm_data['day_month'] = subm_data['time'].apply(lambda x: int(time.strftime("%d", x)))
subm_data['month'] = subm_data['time'].apply(lambda x: int(time.strftime("%m", x)))
subm_data['time_absolut'] = subm_data['time'].apply(time.mktime)
# subm_data['hours'] = subm_data['time'].apply(lambda x: float(time.strftime("%H", x)))
subm_data['hours'] = subm_data['time'].apply(lambda x: float(time.strftime("%H", x))+np.sign(float(time.strftime("%M", x)))*0.5)



subm_data['year'] = subm_data['time'].apply(lambda x: float(time.strftime("%Y", x)))
subm_data['if_weekend'] = subm_data['day_week'].apply(lambda x: 1 if x == 0 or x == 6 else 0)
subm_data['if_night'] = subm_data['hours'].apply(lambda x: 1 if x >= 7.5 and x <= 23.5 else 0)

d = data.groupby(['hours','year','month','day_month','day_week',
                  'ASS_ASSIGNMENT','if_weekend','if_night',
                  'time','time_absolut'],as_index=False)['CSPL_RECEIVED_CALLS'].sum()
data_merged = d

cats = data['ASS_ASSIGNMENT'].unique()

#models << xgb
models_rf = {}
indexes = ['hours','year','month','day_month','day_week','if_weekend','if_night']
for cat in tqdm(cats):
    if (cat not in models_rf) and (cat != 'Téléphonie'):
        print(cat)
        print(len(data[data['ASS_ASSIGNMENT'] == cat]))
        start_time = time.time()
        data_sub = data[data['ASS_ASSIGNMENT'] == cat]
        data_sub = data_sub[data_sub['year'] < 2013]
        X = data_sub[indexes]
        Y = data_sub['CSPL_RECEIVED_CALLS']
#         model = xgb.XGBRegressor(max_depth=3, learning_rate=0.5, n_estimators=80, silent=False,
#                                  objective='reg:linear', nthread=-1, gamma=0, min_child_weight=1,
#                                  max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1,
#                                  reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, seed=0, missing=None)

#         X_train, X_test, y_train, y_test = train_test_split(X,Y)
        model = RandomForestRegressor(n_estimators=20, criterion='mse', max_depth=None,
                              min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                              max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07,
                              bootstrap=True, oob_score=False, n_jobs=-1,
                              random_state=None, verbose=0, warm_start=False)

        model.fit(X,Y)
#         scores = cross_val_score(model,X,Y,scoring=scorer)
#         print("eval: ",scores)

        models_rf[cat] = model
        print("--- %s seconds ---" % (time.time() - start_time))

indexes = ['hours','year','month','day_month','day_week','if_weekend','if_night']

for cat in tqdm(subm_data['ASS_ASSIGNMENT'].unique()):
    print(cat)
    mask = subm_data['ASS_ASSIGNMENT'] == cat
    if cat == 'Téléphonie':
        # here must be telephonie)))
        data_sub1 = data[data['ASS_ASSIGNMENT'] == 'Téléphonie']

        def get_max(x):
            tmp = data_sub1.query('hours >= @x.hours - 4 & hours <= @x.hours + 4 & day_week == @x.day_week & time_absolut <= @x.time_absolut')
            return tmp['CSPL_RECEIVED_CALLS'].max()

        data_subm_sub1 = subm_data[subm_data['ASS_ASSIGNMENT'] == 'Téléphonie']
        data_subm_sub1['max_in_hour'] = data_subm_sub1.apply(get_max, axis=1)
        res = data_subm_sub1['max_in_hour']

#     else:
#         data_sub1 = data[data['ASS_ASSIGNMENT'] == cat]

#         def get_max(x):
#             tmp = data_sub1.query('hours >= @x.hours -1 & hours <= @x.hours + 1 & day_week == @x.day_week & time_absolut >= @x.time_absolut')
#             return tmp['CSPL_RECEIVED_CALLS'].max()

#         data_subm_sub1 = subm_data[subm_data['ASS_ASSIGNMENT'] == cat]
#         data_subm_sub1['max_in_hour'] = data_subm_sub1.apply(get_max, axis=1)
#         res = data_subm_sub1['max_in_hour']

#         print(res)
#         break
#         print(data_sub1['day_week'].unique())
    else:
        if cat == 'Nuit':
        # here must be telephonie)))
            data_sub1 = data[data['ASS_ASSIGNMENT'] == 'Nuit']

            def get_max(x):
                tmp = data_sub1.query('hours >= @x.hours - 4 & hours <= @x.hours + 4 & day_week == @x.day_week & time_absolut <= @x.time_absolut & time_absolut >= (@x.time_absolut - 3*791200)')
                return tmp['CSPL_RECEIVED_CALLS'].max()

            data_subm_sub1 = subm_data[subm_data['ASS_ASSIGNMENT'] == 'Nuit']
            data_subm_sub1['max_in_hour'] = data_subm_sub1.apply(get_max, axis=1)
            res = data_subm_sub1['max_in_hour']
        else:
            data_sub1 = data[data['ASS_ASSIGNMENT'] == cat]

            def get_max(x):
                tmp = data_sub1.query('hours >= @x.hours - 4 & hours <= @x.hours + 4 & day_week == @x.day_week & time_absolut <= @x.time_absolut & time_absolut >= (@x.time_absolut - 3*791200)')
                return tmp['CSPL_RECEIVED_CALLS'].max()

            data_subm_sub1 = subm_data[subm_data['ASS_ASSIGNMENT'] == cat]
            data_subm_sub1['max_in_hour'] = data_subm_sub1.apply(get_max, axis=1)
            res = data_subm_sub1['max_in_hour']


    #         continue
            #predict by random forest
            res2 = models_rf[cat].predict(subm_data[mask][indexes])

            if cat == 'CAT':
                res2 = res2*(np.exp(-res2/100)*2+1)
            if cat == 'Tech. Axa':
                res2 = res2*(np.exp(-res2/100)*2+1)

#             if cat == 'Gestion - Accueil Telephonique':
#                 res = res*1.2

            res = pd.Series(res, index=subm_data[mask].index)
            res2 = pd.Series(res2, index=subm_data[mask].index)
            res = pd.concat([res, res2], axis=1)
            res = res.fillna(0)
            res = res.max(axis=1)
    if np.sum(pd.isnull(res)) > 0:
        print(res)
        print('----')
    subm_data.ix[mask, 'xgb_pred'] = res

subm_data['xgb_pred'] = subm_data['xgb_pred'].apply(lambda x: int(np.round(x)))
subm_data = subm_data.drop('prediction', 1)
subm_data = subm_data.rename(index=str, columns={'xgb_pred':'prediction'})
subm_data[['DATE','ASS_ASSIGNMENT','prediction']].to_csv('submission_xgb.txt',sep='\t',index=False)
