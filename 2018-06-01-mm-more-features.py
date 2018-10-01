
# coding: utf-8

# In[1]:


import xgboost as xgb
import numpy as np
import pandas as pd
import math
import pyarrow.parquet as pq
import gc
import re
import seaborn
import matplotlib.pyplot as plt

from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics import fbeta_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix, hstack
from collections import Counter
from sklearn.metrics import log_loss
from tachis.sklearn.extraction import exoctet
from tachis.sklearn.xgb import save_xjs_model

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


def replace_low_freq(df, col, min_count, replace_value=''):
    """
    Replaces cells in the dataframe for column :col: with :replace_value: where occurrences of values is less than
    :min_count:. Returns a copy of :param df:

    >>> df = replace_low_freq(df, 'partner_subid', 1000)
    :param df: the DataFrame
    :param col: the col to check for occurrences
    :param min_count: the minimum number of times each value in col needs to occur
    :return: a copy of df with values of col that occur less than min_count times replaced with replace_value
    """
    value_counts = df[col].value_counts()
    to_remove = value_counts[value_counts < min_count].index
    df = df.copy()
    df.loc[df[col].isin(to_remove), col] = replace_value
    return df


# In[3]:


train = pq.ParquetDataset('marvel_data/balanced_train/').read().to_pandas()


# In[4]:


# dev = pq.ParquetDataset('marvel_data/dev/').read().to_pandas()
test = pq.ParquetDataset('marvel_data/test/').read().to_pandas()


# In[5]:


len(train[train.target == 1])


# In[6]:


len(train[train.target == 0])


# In[7]:


train['is_rt'] = np.zeros((train.shape[0], 1))


# In[8]:


train.loc[pd.isna(train['rt_ts']), 'is_rt'] = False


# In[9]:


train.loc[pd.notna(train['rt_ts']), 'is_rt'] = True


# In[12]:


test['is_rt'] = np.zeros((test.shape[0], 1))
test.loc[pd.isna(test['rt_ts']), 'is_rt'] = False
test.loc[pd.notna(test['rt_ts']), 'is_rt'] = True


# In[11]:


train


# In[13]:


train.dtypes


# In[13]:


train['time'] = pd.to_datetime(train.time)


# In[14]:


import datetime as datetime


# In[16]:


train.sort_values('epoch')


# In[15]:


features = ['domain', 'root_domain', 'country', 'region', 'city', 'browser', 'browser_family', 'device_name', 'source_id', 'subid', 'device_type', 'adult', 'is_rt', 'tld']


# In[16]:


dev, test = train_test_split(test, test_size=.5, shuffle=True, stratify=test.target, random_state=12398)


# In[17]:


dev.shape, test.shape


# In[18]:


for feature in features:
    train = replace_low_freq(train, feature, 30)


# In[19]:


ua_splitter = re.compile(r'\W+')
subid_splitter = re.compile(r'[^a-zA-Z0-9]')
def get_features(df, features):
    for row in df.itertuples(index=False):
        d = {}
        for feature in features:
            val = getattr(row, feature)
            if val is None or val == '' or isinstance(val, float) and math.isnan(val):
                continue
            if feature == 'ua':
                parts = ua_splitter.split(val)
                for part in parts:
                    if part:
                        hashName = f'{feature}={part}'
                        d[hashName] = 1
                continue
            if feature == 'subid':
                parts = subid_splitter.split(val)
                for part in parts:
                    if part:
                        hashName = f'{feature}={part}'
                        d[hashName] = 1
                continue                
            hashName = f'{feature}={val}'
            d[hashName] = 1
        yield d
        


# In[20]:


hasher = FeatureHasher(non_negative=True, alternate_sign=False)


# In[21]:


exoc_train = exoctet(train)


# In[22]:


exoc_dev = exoctet(dev)


# In[23]:


exoc_test = exoctet(test)


# In[24]:


y_train = train['target'].values.astype(np.bool)


# In[25]:


y_dev = dev['target'].values.astype(np.bool)


# In[26]:


y_test = test['target'].values.astype(np.bool)


# In[27]:


test = test.drop('target', axis=1)


# In[28]:


train = train.drop('target', axis=1)


# In[29]:


dev = dev.drop('target', axis=1)


# In[30]:


hashed_train = hasher.transform(get_features(train, features))


# In[31]:


hashed_dev = hasher.transform(get_features(dev, features))


# In[32]:


hashed_test = hasher.transform(get_features(test, features))


# In[33]:


stacked_train = hstack((hashed_train, exoc_train))


# In[34]:


stacked_dev = hstack((hashed_dev, exoc_dev))


# In[35]:


stacked_test = hstack((hashed_test, exoc_test))


# In[36]:


dtrain = xgb.DMatrix(stacked_train, y_train, nthread=-1)


# In[37]:


ddev = xgb.DMatrix(stacked_dev, y_dev, nthread=-1)


# In[38]:


dtest = xgb.DMatrix(stacked_test, y_test, nthread=-1)


# In[39]:


def plot_learning_curve(eval_results):
    plt.plot(eval_results['train']['auc'])
    plt.plot(eval_results['dev']['auc'])
    plt.title('model learning curve')
    plt.ylabel('ROC AUC')
    plt.xlabel('boosting rounds')
    plt.legend(['train', 'dev'], loc='lower right')
    plt.show()


# In[40]:


from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


# In[41]:


from math import log


# In[42]:


from hyperopt.pyll.stochastic import sample


# In[43]:


space = {
    'learning_rate': hp.loguniform('learning_rate', log(0.01), log(0.3)),
    'max_depth': hp.quniform('max_depth', 2, 20, 1),
    'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
    'subsample': hp.uniform('subsample', 0.5, 1.0),
    'colsample': hp.choice('colsample', [
        ('colsample_bytree', hp.uniform('colsample_bytree', 0.5, 1.0)),
        ('colsample_bylevel', hp.uniform('colsample_bylevel', 0.5, 1.0))
    ]),
    'gamma': hp.loguniform('gamma', log(1e-8), log(10)),
    'reg_alpha': hp.loguniform('reg_alpha', log(1), log(100)),
    'reg_lambda': hp.uniform('reg_lambda', 0.1, 10),
    'scale_pos_weight': hp.uniform('scale_pos_weight', 1, 100),
    'silent': 1,
    'eval_metric': 'auc',
    'objective': 'binary:logistic'
}


# In[44]:


dtrain_limited = dtrain.slice(train[train.time >= datetime.datetime(2018, 2, 1)].index)


# In[45]:


def train_eval(params):
    colsample_type, colsample_val = params.pop('colsample')
    params[colsample_type] = colsample_val
    params['max_depth'] = int(params['max_depth'])
    evals_result = {}
    booster = xgb.train(params, dtrain_limited, num_boost_round=5000, early_stopping_rounds=20, verbose_eval=False, evals=[(dtrain_limited, 'train'),(ddev, 'dev')], evals_result=evals_result)
    loss = -evals_result['dev']['auc'][booster.best_iteration]
    return {'loss': loss, 'trees': booster.best_ntree_limit, 'status': STATUS_OK}


# In[124]:


get_ipython().system('pip install -U git+https://github.com/hyperopt/hyperopt.git')


# In[48]:


trials = Trials()
best = fmin(train_eval,
    space=space,
    algo=tpe.suggest,
    max_evals=300, trials=trials, verbose=1)


# In[49]:


trials.best_trial


# In[79]:


best


# In[50]:


0.759879 - 0.74543


# In[52]:


((0.759879 - 0.74543) / 0.74543)*100


# In[80]:


params = {
 'max_depth': 14,
 'objective': 'binary:logistic',
 'eval_metric': ['logloss', 'auc'],
 'eta': 0.2613422990687285,
 'min_child_weight': 2,
 'reg_alpha': 1.547978704593122,
 'reg_lambda': 1.6862550425820109,
 'scale_pos_weight': 3.5085580437807575,
 'subsample': 0.8603983319000923,
 'colsample_bylevel': 0.5745968600980427,
 'gamma': 1.2181926608051464e-07,
 'silent': 1,
}

evals_result = {}
booster = xgb.train(params, dtrain_limited, num_boost_round=2000, early_stopping_rounds=20, verbose_eval=True, evals=[(dtrain_limited, 'train'),(ddev, 'dev')], evals_result=evals_result)
plot_learning_curve(evals_result)


# In[81]:


params = {
 'max_depth': 14,
 'objective': 'binary:logistic',
 'eval_metric': ['logloss', 'auc'],
 'eta': 0.2613422990687285,
 'min_child_weight': 2,
 'reg_alpha': 1.547978704593122,
 'reg_lambda': 1.6862550425820109,
 'scale_pos_weight': 3.5085580437807575,
 'subsample': 0.8603983319000923,
 'colsample_bylevel': 0.5745968600980427,
 'gamma': 1.2181926608051464e-07,
 'silent': 1,
}

evals_result = {}
booster = xgb.train(params, dtrain.slice(train[train.time >= datetime.datetime(2018, 1, 1)].index), num_boost_round=2000, early_stopping_rounds=20, verbose_eval=True, evals=[(dtrain.slice(train[train.time >= datetime.datetime(2018, 1, 1)].index), 'train'),(ddev, 'dev')], evals_result=evals_result)
plot_learning_curve(evals_result)


# In[84]:


params = {
 'max_depth': 14,
 'objective': 'binary:logistic',
 'eval_metric': ['logloss', 'auc'],
 'eta': 0.2613422990687285,
 'min_child_weight': 2,
 'reg_alpha': 1.547978704593122,
 'reg_lambda': 1.6862550425820109,
 'scale_pos_weight': 3.5085580437807575,
 'subsample': 0.8603983319000923,
 'colsample_bylevel': 0.5745968600980427,
 'gamma': 1.2181926608051464e-07,
 'silent': 1,
}

evals_result = {}
booster = xgb.train(params, dtrain.slice(train[train.time >= datetime.datetime(2017, 11, 1)].index), num_boost_round=2000, early_stopping_rounds=20, verbose_eval=True, evals=[(dtrain.slice(train[train.time >= datetime.datetime(2017, 11, 1)].index), 'train'),(ddev, 'dev')], evals_result=evals_result)
plot_learning_curve(evals_result)


# In[87]:


params = {
 'max_depth': 14,
 'objective': 'binary:logistic',
 'eval_metric': ['logloss', 'auc'],
 'eta': 0.2613422990687285,
 'min_child_weight': 2,
 'reg_alpha': 1.547978704593122,
 'reg_lambda': 1.6862550425820109,
 'scale_pos_weight': 3.5085580437807575,
 'subsample': 0.8603983319000923,
 'colsample_bylevel': 0.5745968600980427,
 'gamma': 1.2181926608051464e-07,
 'silent': 1,
}

evals_result = {}
booster = xgb.train(params, dtrain_limited, num_boost_round=239, early_stopping_rounds=20, verbose_eval=True, evals=[(dtrain_limited, 'train'),(ddev, 'dev')], evals_result=evals_result)
plot_learning_curve(evals_result)


# In[46]:


params = {
 'max_depth': 14,
 'objective': 'binary:logistic',
 'eval_metric': ['logloss', 'auc'],
 'eta': 0.2613422990687285,
 'min_child_weight': 2,
 'reg_alpha': 1.547978704593122,
 'reg_lambda': 1.6862550425820109,
 'scale_pos_weight': 3.5085580437807575,
 'subsample': 0.8603983319000923,
 'colsample_bylevel': 0.5745968600980427,
 'gamma': 1.2181926608051464e-07,
 'silent': 1,
}

evals_result = {}
booster = xgb.train(params, dtrain_limited, num_boost_round=239, early_stopping_rounds=20, verbose_eval=True, evals=[(dtrain_limited, 'train'),(ddev, 'dev')], evals_result=evals_result)
plot_learning_curve(evals_result)


# In[47]:


xgb.plot_importance(booster, max_num_features=20)


# In[89]:


preds = booster.predict(dtest)


# In[90]:


roc_auc_score(y_test, preds)


# In[ ]:


params = {
 'base_score': y_train.mean(),
 'max_depth': 3,
 'objective': 'binary:logistic',
 'eval_metric': ['logloss', 'auc'],
 'eta': .1,
 'silent': 1,
 'reg_lambda': 2
}

evals_result = {}
booster = xgb.train(params, dtrain, num_boost_round=2000, early_stopping_rounds=20, verbose_eval=True, evals=[(dtrain, 'train'),(ddev, 'dev')], evals_result=evals_result)
plot_learning_curve(evals_result)


# In[67]:


preds = booster.predict(dtest, ntree_limit=booster.best_ntree_limit)
roc_auc_score(dtest.get_label(), preds)


# In[68]:


recent_test = pq.ParquetDataset('marvel-recent/').read().to_pandas()

exoc_recent = exoctet(recent_test)

y_recent = recent_test.pop('target').astype(np.bool)

hashed_recent = hasher.transform(get_features(recent_test, features))

drecent = xgb.DMatrix(hstack((hashed_recent, exoc_recent)), y_recent)

preds_recent = booster.predict(drecent)
roc_auc_score(drecent.get_label(), preds_recent)


# In[69]:


aucs = []
days = []
for day in np.sort(recent_test.day.unique()):
    days.append(day)
    indices = recent_test[recent_test.day == day].index
    sliced = drecent.slice(indices)
    day_preds = booster.predict(sliced, ntree_limit=booster.best_ntree_limit)
    aucs.append(roc_auc_score(sliced.get_label(), day_preds))
    
plt.plot(days, aucs)
plt.title('ROC AUC by day May')
plt.ylabel('ROC AUC')
plt.xlabel('Day')
plt.xticks(days)
plt.show()


# In[70]:


plt.plot(days, aucs)
plt.title('ROC AUC by day May')
plt.ylabel('ROC AUC')
plt.xlabel('Day')
plt.xticks(days)
plt.show()


# In[74]:


recent_test['conversion'] = y_recent


# In[ ]:


conversions = []
days = []
for day in np.sort(recent_test.day.unique()):
    days.append(day)
    day_data = recent_test[recent_test.day == day]    
    conversions.append()


# In[ ]:


plt.plot(days, aucs)
plt.title('Conversions by day May')
plt.ylabel('Conversion count')
plt.xlabel('Day')
plt.xticks(days)
plt.show()


# In[79]:


recent_test = pq.ParquetDataset('marvel-recent/').read().to_pandas()

exoc_recent = exoctet(recent_test)

y_recent = recent_test.pop('target').astype(np.bool)

hashed_recent = hasher.transform(get_features(recent_test, features))

drecent = xgb.DMatrix(hstack((hashed_recent, exoc_recent)), y_recent)

preds_recent = booster.predict(drecent)
roc_auc_score(drecent.get_label(), preds_recent)


# In[80]:


aucs = []
days = []
for day in np.sort(recent_test.day.unique()):
    days.append(day)
    indices = recent_test[recent_test.day == day].index
    sliced = drecent.slice(indices)
    day_preds = booster.predict(sliced, ntree_limit=booster.best_ntree_limit)
    aucs.append(roc_auc_score(sliced.get_label(), day_preds))
    
plt.plot(days, aucs)
plt.title('ROC AUC by day May')
plt.ylabel('ROC AUC')
plt.xlabel('Day')
plt.xticks(days)
plt.show()


# In[88]:


recent_test['predictions'] = preds_recent


# In[89]:


recent = recent_test


# In[90]:


recent['predictions'].describe(percentiles=[.25, .85])


# In[91]:


recent['conversion'] = y_recent


# In[92]:


recent['impression'] = 1


# In[93]:


percentiles = np.percentile(recent.predictions, np.arange(5, 100, 5), interpolation='nearest')


# In[ ]:


# impressions, conversions, revenue from .05 to .95


# In[94]:


list(zip(np.arange(5, 100, 5), percentiles))


# In[95]:


low = percentiles[0]
datas = [(recent[recent.predictions < low], f'< {low}')]
for percentile in percentiles[1:-1]:
    data = recent[(recent.predictions >= low) & (recent.predictions < percentile)]
    datas.append((data, f'>= {low} and < {percentile}'))
    low = percentile
    
datas.append((recent[recent.predictions >= percentiles[-1]], f'>= {percentile}'))


# In[96]:


len(datas)


# In[97]:


len(percentiles)


# In[98]:


groups = []
for data, cutoff in datas:
    df_data = dict(data.agg({'impression': 'count', 'conversion': 'sum', 'bid_amount': 'sum'}))
    df_data['cutoff'] = cutoff
    groups.append(df_data)


# In[99]:


cutoff_data = pd.DataFrame(groups)


# In[100]:


cutoff_data


# In[114]:


save_xjs_model(booster, 'classifier', 'conversion-boost-2018-05-24.xjs', base_score=y_train.mean())


# In[115]:


booster.save_model('conversion-boost-2018-05-24.bin')


# # Checking internal sources

# In[ ]:


recent_test


# In[102]:


source_conv_rate = recent_test.groupby('source_id')['conversion'].sum() / recent_test.groupby('source_id')['conversion'].count()


# In[103]:


source_conv_rate.sort_values(ascending=False)


# In[104]:


source_conv_rate[source_conv_rate.index == '1']


# In[105]:


source_conv_rate[source_conv_rate.index == '8']


# In[123]:


code_preform = recent_test[recent_test.subid == '2606']


# In[124]:


code_preform['conversion'].sum() / code_preform['conversion'].count()


# In[108]:


train['conversion'] = dtrain.get_label().astype(np.bool)


# In[109]:


source_conv_rate_train = train.groupby('source_id')['conversion'].sum() / train.groupby('source_id')['conversion'].count()


# In[110]:


source_conv_rate_train.sort_values(ascending=False)


# In[111]:


code_preform_train = train[train.subid == '2606']


# In[112]:


code_preform_train['conversion'].sum() / code_preform_train['conversion'].count()


# In[113]:


source_conv_rate_train.sort_values(ascending=False)


# In[116]:


len(code_preform[code_preform.predictions < 0.1305853873]) / len(code_preform)


# In[117]:


len(code_preform[code_preform.predictions >= 0.31]) / len(code_preform)


# In[ ]:


bundlore pops - 2267 , 2484


# In[119]:


bundlore = recent_test[(recent_test.subid == '2267') | (recent_test.subid == '2484')]


# In[120]:


len(bundlore)


# In[121]:


len(bundlore[bundlore.predictions < 0.1305853873]) / len(bundlore)


# In[122]:


len(bundlore[bundlore.predictions >= 0.31]) / len(bundlore)

