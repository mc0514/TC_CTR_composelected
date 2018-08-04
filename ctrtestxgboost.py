# coding=utf-8
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
#from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import StratifiedKFold


#train_dataX_p='E:\\ctrtestxgboost\\data\\trainALL_x.csv'
#train_dataY_p='E:\\ctrtestxgboost\\data\\trainALL_y.csv'
#testAll_p='E:\\ctrtestxgboost\\data\\test_ALL.csv'

#train_x_all = pd.read_csv(train_dataX_p)
#train_y_all = pd.read_csv(train_dataY_p)
#test_A = pd.read_csv(testAll_p)
train_x_p='E:\\ctrtestxgboost\\data1\\train_X.csv'
train_y_p='E:\\ctrtestxgboost\\data1\\train_Y.csv'
test_x_p='E:\\ctrtestxgboost\\data1\\test_X.csv'
test_y_p='E:\\ctrtestxgboost\\data1\\test_Y.csv'
test_A_p='E:\\ctrtestxgboost\\data1\\test_B_final.csv'
train_x=pd.read_csv(train_x_p)
train_y=pd.read_csv(train_y_p)
test_x=pd.read_csv(test_x_p)
test_y=pd.read_csv(test_y_p)
test_a=pd.read_csv(test_A_p)
print('step1: read data finished!')

min_max_scaler = preprocessing.MinMaxScaler()
train_x_scaler=min_max_scaler.fit_transform(train_x)
test_x_scaler=min_max_scaler.fit_transform(test_x)
test_a_scaler=min_max_scaler.fit_transform(test_a)
print('data standardization ',train_x_scaler.shape, test_x_scaler.shape, test_a_scaler.shape)

#设置方差的阈值为0.8
sel = VarianceThreshold(threshold=.08)
#选择方差大于0.8的特征
train_x_scaler_sel=sel.fit_transform(train_x_scaler)
#test_x_scaler_sel=sel.fit_transform(test_x_scaler)
#test_a_scaler_sel=sel.fit_transform(test_a_scaler)
print('data threshold ',train_x_scaler_sel.shape)
#print('threshold colums ',train_x_scaler_sel.columns.values)



#train_x,test_x,train_y,test_y=train_test_split(train_x_all,train_y_all,test_size=0.3, random_state=0)
#print('step2: split data finished!')

xgb_model= XGBClassifier(
n_estimattors=100,learning_rate=0.2,max_depth=8,subsample=0.7
)
eval_set = [(test_x_scaler, test_y)]
xgb_model.fit(train_x_scaler, train_y, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=True)
# parameters = {'nthread':[4],
#               'objective':['binary:logistic'],
#               'learning_rate': [0.05],
#               'max_depth': [6],
#               'min_child_weight': [11],
#               'silent': [1],
#               'subsample': [0.8],
#               'colsample_bytree': [0.7],
#               'n_estimators': [5],
#               'missing':[-999],
#               'seed': [1337]}
# clf = GridSearchCV(xgb_model, parameters, n_jobs=1,
#                    cv=StratifiedKFold( n_splits=5, shuffle=True, random_state=0),
#                    scoring='neg_log_loss',
#                    verbose=2, refit=True)
#xgb_model.fit(train_x_all, train_y_all)
print('step3: model fit finished!')

column_headers = list(train_x.columns.values)
features_importance=dict(zip(column_headers,xgb_model.feature_importances_))
#print(column_headers)
#print(xgb_model.feature_importances_)
print(features_importance)


test_a['isgoumai'] = xgb_model.predict_proba(test_a_scaler)[:, 1]
test_a['isgoumai'] = test_a['isgoumai'].apply(lambda x: float('%.6f' % x))
test_a.to_csv('./submission0802.csv', index=False)
print('step4: write the result finished')