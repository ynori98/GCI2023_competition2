#ライブラリのインポート
import os as os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

# データの読み込み
cwd = os.getcwd()

train_df = pd.read_csv(cwd + "\\input\\train.csv")
test_df= pd.read_csv(cwd + "\\input\\test.csv")
sample_sub = pd.read_csv(cwd + "\\input\\sample_submission.csv")
df_all = pd.concat([train_df, test_df], axis=0)

#one_hot_encoding
def one_hot_encoding(df):

    return_df = pd.get_dummies(df, drop_first=True)

    return return_df

#new_features
def to_add_feature(df):
    #new_features_based_on_ext_sources
    df['EXT_123_MEAN'] = (df['EXT_SOURCE_1'] + df['EXT_SOURCE_2'] + df['EXT_SOURCE_3']) / 3
    df['EXT_23_MEAN'] = (df['EXT_SOURCE_2'] + df['EXT_SOURCE_3']) / 2
    df['EXT_12_MEAN'] = (df['EXT_SOURCE_1'] + df['EXT_SOURCE_2']) / 2
    df['EXT_13_MEAN'] = (df['EXT_SOURCE_1'] + df['EXT_SOURCE_3']) / 2
    df['EXT_23_DIFFERENCE'] = abs(df['EXT_SOURCE_2'] - df['EXT_SOURCE_3'])
    df['EXT_12_DIFFERENCE'] = abs(df['EXT_SOURCE_1'] - df['EXT_SOURCE_2'])
    df['EXT_13_DIFFERENCE'] = abs(df['EXT_SOURCE_1'] - df['EXT_SOURCE_3'])
    
    #new_features_based_on_credit
    df['ANNUITY_TO_CREDIT_RATIO'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    df['CREDIT_TO_GOODS_PRICE_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    df['INCOME_TO_CREDIT_RATIO'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    
    #new_features_based_on_income
    df['ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    
    #new_features_based_on_time
    df['BOUGHT_CAR_AGE'] = (df['DAYS_BIRTH'] / 365) - df['OWN_CAR_AGE']
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

    return df

#カラム削除
def to_drop(df):

    drop_list = ['FLAG_MOBIL', 'FLAG_CONT_MOBILE', 'REG_REGION_NOT_LIVE_REGION', 'LIVE_REGION_NOT_WORK_REGION']
    droped_df = df.drop(columns=drop_list)

    return droped_df

df_encoded = one_hot_encoding(df_all)
added_features_df = to_add_feature(df_encoded)
all_features_df = to_drop(added_features_df)

#データ分割
train = all_features_df[all_features_df.loc[:, 'SK_ID_CURR'] < 171202]
test = all_features_df[all_features_df.loc[:, 'SK_ID_CURR'] > 171201]

train_x = train.drop(columns=['TARGET', 'SK_ID_CURR'])
train_y = train['TARGET']
test_x = test.drop(columns=['TARGET', 'SK_ID_CURR'])

X = train_x.values
y = train_y.values

#stratifiedKFold
fold = StratifiedKFold(n_splits=8, shuffle=True, random_state=69)
cv = list(fold.split(X, y))

#optunaで探索したハイパラ
xgb_best_param = {'max_depth': 6, 'gamma': 0.7682327129628294, 'subsample': 0.8993802982181226, 'colsample_bytree': 0.2115912065407456, 'reg_alpha': 0.024916662832175553, 'reg_lambda': 2.8159567647636456, 'learning_rate': 0.029633895526920813}
#Best score: 0.7632018658782871

#CVのための関数
def fit_xgb(X, y, cv, params: dict=None, verbose=100):

    oof_preds = np.zeros(X.shape[0])

    if params is None:
        params = {}

    models = []

    for i, (idx_train, idx_valid) in enumerate(cv):
        x_train, y_train = X[idx_train], y[idx_train]
        x_valid, y_valid = X[idx_valid], y[idx_valid]

        clf = XGBClassifier(n_estimators=1000,
                            verbosity=0,
                            n_jobs=-1,
                            random_state=0,
                            **params)
        clf.fit(x_train, y_train,
                eval_set=[(x_valid, y_valid)],
                early_stopping_rounds=100,
                eval_metric='auc',
                verbose=verbose)

        models.append(clf)
        oof_preds[idx_valid] = clf.predict_proba(x_valid)[:, 1]
        print('Fold %2d AUC : %.6f' % (i + 1, roc_auc_score(y_valid, oof_preds[idx_valid])))

    score = roc_auc_score(y, oof_preds)
    print('Full AUC score %.6f' % score)
    return oof_preds, models

oof, models = fit_xgb(X, y, cv=cv, params=xgb_best_param)

pred = np.array([model.predict_proba(test_x.values)[:, 1] for model in models])
pred = np.mean(pred, axis=0)

submission = sample_sub.copy()
submission['TARGET'] = pred

submission.to_csv('xgb_optimized2.csv', index=False)