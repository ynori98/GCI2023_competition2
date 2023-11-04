#ライブラリのインポート
import os as os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import optuna
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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

#特徴量作成
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

# 訓練データと評価データに分割
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

def objective(trial):

    #評価するハイパーパラメータの値を指定
    params ={
        'max_depth':trial.suggest_int('max_depth',3,8),
        'gamma':trial.suggest_uniform('gamma',0,1),
        'subsample':trial.suggest_uniform('subsample',0,1),
        'colsample_bytree':trial.suggest_uniform('colsample_bytree',0,1),
        'reg_alpha':trial.suggest_loguniform('reg_alpha',1e-5,100),
        'reg_lambda':trial.suggest_loguniform('reg_lambda',1e-5,100),
        'learning_rate':trial.suggest_uniform('learning_rate',0,1)}

    model = XGBClassifier(n_estimators=10000,
                            min_child_weight=1,
                            verbosity=0,
                            n_jobs=-1,
                            random_state=0,
                            **params)

    model.fit(X_train, y_train,eval_set=[(X_valid,y_valid)],early_stopping_rounds=100,verbose=False)
    xgb_valid_pred = model.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, xgb_valid_pred)

    return auc

#optuna.create_study()でoptuna.studyインスタンスを作る。
study = optuna.create_study(direction="maximize")

#studyインスタンスのoptimize()に作った関数を渡して最適化する。
study.optimize(objective, n_trials=1000)

#スコアを見る
print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)
print('Best score:', study.best_value)

#Best trial: {'max_depth': 6, 'gamma': 0.7682327129628294, 'subsample': 0.8993802982181226, 'colsample_bytree': 0.2115912065407456, 'reg_alpha': 0.024916662832175553, 'reg_lambda': 2.8159567647636456, 'learning_rate': 0.029633895526920813}
#Best score: 0.7632018658782871