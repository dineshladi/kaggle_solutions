import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import joblib

if __name__ == '__main__':
    df = pd.read_csv("data/train.csv")
    df['fold'] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    print(df.shape)
    print(df.columns)
    skf = KFold(n_splits=10, random_state=123, shuffle=False)
    for fold, (train_index, test_index) in enumerate(skf.split(X=df, y=df.target.values)):
        df.loc[test_index, 'fold'] = fold
    
    k = 0
    #for k in range(5):
    train = df[df.fold != k]
    test = df[df.fold == k]
    
    y_train = train.target.values
    y_test = test.target.values
    
    train = train.drop(["id", "target", "fold"], axis=1)
    test = test.drop(["id", "target", "fold"], axis=1)
    test = test[train.columns]

    encoders = {}
    for col in train.columns:
        #print(col)
        le = LabelEncoder()
        le.fit(train[col].values.tolist() + test[col].values.tolist())
        train.loc[:, col] = le.transform(train[col].values.tolist())
        test.loc[:, col] = le.transform(test[col].values.tolist())
        encoders[col] = le

    dtrain = xgb.DMatrix(train, label=y_train)
    dtest = xgb.DMatrix(test, label=y_test)
    
    params = {'max_depth': 3, 
              'eta': 0.1, 
              'objective': 'binary:logistic',
              'nthread': -1,
              'eval_metric': 'auc',
              'verbosity': 2 }
    model = xgb.train(params, dtrain, num_boost_round=200)
    preds = model.predict(dtest)
    print(roc_auc_score(y_test, preds))
    joblib.dump(model, "models/xgboost.pkl")
    joblib.dump(encoders, "models/encoders.pkl")


    