import pandas as pd
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib
import numpy as np

if __name__ == '__main__':
    test = pd.read_csv("data/test.csv")

    ids = test.id
    test = test.drop("id", axis=1)

    model = joblib.load("models/xgboost.pkl")
    encoders = joblib.load("models/encoders.pkl")
    for col in test.columns:
        le = encoders.get(col)
        le_dict = dict(zip(le.classes_, le.transform(le.classes_)))
        test.loc[:, col] = test[col].apply(lambda x: le_dict.get(x, -1))

    dtest = xgb.DMatrix(test)
    preds = model.predict(dtest)

    to_submit = pd.DataFrame(np.column_stack((ids, preds)), columns=["id", "target"])
    to_submit['id'] = to_submit.id.astype('int')
    print(to_submit.head())
    to_submit.to_csv("data/submission.csv", index=False)
    