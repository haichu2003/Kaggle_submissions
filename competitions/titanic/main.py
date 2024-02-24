import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, classification_report, ConfusionMatrixDisplay

# load data from file
def load_data(file):
    df = pd.read_csv(file)
    return df

# preprocess data
# exclusive for these data only
def preprocess_train_data(df):
    data = df.drop(['PassengerId',
                'Name',
                'Ticket',
                'Cabin',
                'SibSp',
                'Parch'], axis=1)
    data['Age'] = data['Age'].fillna(data["Age"].mean())
    data['Embarked'] = data['Embarked'].fillna('C')
    X = data.drop('Survived', axis=1)
    y = data['Survived']

    numeric_features = ['Age', 'Fare']

    numeric_transformer = Pipeline(
        steps=[('scaler', StandardScaler())]
    )

    categorical_features = ['Pclass', 'Sex', 'Embarked']
    categorical_transformer = Pipeline(
        steps=[("encoder", OneHotEncoder(handle_unknown='ignore'))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    X_transformed = preprocessor.fit_transform(X)
    return X_transformed, y

def preprocess_test_data(df):
    test_data_used = df.loc[:,['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]

    # fill NaN values in test data
    test_data_used['Age'] = test_data_used['Age'].fillna(test_data_used['Age'].mean())
    test_data_used['Fare'] = test_data_used['Fare'].fillna(test_data_used['Fare'].mean())

    # scale numeric features
    numeric_features = ['Age', 'Fare']
    numeric_transformer = Pipeline(
        steps=[('scaler', StandardScaler())]
    )

    # one hot encode categorical features
    categorical_features = ['Pclass', 'Sex', 'Embarked']
    categorical_transformer = Pipeline(
        steps=[("encoder", OneHotEncoder(handle_unknown='ignore'))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # transform test data
    test_data_transformed = preprocessor.fit_transform(test_data_used)
    return test_data_transformed

# linear regression class
class LR:
    def __init__(self, train_data) -> None:
        X, y = preprocess_train_data(train_data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        self.lr = LogisticRegression()
        self.lr.fit(X_train, y_train)

    def predict(self, X):
        return self.lr.predict(X)

    def evalualte(self, test_data, out_file):
        test_data_transformed = preprocess_test_data(test_data)
        y_pred = self.lr.predict(test_data_transformed)
        print(f"shape of y_pred: {y_pred.shape}")
        test_data["Survived"] = y_pred
        result = pd.DataFrame({"PassengerId": test_data['PassengerId'], "Survived": y_pred})

        # right result to a .csv file
        result.to_csv(out_file, index=False)

# random forest classifier class
class RFC:
    def __init__(self, train_data) -> None:
        X, y = preprocess_train_data(train_data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # find best n_estimators
        n = range(1,100)
        results = np.array([])
        for i in n:
            rfc_temp = RandomForestClassifier(n_estimators=i)
            rfc_temp.fit(X_train, y_train)
            results = np.append(results, rfc_temp.score(X_test, y_test))

        best_estimator = results.argmax()
        self.rfc = RandomForestClassifier(n_estimators=best_estimator)
        self.rfc.fit(X_train, y_train)

    def predict(self, X):
        return self.rfc.predict(X)

    def evaluate(self, test_data, out_file):
        test_data_transformed = preprocess_test_data(test_data)
        y_pred = self.rfc.predict(test_data_transformed)
        print(f"shape of y_pred: {y_pred.shape}")
        test_data["Survived"] = y_pred
        result = pd.DataFrame({"PassengerId": test_data['PassengerId'], "Survived": y_pred})

        # right result to a .csv file
        result.to_csv(out_file, index=False)

# load train and test data
train_data = load_data("data/train.csv")
test_data = load_data("data/test.csv")

# lr = LR(train_data)
# lr.evalualte(test_data, "lr_result.csv")

rfc = RFC(train_data)
rfc.evaluate(test_data, "rfc_result.csv")
