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
# exclusive for this data only
def preprocess(df):
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

# linear regression class
class LR:
    def __init__(self, X, y) -> None:
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        self.lr = LogisticRegression()
        self.lr.fit(self.X_train, self.y_train)

    def predict(self, X):
        return self.lr.predict(X)

# load train and test data
train_data = load_data("data/train.csv")
test_data = load_data("data/test.csv")

# preprocess test data, should be turned into a function
test_data_used = test_data.loc[:,['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]

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

X_train, y_train = preprocess(train_data)
lr = LR(X_train, y_train)

y_pred = lr.predict(test_data_transformed)

print(f"type of test_data: {type(test_data)}")
print(f"type of test_data_used: {type(test_data_used)}")
print(f"type of test_data_transformed: {type(test_data_transformed)}")
print(f"type of y_pred: {type(y_pred)}")
print(f"shape of y_pred: {y_pred.shape}")

test_data["Survived"] = y_pred

result = pd.DataFrame({"PassengerId": test_data['PassengerId'], "Survived": y_pred})

# right result to a .csv file
result.to_csv("lr_result.csv", index=False)