import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report

project_root = os.getcwd()
raw_data_file = os.path.join(project_root, "adult_data.csv")
data = pd.read_csv(raw_data_file)

data['occupation'] = data['occupation'].str.strip().str.lower() # Corrected text normalization
data['native-country'] = data['native-country'].replace({'United-States': 'United-States', 'Canada': 'North America', 'Mexico': 'North America'}, regex=True) # Corrected spatial aggregation

X_train, X_test, y_train, y_test = train_test_split(data.drop('salary', axis=1), data['salary'], test_size=0.2, random_state=42)

numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

pipeline.fit(X_train, y_train)

score = pipeline.score(X_test, y_test)
print(f"Model accuracy: {score:.2f}")
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))