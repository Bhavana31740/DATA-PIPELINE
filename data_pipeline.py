import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

data = {
    "Name": ["John", "Jane", "Doe", "Smith"],
    "Age": [28, 34, np.nan, 45],
    "Gender": ['Female', 'Male', 'Female', 'Male'],
    "Salary": [50000, 60000, 70000, np.nan]
}
df = pd.DataFrame(data)
print(df)

numeric_features = ['Age', 'Salary']
numeric_imputer = SimpleImputer(strategy='mean')
df[numeric_features] = numeric_imputer.fit_transform(df[numeric_features])

categorical_features = ['Name', 'Gender']
categorical_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_features] = categorical_imputer.fit_transform(df[categorical_features])

print(df)
df_features = df.drop(columns=['Name'])

categorical_features = ['Gender']

numeric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop=None))  # keep all gender columns
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_pipeline, numeric_features),
    ('cat', categorical_pipeline, categorical_features)
])

processed_data = preprocessor.fit_transform(df_features)

processed_df = pd.DataFrame(
    processed_data,
    columns=['Age', 'Salary', 'Gender_Female', 'Gender_Male']
)

print(processed_df)
