import unittest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class TestDataCollection(unittest.TestCase):
    def setUp(self):
        self.data = pd.read_csv('/Users/minseokoh/Desktop/00. SCU/04. Summer Break/git-practice/dags/titanic.csv')

    def test_data_type(self):
        self.assertIsInstance(self.data, pd.DataFrame)
    
    def test_column_length(self):
        self.assertEqual(self.data.shape[1], 15)
    
    def test_row_length(self):
        self.assertEqual(len(self.data), 891)


class TestDataPreprocessing(unittest.TestCase):
    def setUp(self):
        self.data = pd.read_csv('/Users/minseokoh/Desktop/00. SCU/04. Summer Break/git-practice/dags/titanic.csv')
        self.label_encoder = LabelEncoder()

    def test_num_of_columns_after_drop(self):
        self.data = self.data.drop(columns=['deck', 'alive'])
        num_of_columns = self.data.shape[1]
        self.assertEqual(num_of_columns, 13)

    def test_missing_values_in_df(self):
        self.data = self.data.dropna()
        num_of_missing_values = self.data.isna().sum().sum()
        self.assertEqual(num_of_missing_values, 0)

    def test_num_of_category_columns(self):
        categorical_columns = self.data.select_dtypes(include=['object', 'category']).columns
        num_of_categorical_columns = len(categorical_columns)
        self.assertEqual(num_of_categorical_columns, 7)

    def test_num_of_categorically_converted_columns(self):
        categorical_columns = self.data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_columns:
            self.data[col] = self.data[col].astype(str)
            self.data[col] = self.label_encoder.fit_transform(self.data[col])

        num_of_categorically_converted_columns = len(self.data[categorical_columns].dtypes == np.int64)
        self.assertEqual(num_of_categorically_converted_columns, 7)

    def test_num_of_train_test_data(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.data.drop(columns='survived'), self.data.survived, test_size=0.2, random_state=42
        )
        num_of_train_data = len(X_train)
        self.assertEqual(num_of_train_data, 712)
        num_of_test_data = len(X_test)
        self.assertEqual(num_of_test_data, 179)


class TestModelTraining(unittest.TestCase):
    def setUp(self):
        self.n_estimator = 100
        self.max_depth = 5
        self.random_state = 42

        data = pd.read_csv('/Users/minseokoh/Desktop/00. SCU/04. Summer Break/git-practice/dags/titanic.csv')
        data = data.drop(columns=['deck', 'alive'])
        data = data.dropna()
        cat_cols = data.select_dtypes(include=['object', 'category']).columns

        le = LabelEncoder()
        for col in cat_cols:
            data[col] = data[col].astype(str)
            data[col] = le.fit_transform(data[col])

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        data.drop(columns='survived'), data.survived, test_size=0.2, random_state=42
        )
        self.random_forest_classifier = RandomForestClassifier(
            n_estimators=self.n_estimator, max_depth=self.max_depth, random_state=self.random_state
        )

    def test_ypred_is_null_or_inf(self):
        self.random_forest_classifier.fit(self.X_train, self.y_train)
        self.y_pred = self.random_forest_classifier.predict(self.X_test)
        # y_pred에서 null이 있는지 확인 --> null 없다면 OK 
        self.assertFalse(np.isnan(self.y_pred).any())
        self.assertFalse(np.isinf(self.y_pred).any())


    def test_accuracy_precision_recall_fscore_are_null_or_inf(self):
        self.random_forest_classifier.fit(self.X_train, self.y_train)
        y_pred = self.random_forest_classifier.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision_recall_fscore = precision_recall_fscore_support(self.y_test, y_pred, average='binary')
        self.assertEqual(round(accuracy, 2), 0.79)
        self.assertEqual(round(precision_recall_fscore[2], 2), 0.72)