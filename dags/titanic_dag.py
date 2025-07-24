# import sys 
# import seaborn as sns
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 6, 3),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'tiianic_pipeline',
    default_args=default_args,
    description='A titanic DAG for practice',
    schedule=timedelta(days=1),
)


def data_collection(**kwargs):
    data = pd.read_csv('/opt/airflow/dags/titanic.csv')
    return data


def data_preprocessing(**kwargs):
    data = pd.read_csv('/opt/airflow/dags/titanic.csv')
    data = data.drop(columns=['deck', 'alive'])
    data = data.dropna()
    cat_cols = data.select_dtypes(include=['object', 'category']).columns

    le = LabelEncoder()
    for col in cat_cols:
        data[col] = data[col].astype(str)
        data[col] = le.fit_transform(data[col])

    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(columns='survived'), data.survived, test_size=0.2, random_state=42
    )
    # return X_train, X_test, y_train, y_test

data = PythonOperator(
    task_id='data_collection',
    python_callable=data_collection,
    # op_kwargs={'word': word},
    dag=dag,
)

preprocessed_data = PythonOperator(
    task_id='data_preprocessing',
    python_callable=data_preprocessing,
    # op_kwargs={'data': data},
    dag=dag,
)

data >> preprocessed_data
