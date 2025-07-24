import seaborn as sns
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta


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

def data_collection():
    data = sns.load_dataset('titanic')
    return data

data = PythonOperator(
    task_id=f'data',
    python_callable=data_collection,
    # op_kwargs={'word': word},
    dag=dag,
)
