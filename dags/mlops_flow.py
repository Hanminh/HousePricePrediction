import airflow
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    'start_date': datetime(2025, 1, 1),
}

with DAG(
    dag_id= 'machine_learning_pipeline',
    default_args=default_args,
    schedule= None,
    catchup= False,
    tags=['mlops'],
) as dag:
    preprocess = BashOperator(
        task_id='preprocess_data',
        bash_command= 'python /opt/airflow/src/data/run_processing.py   --input /opt/airflow/data/raw/house_data.csv   --output /opt/airflow/data/processed/cleaned_house_data.csv'
    )
    feature_engineering = BashOperator(
        task_id='feature_engineering',
        bash_command= 'python /opt/airflow/src/features/engineer.py   --input /opt/airflow/data/processed/cleaned_house_data.csv   --output /opt/airflow/data/processed/featured_house_data.csv   --preprocessor /opt/airflow/models/trained/preprocessor.pkl'
    )
    train_model = BashOperator(
        task_id='train_model',
        bash_command= 'python /opt/airflow/src/models/train_model.py   --config /opt/airflow/configs/model_config.yaml   --data /opt/airflow/data/processed/featured_house_data.csv   --models-dir /opt/airflow/models   --mlflow-tracking-uri http://mlflow:5000'
    )
    preprocess >> feature_engineering >> train_model