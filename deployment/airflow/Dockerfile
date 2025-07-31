FROM apache/airflow:latest

USER root 
RUN apt-get update && \
    apt-get -y install git && \
    apt-get clean

USER airflow
COPY deployment/airflow/requirements.txt .
COPY src src
RUN mkdir -p data/processed data/raw configs models/trained
COPY data/raw/house_data.csv data/raw/
COPY configs/model_config.yaml configs/
RUN pip install --no-cache-dir -r requirements.txt