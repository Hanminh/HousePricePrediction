FROM jenkins/jenkins:lts

USER root

RUN apt-get update && \
    apt-get install -y docker.io docker-compose python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

USER jenkins

EXPOSE 6060