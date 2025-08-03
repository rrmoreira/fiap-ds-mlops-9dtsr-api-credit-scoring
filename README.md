# Projeto Final QuantumFinance Credit Scoring - MLOps - 9dtsr 2025

Este é uma projeto realizado como parte final da entrega da disciplica de Machine Learning Engineering - MLOps, 
onde tem como objetivo criar um fuxo end-to-end da analise para um sore de credito do cliente da QuantumFinance.

Nele contempla desde a analise do <a target="_blank" href="https://www.kaggle.com/datasets/parisrohan/credit-score-classification">dataset<a/> 
para pré-processamento e criação do modelo preditivo até a disponibilização de um frontend para consumo deste modelo.


## Project Organization

Este projeto é a criação da API que executa o modelo de ML e disponibiliza os dados da predição, no fluxo ela sera utilizada pelo frontend. 
Ela foi desenvolvida para ser utilizada em uma lambda functions, a extruturação do container que será registrado no ECR se encontra no Dockerfile.

```
├── README.md                <- The top-level README for developers using this project.
├── model
│   ├── model.pkl            <- Data from model prediction.
│   └── model_metadata.json  <- Json model template
├── src
│   └── app.py               <- Code to create visualizations
│
├── Dockerfile               <- Template to create cotainer
│
├── data.json                <- Json template
│
├── model_downloader.py      <- Code to download model
│                  
│
├── teste_api.py             <- Code to test model execution
│
└── requirements.txt         <- The requirements file for reproducing the analysis environment, e.g.

```
