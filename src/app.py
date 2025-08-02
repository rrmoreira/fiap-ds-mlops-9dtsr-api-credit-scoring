"Api para validacao de score de credito usando Machine Learning"
from datetime import datetime
import json
import boto3
import joblib

model = joblib.load("model/model.pkl")

with open("model/model_metadata.json", "r", encoding="utf-8") as f:
    model_info = json.load(f)
    
cloudwatch = boto3.client("cloudwatch", region_name="us-east-1")

def write_real_data(data, prediction):
    """
    Funcao para escrever os dados consumidos para depois serem estudados 
    para desvios de dados, modelo ou conceito.
    Args:
        data (dict): dicionario de dados com o perfil do cliente a ser analisado.
        prediction (float): valor de predicao (preço do laptop).
    """
    now = datetime.now()
    now_formatted = now.strftime("%d-%m-%Y %H:%M")
    
    file_name = f"{now.strftime('%Y-%m-%d')}_score_prediction_data.csv"
    
    data["price"] = prediction
    data["timestamp"] = now_formatted
    data["model_version"] = model_info["version"]
    
    s3 = boto3.client("s3")
    bucket_name = "fiap-ds-mlops-credit-scoring"
    s3_path = "credit-scoring-data"
    
    try:
        existing_object = s3.get_object(Bucket=bucket_name, Key=f'{s3_path}/{file_name}')
        existing_data = existing_object['Body'].read().decode('utf-8').strip().split('\n')
        existing_data.append(','.join(map(str, data.values())))
        updated_content = '\n'.join(existing_data)
    except s3.exceptions.NoSuchKey:
        # Se o arquivo não existir, cria um novo
        updated_content = ','.join(data.keys()) + '\n' + ','.join(map(str, data.values()))
        
    s3.put_object(Body=updated_content, Bucket=bucket_name, Key=f'{s3_path}/{file_name}')

def input_metrics(data, prediction):
    """
    Funcao para escrever metricas customizadas no CloudWatch.
    
    Args:
        data (dict): dicionario de dados com todos atributos do cliente.
        prediction (float): valor da predicao (score).
    """
    cloudwatch.put_metric_data(
        MetricData = [
            {
                'MetricName': 'Score Prediction',
                'Value': prediction,
                'Dimensions': [{'Name': "Currency", 'Value': "INR"}]
            },
        ], Namespace='Credit Score Model')
        
    for key, value in data.items():
        cloudwatch.put_metric_data(
            MetricData = [
                {
                    'MetricName': 'Score Feature',
                    'Value': 1,
                    'Unit': 'Count',
                    'Dimensions': [{'Name': key, 'Value': str(value)}]
                },
            ], Namespace='Credit Score Features')

def prepare_payload(data):
    """
    Prepara o payload para a API.
    
    Args:
        data (dict): dicionario de dados com todos atributos do cliente.
    
    Returns:
        dict: dicionario com os dados preparados para a API.
    """
    data_processed = []
    
    data_processed.append(int(data["age"]))
    data_processed.append(float(data["monthly_inhand_salary"]))
    data_processed.append(int(data["num_bank_accounts"]))
    data_processed.append(int(data["num_credit_card"]))
    data_processed.append(float(data["interest_rate"]))
    data_processed.append(int(data["num_of_loan"]))
    data_processed.append(int(data["delay_from_due_date"]))
    data_processed.append(int(data["num_of_delayed_payment"]))
    data_processed.append(int(data["num_credit_inquiries"]))
    data_processed.append(float(data["credit_utilization_ratio"]))
    data_processed.append(float(data["total_emi_per_month"]))
    data_processed.append(float(data["amount_invested_monthly"]))
    data_processed.append(float(data["monthly_balance"]))
    data_processed.append(float(data["outstanding_debt"]))
    data_processed.append(float(data["changed_credit_limit"]))
    data_processed.append(float(data["annual_income"]))
        
    conditions = {
        "payment_behaviour": {
            "high_spent_large_value_payments","high_spent_medium_value_payments",
            "high_spent_small_value_payments","low_spent_large_value_payments",
            "low_spent_medium_value_payments","low_spent_small_value_payments",
            "other"
        },
        "credit_mix": {"bad","good","standard","other"},
        "payment_of_min_amount": {"yes","no","other"},
    }
    
    for key, values in conditions.items():
        for value in values:
            data_processed.append(1 if data[key] == value else 0)
            
    return data_processed

def handler(event, context=False):
    """
        Função de entrada para execucao do modelo.
        Args:
            event (dict): dicionario de dados com todos os atributos do cliente.
            context (object, optional): Contexto da execução (opcional).
            
            Returns:
                json: predicao do score "0 -  Good, 1 -  Standard, 2 -  Poor".
    """

    print(event)
    print(context)
    
    if "body" in event:
        print("Body found in event, invoke by API Gateway")
        
        body_str = event.get("body", "{}")
        body = json.loads(body_str)
        print(body)
        
        data = body.get("data", {})
    else:
        print("No body found in event, invoke by Lambda")
        data = event.get("data", {})
        
    print(data)
        
    data_processed = prepare_payload(data)
    
    prediction = model.predict([data_processed])
   
    
    prediction = int(prediction[0])
    print(f"Prediction: {prediction}")
     
    write_real_data(data, prediction)
    input_metrics(data, prediction)
    
    return { 
        "statusCode": 200,
        "headers": {
            'Content-Type': 'application/json'
        },
        "body": json.dumps({
            'prediction': prediction,
            'version': model_info["version"],
        })
    }
