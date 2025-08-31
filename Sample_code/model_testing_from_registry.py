import mlflow
from mlflow.tracking import MlflowClient
import dagshub

dagshub.init(repo_owner='satyajeetrai007', repo_name='Youtube-Comment-Sentiment-Analysis', mlflow=True)

def load_model_from_registry(model_name, model_version):
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)

    return model

model = load_model_from_registry("yt_chrome_plugin_model","2")
print("model loaded successfully")