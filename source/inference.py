import json
import pickle
import numpy as np
import logging
from sklearn.ensemble import GradientBoostingClassifier

logger = logging.getLogger(__name__)

def model_fn(model_dir):
    """モデルをロード"""
    try:
        with open(f"{model_dir}/model.pkl", "rb") as f:
            model = pickle.load(f)
        logger.info("Model loaded successfully from pickle file")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def input_fn(request_body, request_content_type):
    """入力データを処理"""
    if request_content_type == "application/json":
        try:
            input_data = json.loads(request_body)
            
            # 単一サンプルの場合
            if "instances" in input_data:
                data = np.array(input_data["instances"])
            elif isinstance(input_data, list):
                data = np.array(input_data)
            else:
                data = np.array([input_data])
                
            logger.info(f"Input data shape: {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Error parsing input: {str(e)}")
            raise ValueError(f"Invalid input format: {str(e)}")
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """予測を実行"""
    try:
        # 確率予測
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(input_data)
            # 2値分類の場合、クラス1の確率を返す
            if len(probabilities.shape) == 2 and probabilities.shape[1] == 2:
                predictions = probabilities[:, 1]
            else:
                predictions = probabilities
        else:
            # predict_probaがない場合はdecision_functionを使用
            predictions = model.decision_function(input_data)
            # シグモイド関数で確率に変換
            predictions = 1 / (1 + np.exp(-predictions))
            
        # 閾値0.5でクラス予測
        classes = (predictions > 0.5).astype(int)
        
        logger.info(f"Predictions generated for {len(predictions)} samples")
        
        return {
            "predictions": classes.tolist(),
            "probabilities": predictions.tolist()
        }
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise

def output_fn(prediction, content_type):
    """出力を処理"""
    if content_type == "application/json":
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")
