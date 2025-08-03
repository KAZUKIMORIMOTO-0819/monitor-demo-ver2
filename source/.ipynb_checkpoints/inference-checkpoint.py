import json
import os
import subprocess
import sys
import logging
import pickle
import time

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def install_lightgbm():
    """LightGBMをインストールする関数"""
    try:
        import lightgbm
        logger.info("LightGBM is already installed")
        return True
    except ImportError:
        logger.info("Installing LightGBM...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm==4.1.0", "--no-cache-dir"])
            logger.info("LightGBM installation completed")
            return True
        except Exception as e:
            logger.error(f"Failed to install LightGBM: {e}")
            return False

# インストール実行
if not install_lightgbm():
    raise ImportError("Failed to install LightGBM")

# インストール後にインポート
import lightgbm as lgb
import numpy as np

def model_fn(model_dir):
    """モデルを読み込む関数（Pickle形式）"""
    logger.info(f"Loading model from {model_dir}")
    
    model_path = os.path.join(model_dir, 'model.pkl')
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        if os.path.exists(model_dir):
            files = os.listdir(model_dir)
            logger.info(f"Files in model directory: {files}")
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        logger.info(f"Model loaded successfully from pickle. Type: {type(model)}")
        
        if hasattr(model, 'num_feature'):
            logger.info(f"Model features: {model.num_feature()}")
        
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def input_fn(request_body, request_content_type):
    """入力データを処理する関数（Model Monitor対応）"""
    logger.info(f"Processing input with content type: {request_content_type}")
    
    if request_content_type == 'application/json':
        try:
            if isinstance(request_body, str):
                input_data = json.loads(request_body)
            else:
                input_data = request_body
                
            logger.info(f"Input data structure: {type(input_data)}")
            
            # 複数のデータ形式に対応
            if 'instances' in input_data:
                data = input_data['instances']
            elif 'data' in input_data:
                data = input_data['data']
            elif isinstance(input_data, list):
                data = input_data
            else:
                data = [input_data]
            
            # numpy配列に変換
            np_data = np.array(data, dtype=np.float32)
            logger.info(f"Converted to numpy array with shape: {np_data.shape}")
            
            # 特徴量数をチェック（20個の特徴量を期待）
            if np_data.shape[1] != 20:
                raise ValueError(f"Expected 20 features, got {np_data.shape[1]}")
            
            return np_data
            
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            raise ValueError(f"Error processing JSON input: {e}")
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """予測を実行する関数"""
    try:
        logger.info(f"Making prediction for data shape: {input_data.shape}")
        
        # モデルの特徴量数をチェック
        if hasattr(model, 'num_feature'):
            expected_features = model.num_feature()
            if input_data.shape[1] != expected_features:
                raise ValueError(f"Feature mismatch: model expects {expected_features}, got {input_data.shape[1]}")
        
        # 予測実行
        predictions = model.predict(input_data)
        logger.info(f"Predictions completed. Shape: {predictions.shape}")
        
        return predictions.tolist()
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise

def output_fn(prediction, content_type):
    """出力を処理する関数（Model Monitor対応）"""
    logger.info(f"Formatting output with content type: {content_type}")
    
    if content_type == 'application/json':
        try:
            # Model Monitorが解析しやすい形式で出力
            result = {
                'predictions': prediction,
                'model_name': 'lightgbm_binary_classifier',
                'model_version': '1.0',
                'timestamp': str(int(time.time() * 1000))
            }
            return json.dumps(result)
        except Exception as e:
            logger.error(f"Error formatting output: {e}")
            raise
    else:
        raise ValueError(f"Unsupported content type: {content_type}")
