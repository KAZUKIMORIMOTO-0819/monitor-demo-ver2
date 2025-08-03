#!/usr/bin/env python3
"""
LightGBM Binary Classification + SageMaker Model Monitor v2
改良版実装：
- より堅牢なエラーハンドリング
- 詳細なログ出力
- 設定可能なパラメータ
- 自動クリーンアップ機能
- 包括的なテスト機能
"""

import sagemaker
import boto3
import pandas as pd
import numpy as np
import lightgbm as lgb
import json
import os
import time
import logging
import tarfile
import pickle
import warnings
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# SageMaker関連
from sagemaker.session import Session
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
from sagemaker.image_uris import retrieve
from sagemaker.model_monitor import (
    DataCaptureConfig,
    DefaultModelMonitor,
    DatasetFormat,
    EndpointInput,
    CronExpressionGenerator
)

# 警告を抑制
warnings.filterwarnings('ignore')

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LightGBMBinaryClassifierV2:
    """
    LightGBM 2値分類モデル + SageMaker Model Monitor v2
    """
    
    def __init__(self, config=None):
        """
        初期化
        
        Args:
            config (dict): 設定パラメータ
        """
        self.session = sagemaker.Session()
        self.role = sagemaker.get_execution_role()
        self.region = self.session.boto_region_name
        self.bucket = self.session.default_bucket()
        self.s3_client = boto3.client('s3')
        self.sm_client = boto3.client('sagemaker')
        
        # デフォルト設定
        self.config = {
            # データ生成設定
            'n_samples': 3000,
            'n_features': 25,
            'n_informative': 20,
            'n_redundant': 5,
            'n_clusters_per_class': 2,
            'test_size': 0.2,
            'val_size': 0.1,
            'random_state': 42,
            
            # LightGBMパラメータ
            'lgb_params': {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42
            },
            'num_boost_round': 100,
            'early_stopping_rounds': 10,
            
            # デプロイ設定
            'model_name': f'lightgbm-binary-v2-{int(time.time())}',
            'endpoint_name': f'lightgbm-binary-endpoint-v2-{int(time.time())}',
            'instance_type': 'ml.m5.large',
            'initial_instance_count': 1,
            
            # Data Capture設定
            'enable_data_capture': True,
            'sampling_percentage': 100,
            'capture_modes': ['Input', 'Output'],
            
            # Model Monitor設定
            'monitor_schedule_name': f'lightgbm-monitor-v2-{int(time.time())}',
            'monitoring_instance_type': 'ml.m5.xlarge',
            'monitoring_cron': 'cron(0/15 * ? * * *)',  # 15分毎
            'max_runtime_seconds': 3600,
            
            # S3設定
            's3_prefix': 'lightgbm-binary-classification-v2',
            'model_artifacts_path': 'model',
            'baseline_path': 'baseline',
            'monitoring_reports_path': 'monitoring-reports',
            'data_capture_path': 'datacapture'
        }
        
        # 設定を更新
        if config:
            self.config.update(config)
            
        # S3パス設定
        self.s3_model_path = f's3://{self.bucket}/{self.config["s3_prefix"]}/{self.config["model_artifacts_path"]}'
        self.s3_baseline_path = f's3://{self.bucket}/{self.config["s3_prefix"]}/{self.config["baseline_path"]}'
        self.s3_monitoring_path = f's3://{self.bucket}/{self.config["s3_prefix"]}/{self.config["monitoring_reports_path"]}'
        self.s3_data_capture_path = f's3://{self.bucket}/{self.config["s3_prefix"]}/{self.config["data_capture_path"]}'
        
        # 状態管理
        self.model = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.predictor = None
        self.monitor = None
        
        logger.info(f"LightGBM Binary Classifier v2 initialized")
        logger.info(f"Bucket: {self.bucket}")
        logger.info(f"Region: {self.region}")
        logger.info(f"S3 Prefix: {self.config['s3_prefix']}")
    
    def generate_data(self):
        """
        2値分類用のサンプルデータを生成
        """
        logger.info("Generating sample data for binary classification...")
        
        # データ生成
        X, y = make_classification(
            n_samples=self.config['n_samples'],
            n_features=self.config['n_features'],
            n_informative=self.config['n_informative'],
            n_redundant=self.config['n_redundant'],
            n_clusters_per_class=self.config['n_clusters_per_class'],
            random_state=self.config['random_state']
        )
        
        # データ分割
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=self.config['test_size'], 
            random_state=self.config['random_state'], stratify=y
        )
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=self.config['val_size']/(1-self.config['test_size']), 
            random_state=self.config['random_state'], stratify=y_temp
        )
        
        logger.info(f"Data generated successfully:")
        logger.info(f"  Train: {self.X_train.shape[0]} samples")
        logger.info(f"  Validation: {self.X_val.shape[0]} samples")
        logger.info(f"  Test: {self.X_test.shape[0]} samples")
        logger.info(f"  Features: {self.X_train.shape[1]}")
        logger.info(f"  Class distribution (train): {np.bincount(self.y_train)}")
        
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
    
    def train_model(self):
        """
        LightGBMモデルを訓練
        """
        if self.X_train is None:
            raise ValueError("Data not generated. Call generate_data() first.")
            
        logger.info("Training LightGBM model...")
        
        # LightGBMデータセット作成
        train_data = lgb.Dataset(self.X_train, label=self.y_train)
        val_data = lgb.Dataset(self.X_val, label=self.y_val, reference=train_data)
        
        # モデル訓練
        self.model = lgb.train(
            self.config['lgb_params'],
            train_data,
            num_boost_round=self.config['num_boost_round'],
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=[
                lgb.early_stopping(self.config['early_stopping_rounds']),
                lgb.log_evaluation(period=10)
            ]
        )
        
        # モデル評価
        self._evaluate_model()
        
        logger.info("Model training completed successfully")
        return self.model
    
    def _evaluate_model(self):
        """
        モデルの性能を評価
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
            
        # 予測
        y_pred_proba = self.model.predict(self.X_test, num_iteration=self.model.best_iteration)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # メトリクス計算
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        auc = roc_auc_score(self.y_test, y_pred_proba)
        
        logger.info("Model Performance Metrics:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1-Score: {f1:.4f}")
        logger.info(f"  AUC: {auc:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc
        }
    
    def save_model(self):
        """
        モデルをS3に保存
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
            
        logger.info("Saving model artifacts...")
        
        # ローカルディレクトリ作成
        os.makedirs('model', exist_ok=True)
        
        # モデルをpickle形式で保存（LightGBMのunordered_map::atエラー回避）
        with open('model/model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        
        # テキスト形式でも保存（デバッグ用）
        self.model.save_model('model/model.txt')
        
        # tar.gz形式でアーカイブ
        with tarfile.open('model.tar.gz', 'w:gz') as tar:
            tar.add('model', arcname='.')
        
        # S3にアップロード
        model_uri = f'{self.s3_model_path}/model.tar.gz'
        self.session.upload_data('model.tar.gz', bucket=self.bucket, 
                                key_prefix=f'{self.config["s3_prefix"]}/{self.config["model_artifacts_path"]}')
        
        logger.info(f"Model saved to: {model_uri}")
        return model_uri
    
    def create_inference_script(self):
        """
        推論スクリプトを作成
        """
        logger.info("Creating inference script...")
        
        os.makedirs('source', exist_ok=True)
        
        inference_code = '''import json
import pickle
import numpy as np
import lightgbm as lgb
import logging

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
        probabilities = model.predict(input_data, num_iteration=model.best_iteration)
        
        # 2値分類の場合、クラス1の確率を返す
        if len(probabilities.shape) == 1:
            predictions = probabilities
        else:
            predictions = probabilities[:, 1]
            
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
'''
        
        with open('source/inference.py', 'w') as f:
            f.write(inference_code)
        
        # requirements.txt作成
        requirements = '''lightgbm>=3.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
'''
        
        with open('source/requirements.txt', 'w') as f:
            f.write(requirements)
        
        logger.info("Inference script created successfully")
    
    def deploy_model(self):
        """
        モデルをSageMakerエンドポイントにデプロイ
        """
        logger.info("Deploying model to SageMaker endpoint...")
        
        # 推論スクリプト作成
        self.create_inference_script()
        
        # モデル保存
        model_uri = self.save_model()
        
        # LightGBM用のコンテナイメージ取得
        image_uri = retrieve(
            framework='sklearn',
            region=self.region,
            version='1.0-1',
            py_version='py3',
            instance_type=self.config['instance_type']
        )
        
        # Data Capture設定
        data_capture_config = None
        if self.config['enable_data_capture']:
            data_capture_config = DataCaptureConfig(
                enable_capture=True,
                sampling_percentage=self.config['sampling_percentage'],
                destination_s3_uri=self.s3_data_capture_path,
                capture_options=self.config['capture_modes']
            )
            logger.info(f"Data capture enabled: {self.s3_data_capture_path}")
        
        # SageMakerモデル作成
        model = Model(
            image_uri=image_uri,
            model_data=model_uri,
            role=self.role,
            name=self.config['model_name'],
            source_dir='source',
            entry_point='inference.py'
        )
        
        # エンドポイントデプロイ
        self.predictor = model.deploy(
            initial_instance_count=self.config['initial_instance_count'],
            instance_type=self.config['instance_type'],
            endpoint_name=self.config['endpoint_name'],
            data_capture_config=data_capture_config,
            serializer=JSONSerializer(),
            deserializer=JSONDeserializer()
        )
        
        logger.info(f"Model deployed successfully to endpoint: {self.config['endpoint_name']}")
        return self.predictor
    
    def test_endpoint(self, n_samples=10):
        """
        エンドポイントをテスト
        """
        if self.predictor is None:
            raise ValueError("Model not deployed. Call deploy_model() first.")
            
        logger.info(f"Testing endpoint with {n_samples} samples...")
        
        # テストデータ準備
        test_samples = self.X_test[:n_samples].tolist()
        
        try:
            # 予測実行
            response = self.predictor.predict({"instances": test_samples})
            
            logger.info("Endpoint test successful!")
            logger.info(f"Sample response: {response}")
            
            return response
            
        except Exception as e:
            logger.error(f"Endpoint test failed: {str(e)}")
            raise
    
    def create_baseline(self):
        """
        Model Monitor用のベースライン統計を作成
        """
        if self.X_train is None:
            raise ValueError("Data not generated. Call generate_data() first.")
            
        logger.info("Creating baseline statistics for Model Monitor...")
        
        # ベースラインデータ準備（訓練データの一部を使用）
        baseline_size = min(1000, len(self.X_train))
        baseline_indices = np.random.choice(len(self.X_train), baseline_size, replace=False)
        baseline_X = self.X_train[baseline_indices]
        
        # 特徴量名を生成
        feature_names = [f'feature_{i}' for i in range(baseline_X.shape[1])]
        
        # DataFrameに変換
        baseline_df = pd.DataFrame(baseline_X, columns=feature_names)
        
        # ローカルに保存
        os.makedirs('baseline_data', exist_ok=True)
        baseline_path = 'baseline_data/baseline.csv'
        baseline_df.to_csv(baseline_path, index=False)
        
        # S3にアップロード
        s3_baseline_data_uri = f'{self.s3_baseline_path}/baseline.csv'
        self.session.upload_data(
            baseline_path, 
            bucket=self.bucket,
            key_prefix=f'{self.config["s3_prefix"]}/{self.config["baseline_path"]}'
        )
        
        # DefaultModelMonitor初期化
        self.monitor = DefaultModelMonitor(
            role=self.role,
            instance_count=1,
            instance_type=self.config['monitoring_instance_type'],
            volume_size_in_gb=20,
            max_runtime_in_seconds=self.config['max_runtime_seconds']
        )
        
        # ベースライン統計作成
        baseline_job_name = f'baseline-job-{int(time.time())}'
        
        self.monitor.suggest_baseline(
            baseline_dataset=s3_baseline_data_uri,
            dataset_format=DatasetFormat.csv(header=True),
            output_s3_uri=f'{self.s3_baseline_path}/output',
            job_name=baseline_job_name
        )
        
        logger.info(f"Baseline creation job started: {baseline_job_name}")
        logger.info(f"Baseline data uploaded to: {s3_baseline_data_uri}")
        
        return s3_baseline_data_uri
    
    def setup_monitoring(self):
        """
        Model Monitoringスケジュールを設定
        """
        if self.monitor is None:
            raise ValueError("Baseline not created. Call create_baseline() first.")
        if self.predictor is None:
            raise ValueError("Model not deployed. Call deploy_model() first.")
            
        logger.info("Setting up Model Monitor schedule...")
        
        try:
            # モニタリングスケジュール作成
            self.monitor.create_monitoring_schedule(
                monitor_schedule_name=self.config['monitor_schedule_name'],
                endpoint_input=EndpointInput(
                    endpoint_name=self.config['endpoint_name'],
                    destination=f'{self.s3_monitoring_path}/input'
                ),
                output_s3_uri=f'{self.s3_monitoring_path}/output',
                statistics=f'{self.s3_baseline_path}/output/statistics.json',
                constraints=f'{self.s3_baseline_path}/output/constraints.json',
                schedule_cron_expression=CronExpressionGenerator.hourly(),
                enable_cloudwatch_metrics=True
            )
            
            logger.info(f"Monitoring schedule created: {self.config['monitor_schedule_name']}")
            logger.info(f"Monitoring reports will be saved to: {self.s3_monitoring_path}")
            
        except Exception as e:
            logger.error(f"Failed to create monitoring schedule: {str(e)}")
            raise
    
    def generate_traffic(self, n_requests=50, interval=2):
        """
        テストトラフィックを生成してData Captureを促進
        """
        if self.predictor is None:
            raise ValueError("Model not deployed. Call deploy_model() first.")
            
        logger.info(f"Generating {n_requests} test requests...")
        
        responses = []
        
        for i in range(n_requests):
            try:
                # ランダムなテストサンプル選択
                sample_idx = np.random.randint(0, len(self.X_test))
                test_sample = self.X_test[sample_idx:sample_idx+1].tolist()
                
                # 予測実行
                response = self.predictor.predict({"instances": test_sample})
                responses.append(response)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Sent {i + 1}/{n_requests} requests")
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in request {i + 1}: {str(e)}")
                continue
        
        logger.info(f"Traffic generation completed. {len(responses)} successful requests")
        return responses
    
    def check_monitoring_status(self):
        """
        モニタリングスケジュールの状態を確認
        """
        if self.monitor is None:
            logger.warning("Monitor not initialized")
            return None
            
        try:
            # スケジュール状態確認
            schedule_desc = self.monitor.describe_schedule()
            logger.info(f"Monitoring schedule status: {schedule_desc['MonitoringScheduleStatus']}")
            
            # 最新の実行状態確認
            executions = self.monitor.list_executions()
            if executions:
                latest_execution = executions[0]
                logger.info(f"Latest execution status: {latest_execution['ProcessingJobStatus']}")
                logger.info(f"Latest execution time: {latest_execution['CreationTime']}")
            
            return schedule_desc
            
        except Exception as e:
            logger.error(f"Error checking monitoring status: {str(e)}")
            return None
    
    def cleanup(self):
        """
        リソースをクリーンアップ
        """
        logger.info("Starting cleanup process...")
        
        try:
            # モニタリングスケジュール停止・削除
            if self.monitor:
                try:
                    self.monitor.stop_monitoring_schedule()
                    logger.info("Monitoring schedule stopped")
                    
                    time.sleep(10)  # 停止を待つ
                    
                    self.monitor.delete_monitoring_schedule()
                    logger.info("Monitoring schedule deleted")
                except Exception as e:
                    logger.warning(f"Error cleaning up monitor: {str(e)}")
            
            # エンドポイント削除
            if self.predictor:
                try:
                    self.predictor.delete_endpoint()
                    logger.info("Endpoint deleted")
                except Exception as e:
                    logger.warning(f"Error deleting endpoint: {str(e)}")
            
            logger.info("Cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
    
    def run_complete_pipeline(self):
        """
        完全なパイプラインを実行
        """
        logger.info("Starting complete LightGBM + Model Monitor pipeline...")
        
        try:
            # 1. データ生成
            self.generate_data()
            
            # 2. モデル訓練
            self.train_model()
            
            # 3. モデルデプロイ
            self.deploy_model()
            
            # 4. エンドポイントテスト
            self.test_endpoint()
            
            # 5. ベースライン作成
            self.create_baseline()
            
            # 6. モニタリング設定
            self.setup_monitoring()
            
            # 7. テストトラフィック生成
            self.generate_traffic(n_requests=30)
            
            # 8. モニタリング状態確認
            self.check_monitoring_status()
            
            logger.info("Complete pipeline executed successfully!")
            logger.info(f"Endpoint: {self.config['endpoint_name']}")
            logger.info(f"Monitor: {self.config['monitor_schedule_name']}")
            logger.info(f"S3 Bucket: {self.bucket}")
            logger.info(f"S3 Prefix: {self.config['s3_prefix']}")
            
            return {
                'endpoint_name': self.config['endpoint_name'],
                'monitor_schedule_name': self.config['monitor_schedule_name'],
                'bucket': self.bucket,
                's3_prefix': self.config['s3_prefix']
            }
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            # エラー時もクリーンアップを試行
            self.cleanup()
            raise


def main():
    """
    メイン実行関数
    """
    # カスタム設定（必要に応じて変更）
    custom_config = {
        'n_samples': 2500,
        'n_features': 30,
        'lgb_params': {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        },
        'monitoring_cron': 'cron(0/10 * ? * * *)',  # 10分毎
    }
    
    # パイプライン実行
    classifier = LightGBMBinaryClassifierV2(config=custom_config)
    
    try:
        result = classifier.run_complete_pipeline()
        print("\n" + "="*60)
        print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Endpoint Name: {result['endpoint_name']}")
        print(f"Monitor Schedule: {result['monitor_schedule_name']}")
        print(f"S3 Bucket: {result['bucket']}")
        print(f"S3 Prefix: {result['s3_prefix']}")
        print("\nTo clean up resources, run:")
        print("classifier.cleanup()")
        print("="*60)
        
        return classifier
        
    except Exception as e:
        print(f"\nPipeline execution failed: {str(e)}")
        return None


if __name__ == "__main__":
    classifier = main()
