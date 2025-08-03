#!/usr/bin/env python3
"""
LightGBM Binary Classifier v2 テストスクリプト
デプロイされたエンドポイントとModel Monitorの動作をテストします。
"""

import boto3
import json
import numpy as np
import pandas as pd
import time
from datetime import datetime
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LightGBMV2Tester:
    """
    LightGBM v2のテストクラス
    """
    
    def __init__(self, endpoint_name, region='us-east-1'):
        """
        初期化
        
        Args:
            endpoint_name (str): エンドポイント名
            region (str): AWSリージョン
        """
        self.endpoint_name = endpoint_name
        self.region = region
        self.runtime = boto3.client('sagemaker-runtime', region_name=region)
        self.s3_client = boto3.client('s3', region_name=region)
        self.sm_client = boto3.client('sagemaker', region_name=region)
        
        logger.info(f"Tester initialized for endpoint: {endpoint_name}")
    
    def generate_test_data(self, n_samples=100, n_features=25):
        """
        テスト用データを生成
        
        Args:
            n_samples (int): サンプル数
            n_features (int): 特徴量数
            
        Returns:
            np.ndarray: テストデータ
        """
        logger.info(f"Generating {n_samples} test samples with {n_features} features")
        
        # ランダムなテストデータ生成
        np.random.seed(42)
        test_data = np.random.randn(n_samples, n_features)
        
        return test_data
    
    def test_single_prediction(self, test_sample=None):
        """
        単一予測をテスト
        
        Args:
            test_sample (list): テストサンプル（Noneの場合は自動生成）
            
        Returns:
            dict: 予測結果
        """
        if test_sample is None:
            test_sample = self.generate_test_data(1, 25)[0].tolist()
        
        logger.info("Testing single prediction...")
        
        try:
            # 予測実行
            payload = json.dumps({"instances": [test_sample]})
            
            response = self.runtime.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType='application/json',
                Body=payload
            )
            
            result = json.loads(response['Body'].read().decode())
            
            logger.info("✅ Single prediction successful")
            logger.info(f"Result: {result}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Single prediction failed: {str(e)}")
            raise
    
    def test_batch_prediction(self, batch_size=10):
        """
        バッチ予測をテスト
        
        Args:
            batch_size (int): バッチサイズ
            
        Returns:
            dict: 予測結果
        """
        logger.info(f"Testing batch prediction with {batch_size} samples...")
        
        try:
            # バッチデータ生成
            test_batch = self.generate_test_data(batch_size, 25).tolist()
            
            # 予測実行
            payload = json.dumps({"instances": test_batch})
            
            response = self.runtime.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType='application/json',
                Body=payload
            )
            
            result = json.loads(response['Body'].read().decode())
            
            logger.info("✅ Batch prediction successful")
            logger.info(f"Predicted {len(result['predictions'])} samples")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Batch prediction failed: {str(e)}")
            raise
    
    def stress_test(self, n_requests=50, interval=1):
        """
        ストレステスト（連続リクエスト）
        
        Args:
            n_requests (int): リクエスト数
            interval (float): リクエスト間隔（秒）
            
        Returns:
            dict: テスト結果統計
        """
        logger.info(f"Starting stress test: {n_requests} requests with {interval}s interval")
        
        results = {
            'total_requests': n_requests,
            'successful_requests': 0,
            'failed_requests': 0,
            'response_times': [],
            'errors': []
        }
        
        for i in range(n_requests):
            try:
                start_time = time.time()
                
                # テストデータ生成
                test_sample = self.generate_test_data(1, 25)[0].tolist()
                payload = json.dumps({"instances": [test_sample]})
                
                # 予測実行
                response = self.runtime.invoke_endpoint(
                    EndpointName=self.endpoint_name,
                    ContentType='application/json',
                    Body=payload
                )
                
                end_time = time.time()
                response_time = end_time - start_time
                
                results['successful_requests'] += 1
                results['response_times'].append(response_time)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Completed {i + 1}/{n_requests} requests")
                
                time.sleep(interval)
                
            except Exception as e:
                results['failed_requests'] += 1
                results['errors'].append(str(e))
                logger.warning(f"Request {i + 1} failed: {str(e)}")
        
        # 統計計算\n        if results['response_times']:\n            results['avg_response_time'] = np.mean(results['response_times'])\n            results['min_response_time'] = np.min(results['response_times'])\n            results['max_response_time'] = np.max(results['response_times'])\n            results['p95_response_time'] = np.percentile(results['response_times'], 95)\n        \n        logger.info(\"✅ Stress test completed\")\n        logger.info(f\"Success rate: {results['successful_requests']}/{results['total_requests']}\")\n        if results['response_times']:\n            logger.info(f\"Avg response time: {results['avg_response_time']:.3f}s\")\n        \n        return results\n    \n    def check_endpoint_status(self):\n        \"\"\"\n        エンドポイントの状態を確認\n        \n        Returns:\n            dict: エンドポイント情報\n        \"\"\"\n        logger.info(\"Checking endpoint status...\")\n        \n        try:\n            response = self.sm_client.describe_endpoint(\n                EndpointName=self.endpoint_name\n            )\n            \n            status = response['EndpointStatus']\n            logger.info(f\"Endpoint status: {status}\")\n            \n            if status == 'InService':\n                logger.info(\"✅ Endpoint is ready for inference\")\n            else:\n                logger.warning(f\"⚠️  Endpoint is not ready: {status}\")\n            \n            return response\n            \n        except Exception as e:\n            logger.error(f\"❌ Failed to check endpoint status: {str(e)}\")\n            raise\n    \n    def check_data_capture(self, bucket, s3_prefix):\n        \"\"\"\n        Data Captureファイルを確認\n        \n        Args:\n            bucket (str): S3バケット名\n            s3_prefix (str): S3プレフィックス\n            \n        Returns:\n            list: Data Captureファイル一覧\n        \"\"\"\n        logger.info(\"Checking data capture files...\")\n        \n        try:\n            response = self.s3_client.list_objects_v2(\n                Bucket=bucket,\n                Prefix=f'{s3_prefix}/datacapture'\n            )\n            \n            files = []\n            if 'Contents' in response:\n                files = response['Contents']\n                logger.info(f\"Found {len(files)} data capture files\")\n                \n                # 最新の5ファイルを表示\n                sorted_files = sorted(files, key=lambda x: x['LastModified'], reverse=True)\n                for file in sorted_files[:5]:\n                    logger.info(f\"  - {file['Key']} ({file['Size']} bytes, {file['LastModified']})\")\n            else:\n                logger.info(\"No data capture files found\")\n            \n            return files\n            \n        except Exception as e:\n            logger.error(f\"❌ Failed to check data capture: {str(e)}\")\n            return []\n    \n    def run_comprehensive_test(self, bucket=None, s3_prefix=None):\n        \"\"\"\n        包括的なテストを実行\n        \n        Args:\n            bucket (str): S3バケット名（Data Capture確認用）\n            s3_prefix (str): S3プレフィックス（Data Capture確認用）\n            \n        Returns:\n            dict: テスト結果\n        \"\"\"\n        logger.info(\"Starting comprehensive test...\")\n        \n        test_results = {\n            'timestamp': datetime.now().isoformat(),\n            'endpoint_name': self.endpoint_name,\n            'tests': {}\n        }\n        \n        try:\n            # 1. エンドポイント状態確認\n            logger.info(\"\\n1. Checking endpoint status...\")\n            endpoint_status = self.check_endpoint_status()\n            test_results['tests']['endpoint_status'] = {\n                'status': endpoint_status['EndpointStatus'],\n                'success': endpoint_status['EndpointStatus'] == 'InService'\n            }\n            \n            # 2. 単一予測テスト\n            logger.info(\"\\n2. Testing single prediction...\")\n            try:\n                single_result = self.test_single_prediction()\n                test_results['tests']['single_prediction'] = {\n                    'success': True,\n                    'result': single_result\n                }\n            except Exception as e:\n                test_results['tests']['single_prediction'] = {\n                    'success': False,\n                    'error': str(e)\n                }\n            \n            # 3. バッチ予測テスト\n            logger.info(\"\\n3. Testing batch prediction...\")\n            try:\n                batch_result = self.test_batch_prediction()\n                test_results['tests']['batch_prediction'] = {\n                    'success': True,\n                    'batch_size': len(batch_result['predictions'])\n                }\n            except Exception as e:\n                test_results['tests']['batch_prediction'] = {\n                    'success': False,\n                    'error': str(e)\n                }\n            \n            # 4. ストレステスト\n            logger.info(\"\\n4. Running stress test...\")\n            try:\n                stress_result = self.stress_test(n_requests=20, interval=0.5)\n                test_results['tests']['stress_test'] = {\n                    'success': True,\n                    'success_rate': stress_result['successful_requests'] / stress_result['total_requests'],\n                    'avg_response_time': stress_result.get('avg_response_time', 0)\n                }\n            except Exception as e:\n                test_results['tests']['stress_test'] = {\n                    'success': False,\n                    'error': str(e)\n                }\n            \n            # 5. Data Capture確認\n            if bucket and s3_prefix:\n                logger.info(\"\\n5. Checking data capture...\")\n                try:\n                    capture_files = self.check_data_capture(bucket, s3_prefix)\n                    test_results['tests']['data_capture'] = {\n                        'success': True,\n                        'file_count': len(capture_files)\n                    }\n                except Exception as e:\n                    test_results['tests']['data_capture'] = {\n                        'success': False,\n                        'error': str(e)\n                    }\n            \n            # テスト結果サマリー\n            successful_tests = sum(1 for test in test_results['tests'].values() if test['success'])\n            total_tests = len(test_results['tests'])\n            \n            logger.info(\"\\n\" + \"=\"*60)\n            logger.info(\"🧪 COMPREHENSIVE TEST RESULTS\")\n            logger.info(\"=\"*60)\n            logger.info(f\"Endpoint: {self.endpoint_name}\")\n            logger.info(f\"Success Rate: {successful_tests}/{total_tests} tests passed\")\n            \n            for test_name, result in test_results['tests'].items():\n                status = \"✅ PASS\" if result['success'] else \"❌ FAIL\"\n                logger.info(f\"  {test_name}: {status}\")\n            \n            logger.info(\"=\"*60)\n            \n            return test_results\n            \n        except Exception as e:\n            logger.error(f\"❌ Comprehensive test failed: {str(e)}\")\n            test_results['error'] = str(e)\n            return test_results


def main():
    \"\"\"\n    メイン実行関数\n    \"\"\"\n    # 設定（実際の値に変更してください）\n    endpoint_name = \"lightgbm-binary-endpoint-v2-1722697200\"  # 実際のエンドポイント名\n    bucket = \"sagemaker-us-east-1-123456789012\"  # 実際のバケット名\n    s3_prefix = \"lightgbm-binary-classification-v2\"  # 実際のS3プレフィックス\n    region = \"us-east-1\"  # 実際のリージョン\n    \n    # テスター初期化\n    tester = LightGBMV2Tester(endpoint_name, region)\n    \n    # 包括的テスト実行\n    results = tester.run_comprehensive_test(bucket, s3_prefix)\n    \n    # 結果をJSONファイルに保存\n    with open('test_results_v2.json', 'w') as f:\n        json.dump(results, f, indent=2, default=str)\n    \n    logger.info(\"Test results saved to test_results_v2.json\")\n    \n    return results


if __name__ == \"__main__\":\n    results = main()
