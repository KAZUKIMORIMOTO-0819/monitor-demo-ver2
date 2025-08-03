#!/usr/bin/env python3
"""
Model Monitor管理ユーティリティ v2
Model Monitorの状態確認、レポート分析、管理機能を提供します。
"""

import boto3
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelMonitorManagerV2:
    """
    Model Monitor管理クラス v2
    """
    
    def __init__(self, region='us-east-1'):
        """
        初期化
        
        Args:
            region (str): AWSリージョン
        """
        self.region = region
        self.sm_client = boto3.client('sagemaker', region_name=region)
        self.s3_client = boto3.client('s3', region_name=region)
        self.cloudwatch = boto3.client('cloudwatch', region_name=region)
        
        logger.info(f"Model Monitor Manager v2 initialized for region: {region}")
    
    def list_monitoring_schedules(self):
        """
        モニタリングスケジュール一覧を取得
        
        Returns:
            list: モニタリングスケジュール一覧
        """
        logger.info("Listing monitoring schedules...")
        
        try:
            response = self.sm_client.list_monitoring_schedules()
            schedules = response.get('MonitoringScheduleSummaries', [])
            
            logger.info(f"Found {len(schedules)} monitoring schedules")
            
            for schedule in schedules:
                logger.info(f"  - {schedule['MonitoringScheduleName']}: {schedule['MonitoringScheduleStatus']}")
            
            return schedules
            
        except Exception as e:
            logger.error(f"Failed to list monitoring schedules: {str(e)}")
            return []
    
    def get_schedule_details(self, schedule_name):
        """
        モニタリングスケジュールの詳細を取得
        
        Args:
            schedule_name (str): スケジュール名
            
        Returns:
            dict: スケジュール詳細
        """
        logger.info(f"Getting details for schedule: {schedule_name}")
        
        try:
            response = self.sm_client.describe_monitoring_schedule(
                MonitoringScheduleName=schedule_name
            )
            
            logger.info(f"Schedule status: {response['MonitoringScheduleStatus']}")
            logger.info(f"Endpoint: {response['MonitoringScheduleConfig']['MonitoringJobDefinition']['MonitoringInputs'][0]['EndpointInput']['EndpointName']}")
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to get schedule details: {str(e)}")
            return None
    
    def list_executions(self, schedule_name, max_results=10):
        """
        モニタリング実行履歴を取得
        
        Args:
            schedule_name (str): スケジュール名
            max_results (int): 最大取得数
            
        Returns:
            list: 実行履歴
        """
        logger.info(f"Listing executions for schedule: {schedule_name}")
        
        try:
            response = self.sm_client.list_monitoring_executions(
                MonitoringScheduleName=schedule_name,
                MaxResults=max_results,
                SortBy='CreationTime',
                SortOrder='Descending'
            )
            
            executions = response.get('MonitoringExecutionSummaries', [])
            
            logger.info(f"Found {len(executions)} executions")
            
            for execution in executions:
                logger.info(f"  - {execution['ProcessingJobName']}: {execution['ProcessingJobStatus']} ({execution['CreationTime']})")
            
            return executions
            
        except Exception as e:
            logger.error(f"Failed to list executions: {str(e)}")
            return []
    
    def get_execution_details(self, execution_name):
        """
        実行詳細を取得
        
        Args:
            execution_name (str): 実行名
            
        Returns:
            dict: 実行詳細
        """
        logger.info(f"Getting execution details: {execution_name}")
        
        try:\n            response = self.sm_client.describe_processing_job(\n                ProcessingJobName=execution_name\n            )\n            \n            logger.info(f\"Execution status: {response['ProcessingJobStatus']}\")\n            \n            if 'FailureReason' in response:\n                logger.warning(f\"Failure reason: {response['FailureReason']}\")\n            \n            return response\n            \n        except Exception as e:\n            logger.error(f\"Failed to get execution details: {str(e)}\")\n            return None\n    \n    def download_monitoring_report(self, bucket, report_s3_key, local_path=None):\n        \"\"\"\n        モニタリングレポートをダウンロード\n        \n        Args:\n            bucket (str): S3バケット名\n            report_s3_key (str): レポートのS3キー\n            local_path (str): ローカル保存パス\n            \n        Returns:\n            str: ダウンロードしたファイルパス\n        \"\"\"\n        if local_path is None:\n            local_path = f\"monitoring_report_{int(time.time())}.json\"\n        \n        logger.info(f\"Downloading monitoring report: {report_s3_key}\")\n        \n        try:\n            self.s3_client.download_file(bucket, report_s3_key, local_path)\n            logger.info(f\"Report downloaded to: {local_path}\")\n            return local_path\n            \n        except Exception as e:\n            logger.error(f\"Failed to download report: {str(e)}\")\n            return None\n    \n    def analyze_monitoring_report(self, report_path):\n        \"\"\"\n        モニタリングレポートを分析\n        \n        Args:\n            report_path (str): レポートファイルパス\n            \n        Returns:\n            dict: 分析結果\n        \"\"\"\n        logger.info(f\"Analyzing monitoring report: {report_path}\")\n        \n        try:\n            with open(report_path, 'r') as f:\n                report = json.load(f)\n            \n            analysis = {\n                'timestamp': datetime.now().isoformat(),\n                'report_path': report_path,\n                'violations': [],\n                'statistics': {},\n                'summary': {}\n            }\n            \n            # 制約違反の確認\n            if 'violations' in report:\n                violations = report['violations']\n                analysis['violations'] = violations\n                logger.info(f\"Found {len(violations)} constraint violations\")\n                \n                for violation in violations:\n                    logger.warning(f\"Violation: {violation.get('feature_name', 'Unknown')} - {violation.get('constraint_check_type', 'Unknown')}\")\n            \n            # 統計情報の抽出\n            if 'dataset' in report and 'item' in report['dataset']:\n                for item in report['dataset']['item']:\n                    feature_name = item.get('name', 'Unknown')\n                    if 'numerical_statistics' in item:\n                        stats = item['numerical_statistics']\n                        analysis['statistics'][feature_name] = {\n                            'mean': stats.get('mean'),\n                            'stddev': stats.get('stddev'),\n                            'min': stats.get('min'),\n                            'max': stats.get('max')\n                        }\n            \n            # サマリー作成\n            analysis['summary'] = {\n                'total_features': len(analysis['statistics']),\n                'total_violations': len(analysis['violations']),\n                'has_violations': len(analysis['violations']) > 0\n            }\n            \n            logger.info(f\"Analysis completed: {analysis['summary']['total_features']} features, {analysis['summary']['total_violations']} violations\")\n            \n            return analysis\n            \n        except Exception as e:\n            logger.error(f\"Failed to analyze report: {str(e)}\")\n            return None\n    \n    def get_cloudwatch_metrics(self, endpoint_name, start_time=None, end_time=None):\n        \"\"\"\n        CloudWatchメトリクスを取得\n        \n        Args:\n            endpoint_name (str): エンドポイント名\n            start_time (datetime): 開始時刻\n            end_time (datetime): 終了時刻\n            \n        Returns:\n            dict: メトリクス情報\n        \"\"\"\n        if end_time is None:\n            end_time = datetime.utcnow()\n        if start_time is None:\n            start_time = end_time - timedelta(hours=24)\n        \n        logger.info(f\"Getting CloudWatch metrics for endpoint: {endpoint_name}\")\n        \n        metrics = {}\n        \n        # 取得するメトリクス一覧\n        metric_names = [\n            'Invocations',\n            'Errors',\n            'ModelLatency',\n            'OverheadLatency'\n        ]\n        \n        try:\n            for metric_name in metric_names:\n                response = self.cloudwatch.get_metric_statistics(\n                    Namespace='AWS/SageMaker',\n                    MetricName=metric_name,\n                    Dimensions=[\n                        {\n                            'Name': 'EndpointName',\n                            'Value': endpoint_name\n                        },\n                        {\n                            'Name': 'VariantName',\n                            'Value': 'AllTraffic'\n                        }\n                    ],\n                    StartTime=start_time,\n                    EndTime=end_time,\n                    Period=3600,  # 1時間\n                    Statistics=['Sum', 'Average', 'Maximum']\n                )\n                \n                metrics[metric_name] = response['Datapoints']\n                logger.info(f\"Retrieved {len(response['Datapoints'])} datapoints for {metric_name}\")\n            \n            return metrics\n            \n        except Exception as e:\n            logger.error(f\"Failed to get CloudWatch metrics: {str(e)}\")\n            return {}\n    \n    def check_monitoring_health(self, schedule_name):\n        \"\"\"\n        モニタリングの健全性をチェック\n        \n        Args:\n            schedule_name (str): スケジュール名\n            \n        Returns:\n            dict: 健全性チェック結果\n        \"\"\"\n        logger.info(f\"Checking monitoring health for: {schedule_name}\")\n        \n        health_check = {\n            'schedule_name': schedule_name,\n            'timestamp': datetime.now().isoformat(),\n            'status': 'Unknown',\n            'issues': [],\n            'recommendations': []\n        }\n        \n        try:\n            # スケジュール詳細取得\n            schedule_details = self.get_schedule_details(schedule_name)\n            if not schedule_details:\n                health_check['status'] = 'Error'\n                health_check['issues'].append('Failed to get schedule details')\n                return health_check\n            \n            # スケジュール状態確認\n            schedule_status = schedule_details['MonitoringScheduleStatus']\n            if schedule_status != 'Scheduled':\n                health_check['issues'].append(f'Schedule status is {schedule_status}, expected Scheduled')\n            \n            # 最近の実行確認\n            executions = self.list_executions(schedule_name, max_results=5)\n            if not executions:\n                health_check['issues'].append('No recent executions found')\n                health_check['recommendations'].append('Check if the schedule is running and data capture is enabled')\n            else:\n                # 最新実行の状態確認\n                latest_execution = executions[0]\n                if latest_execution['ProcessingJobStatus'] == 'Failed':\n                    health_check['issues'].append('Latest execution failed')\n                    health_check['recommendations'].append('Check execution logs for error details')\n                \n                # 実行頻度確認\n                if len(executions) >= 2:\n                    latest_time = executions[0]['CreationTime']\n                    previous_time = executions[1]['CreationTime']\n                    time_diff = latest_time - previous_time\n                    \n                    if time_diff > timedelta(hours=2):\n                        health_check['issues'].append('Long gap between executions detected')\n                        health_check['recommendations'].append('Check if schedule frequency is appropriate')\n            \n            # 健全性ステータス決定\n            if not health_check['issues']:\n                health_check['status'] = 'Healthy'\n            elif len(health_check['issues']) <= 2:\n                health_check['status'] = 'Warning'\n            else:\n                health_check['status'] = 'Critical'\n            \n            logger.info(f\"Health check completed: {health_check['status']}\")\n            \n            return health_check\n            \n        except Exception as e:\n            logger.error(f\"Health check failed: {str(e)}\")\n            health_check['status'] = 'Error'\n            health_check['issues'].append(f'Health check error: {str(e)}')\n            return health_check\n    \n    def generate_monitoring_dashboard(self, schedule_name, bucket, s3_prefix):\n        \"\"\"\n        モニタリングダッシュボード情報を生成\n        \n        Args:\n            schedule_name (str): スケジュール名\n            bucket (str): S3バケット名\n            s3_prefix (str): S3プレフィックス\n            \n        Returns:\n            dict: ダッシュボード情報\n        \"\"\"\n        logger.info(f\"Generating monitoring dashboard for: {schedule_name}\")\n        \n        dashboard = {\n            'schedule_name': schedule_name,\n            'generated_at': datetime.now().isoformat(),\n            'schedule_info': {},\n            'recent_executions': [],\n            'health_status': {},\n            'monitoring_reports': [],\n            'recommendations': []\n        }\n        \n        try:\n            # スケジュール情報\n            dashboard['schedule_info'] = self.get_schedule_details(schedule_name)\n            \n            # 最近の実行\n            dashboard['recent_executions'] = self.list_executions(schedule_name, max_results=10)\n            \n            # 健全性チェック\n            dashboard['health_status'] = self.check_monitoring_health(schedule_name)\n            \n            # モニタリングレポート一覧\n            try:\n                response = self.s3_client.list_objects_v2(\n                    Bucket=bucket,\n                    Prefix=f'{s3_prefix}/monitoring-reports/output'\n                )\n                \n                if 'Contents' in response:\n                    reports = response['Contents']\n                    dashboard['monitoring_reports'] = [\n                        {\n                            'key': report['Key'],\n                            'size': report['Size'],\n                            'last_modified': report['LastModified'].isoformat()\n                        }\n                        for report in sorted(reports, key=lambda x: x['LastModified'], reverse=True)[:10]\n                    ]\n            except Exception as e:\n                logger.warning(f\"Failed to list monitoring reports: {str(e)}\")\n            \n            # 推奨事項生成\n            if dashboard['health_status']['status'] != 'Healthy':\n                dashboard['recommendations'].extend(dashboard['health_status']['recommendations'])\n            \n            if not dashboard['monitoring_reports']:\n                dashboard['recommendations'].append('No monitoring reports found - check if data capture is working')\n            \n            logger.info(\"Dashboard generated successfully\")\n            \n            return dashboard\n            \n        except Exception as e:\n            logger.error(f\"Failed to generate dashboard: {str(e)}\")\n            dashboard['error'] = str(e)\n            return dashboard\n    \n    def cleanup_old_executions(self, schedule_name, keep_count=10):\n        \"\"\"\n        古い実行を削除（実際にはログ出力のみ）\n        \n        Args:\n            schedule_name (str): スケジュール名\n            keep_count (int): 保持する実行数\n            \n        Returns:\n            list: 削除対象の実行一覧\n        \"\"\"\n        logger.info(f\"Identifying old executions for cleanup: {schedule_name}\")\n        \n        try:\n            executions = self.list_executions(schedule_name, max_results=50)\n            \n            if len(executions) <= keep_count:\n                logger.info(f\"No cleanup needed. Found {len(executions)} executions, keeping {keep_count}\")\n                return []\n            \n            # 削除対象の実行\n            executions_to_cleanup = executions[keep_count:]\n            \n            logger.info(f\"Found {len(executions_to_cleanup)} executions for potential cleanup\")\n            \n            for execution in executions_to_cleanup:\n                logger.info(f\"  - {execution['ProcessingJobName']} ({execution['CreationTime']})\")\n            \n            # 注意: 実際の削除は手動で行う必要があります\n            logger.warning(\"Note: Actual cleanup must be performed manually through AWS Console or CLI\")\n            \n            return executions_to_cleanup\n            \n        except Exception as e:\n            logger.error(f\"Failed to identify cleanup targets: {str(e)}\")\n            return []


def main():\n    \"\"\"\n    メイン実行関数\n    \"\"\"\n    # 設定（実際の値に変更してください）\n    schedule_name = \"lightgbm-monitor-v2-1722697200\"  # 実際のスケジュール名\n    bucket = \"sagemaker-us-east-1-123456789012\"  # 実際のバケット名\n    s3_prefix = \"lightgbm-binary-classification-v2\"  # 実際のS3プレフィックス\n    region = \"us-east-1\"  # 実際のリージョン\n    \n    # マネージャー初期化\n    manager = ModelMonitorManagerV2(region)\n    \n    # 包括的な状態確認\n    logger.info(\"\\n\" + \"=\"*60)\n    logger.info(\"🔍 MODEL MONITOR COMPREHENSIVE CHECK\")\n    logger.info(\"=\"*60)\n    \n    # 1. スケジュール一覧\n    logger.info(\"\\n1. Listing all monitoring schedules...\")\n    schedules = manager.list_monitoring_schedules()\n    \n    # 2. 特定スケジュールの詳細\n    if schedule_name:\n        logger.info(f\"\\n2. Getting details for schedule: {schedule_name}\")\n        schedule_details = manager.get_schedule_details(schedule_name)\n        \n        # 3. 実行履歴\n        logger.info(f\"\\n3. Getting execution history...\")\n        executions = manager.list_executions(schedule_name)\n        \n        # 4. 健全性チェック\n        logger.info(f\"\\n4. Performing health check...\")\n        health_status = manager.check_monitoring_health(schedule_name)\n        \n        # 5. ダッシュボード生成\n        logger.info(f\"\\n5. Generating dashboard...\")\n        dashboard = manager.generate_monitoring_dashboard(schedule_name, bucket, s3_prefix)\n        \n        # 結果をJSONファイルに保存\n        results = {\n            'timestamp': datetime.now().isoformat(),\n            'schedule_name': schedule_name,\n            'schedule_details': schedule_details,\n            'executions': executions,\n            'health_status': health_status,\n            'dashboard': dashboard\n        }\n        \n        with open('monitoring_status_v2.json', 'w') as f:\n            json.dump(results, f, indent=2, default=str)\n        \n        logger.info(\"\\nResults saved to monitoring_status_v2.json\")\n        \n        # サマリー表示\n        logger.info(\"\\n\" + \"=\"*60)\n        logger.info(\"📊 MONITORING STATUS SUMMARY\")\n        logger.info(\"=\"*60)\n        logger.info(f\"Schedule: {schedule_name}\")\n        logger.info(f\"Health Status: {health_status['status']}\")\n        logger.info(f\"Recent Executions: {len(executions)}\")\n        logger.info(f\"Issues Found: {len(health_status['issues'])}\")\n        \n        if health_status['issues']:\n            logger.info(\"\\n⚠️  Issues:\")\n            for issue in health_status['issues']:\n                logger.info(f\"  - {issue}\")\n        \n        if health_status['recommendations']:\n            logger.info(\"\\n💡 Recommendations:\")\n            for rec in health_status['recommendations']:\n                logger.info(f\"  - {rec}\")\n        \n        logger.info(\"=\"*60)\n        \n        return results\n    \n    return {'schedules': schedules}


if __name__ == \"__main__\":\n    results = main()
