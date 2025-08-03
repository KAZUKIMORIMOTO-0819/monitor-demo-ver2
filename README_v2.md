# LightGBM Binary Classification + SageMaker Model Monitor v2

このプロジェクトでは、LightGBMを使った2値分類モデルをSageMakerにデプロイし、Model Monitorで監視する完全なワークフローを提供します。v2では、より堅牢で実用的な機能を追加しました。

## 🆕 v2の新機能・改良点

### 主な改良点
- **堅牢なエラーハンドリング**: 各ステップでの詳細なエラー処理
- **詳細なログ出力**: 実行状況の詳細な追跡
- **設定可能なパラメータ**: 柔軟な設定変更
- **自動クリーンアップ機能**: リソースの自動削除
- **包括的なテスト機能**: エンドポイントとモニタリングの総合テスト
- **モニタリング管理ツール**: Model Monitorの状態管理・分析ツール

### 新しいファイル構成
```
monitor-demo-ver2/
├── lightgbm_binary_classifier_v2.py     # メイン実装（Pythonスクリプト版）
├── lightgbm_complete_v2.ipynb           # 完全統合版notebook（前半）
├── lightgbm_deploy_monitor_v2.ipynb     # デプロイ・モニタリング版notebook（後半）
├── test_lightgbm_v2.py                  # 包括的テストスクリプト
├── model_monitor_utils_v2.py            # Model Monitor管理ユーティリティ
├── README_v2.md                         # このファイル
└── [既存ファイル]                        # v1からコピーされたファイル
```

## 🚀 使用方法

### 方法1: Pythonスクリプト実行（推奨）

```bash
# メインスクリプト実行
python lightgbm_binary_classifier_v2.py
```

### 方法2: Jupyter Notebook実行

1. **前半**: `lightgbm_complete_v2.ipynb`
   - 環境設定
   - データ生成
   - モデル訓練
   - モデル保存

2. **後半**: `lightgbm_deploy_monitor_v2.ipynb`
   - エンドポイントデプロイ
   - Model Monitor設定
   - テスト実行

## 🧪 テスト機能

### エンドポイントテスト
```bash
# 包括的テスト実行
python test_lightgbm_v2.py
```

テスト内容：
- エンドポイント状態確認
- 単一予測テスト
- バッチ予測テスト
- ストレステスト（連続リクエスト）
- Data Capture確認

### Model Monitor管理
```bash
# モニタリング状態確認
python model_monitor_utils_v2.py
```

管理機能：
- モニタリングスケジュール一覧
- 実行履歴確認
- 健全性チェック
- CloudWatchメトリクス取得
- レポート分析

## 📊 主な機能詳細

### 1. データ生成・モデル訓練
- **データサイズ**: 3,000サンプル（デフォルト）
- **特徴量数**: 25次元（デフォルト）
- **分割比率**: 訓練70% / 検証10% / テスト20%
- **LightGBMパラメータ**: 最適化済み設定
- **早期停止**: 過学習防止

### 2. エンドポイントデプロイ
- **インスタンス**: ml.m5.large（デフォルト）
- **Data Capture**: 100%サンプリング
- **推論形式**: JSON入出力
- **エラーハンドリング**: 堅牢な例外処理

### 3. Model Monitor
- **ベースライン**: 自動生成
- **監視頻度**: 10分毎（デフォルト）
- **インスタンス**: ml.m5.xlarge
- **CloudWatch**: メトリクス自動送信
- **レポート**: S3自動保存

### 4. テスト・監視
- **自動テスト**: エンドポイント動作確認
- **ストレステスト**: 負荷テスト
- **健全性チェック**: モニタリング状態確認
- **ダッシュボード**: 統合状態表示

## ⚙️ 設定パラメータ

### データ・モデル設定
```python
config = {
    'n_samples': 3000,           # データサンプル数
    'n_features': 25,            # 特徴量数
    'n_informative': 20,         # 有効特徴量数
    'test_size': 0.2,            # テストデータ割合
    'val_size': 0.1,             # 検証データ割合
    
    'lgb_params': {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_leaves': 31,
        'learning_rate': 0.05,
        # ... その他のパラメータ
    }
}
```

### デプロイ設定
```python
config = {
    'instance_type': 'ml.m5.large',      # エンドポイントインスタンス
    'initial_instance_count': 1,         # インスタンス数
    'enable_data_capture': True,         # Data Capture有効化
    'sampling_percentage': 100,          # キャプチャ率
}
```

### モニタリング設定
```python
config = {
    'monitoring_instance_type': 'ml.m5.xlarge',  # モニタリングインスタンス
    'monitoring_cron': 'cron(0/10 * ? * * *)',   # 10分毎実行
    'max_runtime_seconds': 3600,                 # 最大実行時間
}
```

## 📈 生成されるアーティファクト

### S3構造
```
s3://your-bucket/lightgbm-binary-classification-v2/
├── model/
│   └── model.tar.gz                    # 訓練済みモデル
├── baseline/
│   ├── baseline.csv                    # ベースラインデータ
│   └── output/
│       ├── statistics.json             # ベースライン統計
│       └── constraints.json            # データ制約
├── datacapture/                        # データキャプチャファイル
│   └── AllTraffic/
│       └── [timestamp]/
└── monitoring-reports/                 # モニタリングレポート
    └── output/
        ├── statistics.json
        ├── constraint_violations.json
        └── ...
```

### ローカルファイル
- `model/model.pkl` - Pickleモデルファイル
- `model/model.txt` - LightGBMテキストモデル
- `source/inference.py` - 推論スクリプト
- `baseline_data/baseline.csv` - ベースラインデータ
- `test_results_v2.json` - テスト結果
- `monitoring_status_v2.json` - モニタリング状態

## 🔍 モニタリング・分析

### CloudWatchメトリクス
- `Invocations` - 呼び出し回数
- `Errors` - エラー回数
- `ModelLatency` - モデル推論時間
- `OverheadLatency` - オーバーヘッド時間

### Model Monitorメトリクス
- `feature_baseline_drift_*` - 特徴量ドリフト
- `missing_column_*` - 欠損列検出
- `extra_column_*` - 追加列検出
- `data_type_check_*` - データ型チェック

### レポート分析
```python
# レポート分析例
from model_monitor_utils_v2 import ModelMonitorManagerV2

manager = ModelMonitorManagerV2()
analysis = manager.analyze_monitoring_report('report.json')
print(f"Violations: {len(analysis['violations'])}")
```

## 🧹 クリーンアップ

### 自動クリーンアップ
```python
# Pythonスクリプト内で
classifier = LightGBMBinaryClassifierV2()
classifier.cleanup()
```

### 手動クリーンアップ
```python
# Notebook内で
cleanup()  # 関数を実行
```

### AWS CLIでのクリーンアップ
```bash
# エンドポイント削除
aws sagemaker delete-endpoint --endpoint-name your-endpoint-name

# モニタリングスケジュール削除
aws sagemaker stop-monitoring-schedule --monitoring-schedule-name your-schedule-name
aws sagemaker delete-monitoring-schedule --monitoring-schedule-name your-schedule-name
```

## 🚨 注意事項・ベストプラクティス

### 課金について
- **エンドポイント**: 継続的に課金（ml.m5.large: ~$0.10/時間）
- **Model Monitor**: 実行毎に課金（ml.m5.xlarge: ~$0.20/時間）
- **S3ストレージ**: データ量に応じて課金
- **CloudWatch**: メトリクス・ログに応じて課金

### セキュリティ
- **IAMロール**: 最小権限の原則
- **VPC**: 必要に応じてプライベートサブネット使用
- **暗号化**: S3・EBS暗号化の有効化

### パフォーマンス
- **インスタンスタイプ**: ワークロードに応じて選択
- **Auto Scaling**: 必要に応じて設定
- **バッチ推論**: 大量データ処理時は検討

### モニタリング
- **頻度調整**: 本番環境では適切な頻度に設定
- **アラート設定**: CloudWatchアラームの設定
- **ログ監視**: CloudWatch Logsの定期確認

## 🔧 トラブルシューティング

### よくある問題

#### 1. エンドポイントデプロイ失敗
```
原因: IAMロール権限不足、推論スクリプトエラー
解決: ロール権限確認、ログ確認
```

#### 2. モニタリングレポート未生成
```
原因: Data Capture無効、トラフィック不足
解決: Data Capture設定確認、テストトラフィック送信
```

#### 3. 予測エラー
```
原因: 入力データ形式不正、モデルロードエラー
解決: 入力形式確認、推論スクリプト確認
```

### ログ確認方法
```python
# CloudWatch Logsでログ確認
import boto3

logs_client = boto3.client('logs')
log_groups = [
    '/aws/sagemaker/Endpoints/your-endpoint-name',
    '/aws/sagemaker/ProcessingJobs'
]
```

### デバッグ手順
1. **エンドポイント状態確認**
2. **CloudWatchログ確認**
3. **テストリクエスト送信**
4. **Data Capture確認**
5. **モニタリング実行確認**

## 📚 参考資料

### AWS公式ドキュメント
- [Amazon SageMaker Model Monitor](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor.html)
- [SageMaker Endpoints](https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html)
- [Data Capture](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-data-capture.html)

### ライブラリドキュメント
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [SageMaker Python SDK](https://sagemaker.readthedocs.io/)
- [Boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)

### ベストプラクティス
- [SageMaker Best Practices](https://docs.aws.amazon.com/sagemaker/latest/dg/best-practices.html)
- [MLOps with SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-projects.html)

## 🤝 サポート・貢献

### 問題報告
- GitHub Issues
- AWS Support（有料プラン）

### 機能要望
- プルリクエスト歓迎
- 機能提案はIssuesで

---

**作成日**: 2025-08-03  
**バージョン**: 2.0  
**対応環境**: SageMaker Studio, SageMaker Notebook Instance  
**Python**: 3.8+  
**SageMaker SDK**: 2.100.0+

## 📋 チェックリスト

### デプロイ前
- [ ] IAMロール権限確認
- [ ] S3バケット作成・権限設定
- [ ] 設定パラメータ確認
- [ ] 必要なライブラリインストール

### デプロイ後
- [ ] エンドポイント動作確認
- [ ] Data Capture動作確認
- [ ] モニタリングスケジュール確認
- [ ] テストトラフィック送信

### 運用中
- [ ] 定期的な健全性チェック
- [ ] モニタリングレポート確認
- [ ] CloudWatchメトリクス監視
- [ ] コスト監視

### クリーンアップ
- [ ] エンドポイント削除
- [ ] モニタリングスケジュール削除
- [ ] S3データ削除（必要に応じて）
- [ ] CloudWatchログ削除（必要に応じて）
