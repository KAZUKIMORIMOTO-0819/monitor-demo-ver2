#!/usr/bin/env python3
"""
LightGBM Binary Classifier v2 簡単実行スクリプト
ワンクリックで全体のパイプラインを実行します。
"""

import sys
import os
import logging
from datetime import datetime

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    メイン実行関数
    """
    print("="*80)
    print("🚀 LightGBM Binary Classifier v2 - Quick Start")
    print("="*80)
    print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # メインクラスをインポート
        from lightgbm_binary_classifier_v2 import LightGBMBinaryClassifierV2
        
        # カスタム設定
        custom_config = {
            # データ設定（小さめに設定してテスト時間短縮）
            'n_samples': 2000,
            'n_features': 20,
            'n_informative': 15,
            
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
            'num_boost_round': 50,  # 短縮
            'early_stopping_rounds': 10,
            
            # モニタリング設定
            'monitoring_cron': 'cron(0/15 * ? * * *)',  # 15分毎
        }
        
        print("📋 設定:")
        print(f"  - データサンプル数: {custom_config['n_samples']}")
        print(f"  - 特徴量数: {custom_config['n_features']}")
        print(f"  - 訓練ラウンド数: {custom_config['num_boost_round']}")
        print(f"  - モニタリング頻度: 15分毎")
        print()
        
        # 実行確認
        response = input("実行を開始しますか？ (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("実行をキャンセルしました。")
            return
        
        print("\n🔄 パイプライン実行開始...")
        
        # パイプライン実行
        classifier = LightGBMBinaryClassifierV2(config=custom_config)
        result = classifier.run_complete_pipeline()
        
        print("\n" + "="*80)
        print("🎉 パイプライン実行完了！")
        print("="*80)
        print(f"エンドポイント名: {result['endpoint_name']}")
        print(f"モニタースケジュール名: {result['monitor_schedule_name']}")
        print(f"S3バケット: {result['bucket']}")
        print(f"S3プレフィックス: {result['s3_prefix']}")
        print()
        
        # 次のステップの案内
        print("📋 次のステップ:")
        print("1. エンドポイントテスト:")
        print(f"   python test_lightgbm_v2.py")
        print()
        print("2. モニタリング状態確認:")
        print(f"   python model_monitor_utils_v2.py")
        print()
        print("3. クリーンアップ（課金停止）:")
        print("   classifier.cleanup()  # Pythonで実行")
        print()
        
        # 重要な注意事項
        print("⚠️  重要な注意事項:")
        print("- エンドポイントは継続的に課金されます")
        print("- Model Monitorも定期実行で課金されます")
        print("- 使用後は必ずクリーンアップを実行してください")
        print()
        
        # クリーンアップ確認
        cleanup_response = input("今すぐクリーンアップを実行しますか？ (y/N): ")
        if cleanup_response.lower() in ['y', 'yes']:
            print("\n🧹 クリーンアップ実行中...")
            classifier.cleanup()
            print("✅ クリーンアップ完了")
        else:
            print("\n⚠️  後でクリーンアップを忘れずに実行してください！")
        
        print(f"\n完了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        return classifier
        
    except ImportError as e:
        print(f"❌ インポートエラー: {str(e)}")
        print("必要なライブラリがインストールされていない可能性があります。")
        print("以下のコマンドを実行してください:")
        print("pip install lightgbm sagemaker boto3 pandas numpy scikit-learn")
        return None
        
    except Exception as e:
        print(f"❌ 実行エラー: {str(e)}")
        print("詳細なエラー情報については、ログを確認してください。")
        return None

if __name__ == "__main__":
    classifier = main()
