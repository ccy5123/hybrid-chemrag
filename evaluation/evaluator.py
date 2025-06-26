# evaluation/evaluator.py
"""
고급 평가 모듈
"""
import numpy as np
import math
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

class HybridEvaluator:
    """하이브리드 시스템 전용 평가자"""
    
    def __init__(self, predictor):
        self.predictor = predictor
    
    def evaluate_test_set(self, test_data: List[Dict]) -> Dict:
        """하이브리드 시스템으로 테스트 세트 평가"""
        predictions = []
        actuals = []
        explanations = []
        times = []
        errors = []
        metadata_list = []
        
        logger.info(f"🔬 Starting hybrid evaluation on {len(test_data)} test samples...")
        
        for i, item in enumerate(test_data):
            input_text = item['input_text']
            actual = int(item['output_text'])
            
            try:
                pred, explanation, elapsed_time, metadata = self.predictor.predict_single(input_text)
                
                predictions.append(pred)
                actuals.append(actual)
                explanations.append(explanation)
                times.append(elapsed_time)
                errors.append(None)
                metadata_list.append(metadata)
                
                # 실시간 성능 모니터링
                if i > 0:
                    current_mae = np.mean([abs(a - p) for a, p in zip(actuals, predictions)])
                    logger.info(f"Sample {i+1}/{len(test_data)} | Pred: {pred} | Actual: {actual} | "
                              f"Weights: A={metadata['weights']['assay']:.2f}/C={metadata['weights']['chemical']:.2f} | "
                              f"Running MAE: {current_mae:.2f}")
                else:
                    logger.info(f"Sample {i+1}/{len(test_data)} | Pred: {pred} | Actual: {actual}")
                    
            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                predictions.append(50)
                actuals.append(actual)
                explanations.append(f"Error: {str(e)}")
                times.append(0.0)
                errors.append(str(e))
                metadata_list.append({})
        
        # 기본 성능 메트릭
        metrics = self._calculate_basic_metrics(actuals, predictions)
        
        # 하이브리드 특화 메트릭
        hybrid_metrics = self._calculate_hybrid_metrics(actuals, predictions, metadata_list)
        metrics.update(hybrid_metrics)
        
        # 메타데이터 통계
        metadata_stats = self._calculate_metadata_stats(metadata_list)
        metrics.update(metadata_stats)
        
        results = {
            'predictions': predictions,
            'actuals': actuals,
            'explanations': explanations,
            'times': times,
            'errors': errors,
            'metadata': metadata_list,
            'metrics': metrics
        }
        
        return results
    
    def _calculate_basic_metrics(self, actuals: List[int], predictions: List[int]) -> Dict:
        """기본 성능 메트릭 계산"""
        mae = np.mean([abs(a - p) for a, p in zip(actuals, predictions)])
        mse = np.mean([(a - p)**2 for a, p in zip(actuals, predictions)])
        rmse = math.sqrt(mse)
        
        # R² 계산
        y_mean = np.mean(actuals)
        ss_tot = sum([(y - y_mean)**2 for y in actuals])
        ss_res = sum([(a - p)**2 for a, p in zip(actuals, predictions)])
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        
        within_10 = sum([1 for a, p in zip(actuals, predictions) if abs(a - p) <= 10]) / len(actuals) * 100
        within_20 = sum([1 for a, p in zip(actuals, predictions) if abs(a - p) <= 20]) / len(actuals) * 100
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'within_10_pct': within_10,
            'within_20_pct': within_20,
            'n_samples': len(actuals),
            'n_errors': len([p for p in predictions if p == 50])  # 기본값은 에러로 간주
        }
    
    def _calculate_hybrid_metrics(self, actuals: List[int], predictions: List[int], 
                                 metadata_list: List[Dict]) -> Dict:
        """하이브리드 특화 메트릭 계산"""
        
        # 가중치별 샘플 분류
        assay_weighted_samples = [i for i, meta in enumerate(metadata_list) 
                                 if meta.get('weights', {}).get('assay', 0) > 0.6]
        chemical_weighted_samples = [i for i, meta in enumerate(metadata_list) 
                                   if meta.get('weights', {}).get('chemical', 0) > 0.6]
        balanced_samples = [i for i, meta in enumerate(metadata_list) 
                           if 0.4 <= meta.get('weights', {}).get('assay', 0.5) <= 0.6]
        
        # 가중치별 성능 분석
        assay_mae = np.mean([abs(actuals[i] - predictions[i]) for i in assay_weighted_samples]) if assay_weighted_samples else float('inf')
        chemical_mae = np.mean([abs(actuals[i] - predictions[i]) for i in chemical_weighted_samples]) if chemical_weighted_samples else float('inf')
        balanced_mae = np.mean([abs(actuals[i] - predictions[i]) for i in balanced_samples]) if balanced_samples else float('inf')
        
        # 유사도 기반 분석
        high_assay_sim_samples = [i for i, meta in enumerate(metadata_list) 
                                 if meta.get('max_assay_similarity', 0) > 0.7]
        high_chem_sim_samples = [i for i, meta in enumerate(metadata_list) 
                                if meta.get('max_chemical_similarity', 0) > 0.7]
        
        return {
            'assay_weighted_mae': assay_mae,
            'chemical_weighted_mae': chemical_mae,
            'balanced_mae': balanced_mae,
            'n_assay_weighted': len(assay_weighted_samples),
            'n_chemical_weighted': len(chemical_weighted_samples),
            'n_balanced': len(balanced_samples),
            'n_high_assay_sim': len(high_assay_sim_samples),
            'n_high_chem_sim': len(high_chem_sim_samples)
        }
    
    def _calculate_metadata_stats(self, metadata_list: List[Dict]) -> Dict:
        """메타데이터 통계 계산"""
        assay_weights = [meta.get('weights', {}).get('assay', 0.5) for meta in metadata_list]
        chemical_weights = [meta.get('weights', {}).get('chemical', 0.5) for meta in metadata_list]
        
        return {
            'avg_assay_weight': np.mean(assay_weights),
            'avg_chemical_weight': np.mean(chemical_weights),
            'std_assay_weight': np.std(assay_weights),
            'std_chemical_weight': np.std(chemical_weights),
            'avg_time': np.mean([meta.get('elapsed_time', 0) for meta in metadata_list if meta])
        }
    
    def print_detailed_results(self, results: Dict, cost_breakdown: Dict):
        """상세 결과 출력"""
        print("\n" + "="*90)
        print("🔬 HYBRID CHEMRAG SYSTEM - DETAILED EVALUATION RESULTS")
        print("="*90)
        
        metrics = results['metrics']
        
        print(f"\n📊 Overall Performance Metrics:")
        print(f"   • Mean Absolute Error (MAE): {metrics['mae']:.2f}")
        print(f"   • Root Mean Square Error (RMSE): {metrics['rmse']:.2f}")
        print(f"   • R² Score: {metrics['r2']:.3f}")
        print(f"   • Accuracy within ±10: {metrics['within_10_pct']:.1f}%")
        print(f"   • Accuracy within ±20: {metrics['within_20_pct']:.1f}%")
        
        print(f"\n🔀 Hybrid Analysis Breakdown:")
        print(f"   • Average Assay Weight: {metrics['avg_assay_weight']:.2f} ± {metrics['std_assay_weight']:.2f}")
        print(f"   • Average Chemical Weight: {metrics['avg_chemical_weight']:.2f} ± {metrics['std_chemical_weight']:.2f}")
        print(f"   • Assay-Weighted Samples: {metrics['n_assay_weighted']} (MAE: {metrics['assay_weighted_mae']:.2f})")
        print(f"   • Chemical-Weighted Samples: {metrics['n_chemical_weighted']} (MAE: {metrics['chemical_weighted_mae']:.2f})")
        print(f"   • Balanced Samples: {metrics['n_balanced']} (MAE: {metrics['balanced_mae']:.2f})")
        
        print(f"\n🎯 Similarity Analysis:")
        print(f"   • High Assay Similarity (>0.7): {metrics['n_high_assay_sim']} samples")
        print(f"   • High Chemical Similarity (>0.7): {metrics['n_high_chem_sim']} samples")
        
        print(f"\n⚡ Performance:")
        print(f"   • Average Prediction Time: {metrics.get('avg_time', 0):.2f}s")
        print(f"   • Test Samples: {metrics['n_samples']}")
        print(f"   • Errors: {metrics['n_errors']}")
        
        print(f"\n💰 Cost Breakdown (Claude Sonnet 4):")
        print(f"   • API Calls: {cost_breakdown['api_calls']}")
        print(f"   • Input Tokens: {cost_breakdown['input_tokens']:,}")
        print(f"   • Output Tokens: {cost_breakdown['output_tokens']:,}")
        print(f"   • Total Cost: ${cost_breakdown['total_cost']:.4f}")
        
        # 최고/최악 예측 분석
        self._print_best_worst_predictions(results)
    
    def _print_best_worst_predictions(self, results: Dict):
        """최고/최악 예측 출력"""
        errors = [abs(a - p) for a, p in zip(results['actuals'], results['predictions'])]
        best_indices = np.argsort(errors)[:3]
        worst_indices = np.argsort(errors)[-3:][::-1]
        
        print(f"\n🏆 Best Hybrid Predictions:")
        for i, idx in enumerate(best_indices):
            actual = results['actuals'][idx]
            pred = results['predictions'][idx]
            metadata = results['metadata'][idx]
            error = abs(actual - pred)
            weights = metadata.get('weights', {})
            print(f"   {i+1}. Actual: {actual:2d}, Predicted: {pred:2d}, Error: {error:2d}, "
                  f"Weights: A={weights.get('assay', 0):.2f}/C={weights.get('chemical', 0):.2f}")
        
        print(f"\n🤔 Most Challenging Predictions:")
        for i, idx in enumerate(worst_indices):
            actual = results['actuals'][idx]
            pred = results['predictions'][idx]
            metadata = results['metadata'][idx]
            error = abs(actual - pred)
            weights = metadata.get('weights', {})
            print(f"   {i+1}. Actual: {actual:2d}, Predicted: {pred:2d}, Error: {error:2d}, "
                  f"Weights: A={weights.get('assay', 0):.2f}/C={weights.get('chemical', 0):.2f}")
