# evaluation/metrics.py
import numpy as np
import math
from typing import List, Dict

def calculate_metrics(predictions: List[int], actuals: List[int]) -> Dict:
    """평가 메트릭 계산"""
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
        'within_20_pct': within_20
    }

def print_results(results: Dict, cost_breakdown: Dict):
    """결과 출력"""
    print("\n" + "="*80)
    print("🔬 HYBRID RAG SYSTEM RESULTS")
    print("="*80)
    
    metrics = results['metrics']
    
    print(f"\n📊 Performance Metrics:")
    print(f"   MAE: {metrics['mae']:.2f}")
    print(f"   RMSE: {metrics['rmse']:.2f}")
    print(f"   R²: {metrics['r2']:.3f}")
    print(f"   Within ±10: {metrics['within_10_pct']:.1f}%")
    print(f"   Within ±20: {metrics['within_20_pct']:.1f}%")
    
    print(f"\n💰 Cost:")
    print(f"   API Calls: {cost_breakdown['api_calls']}")
    print(f"   Total Cost: ${cost_breakdown['total_cost']:.4f}")