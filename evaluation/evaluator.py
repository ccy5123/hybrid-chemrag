# evaluator.py

import numpy as np
import json
from typing import Dict

def run_evaluation(predictor, data_path: str):
    data = predictor.load_jsonl_data(data_path)
    train_data, test_data = predictor.simple_train_test_split(data)
    predictor.prepare_hybrid_training_data(train_data)

    predictions = []
    actuals = []
    times = []

    print("\n🔍 Running evaluation on test samples...")
    for item in test_data[:20]:  # limit to 20 for cost control
        input_text = item['input_text']
        actual = int(item['output_text'])
        pred, _, elapsed = predictor.predict_single(input_text)

        predictions.append(pred)
        actuals.append(actual)
        times.append(elapsed)

        print(f"Actual: {actual}, Predicted: {pred}, Time: {elapsed:.2f}s")

    mae = np.mean([abs(a - p) for a, p in zip(actuals, predictions)])
    rmse = np.sqrt(np.mean([(a - p) ** 2 for a, p in zip(actuals, predictions)]))
    within_10 = sum([1 for a, p in zip(actuals, predictions) if abs(a - p) <= 10]) / len(actuals) * 100

    print("\n📊 Evaluation Results")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"Within ±10: {within_10:.1f}%")

    cost = predictor.calculate_cost()
    print("\n💰 Cost Summary")
    print(f"Input tokens: {cost['input_tokens']}, Output tokens: {cost['output_tokens']}")
    print(f"Estimated cost: ${cost['total_cost']:.4f}")


def save_results(predictions, actuals, times, file_path="results/predictions.json"):
    results = [
        {"actual": a, "predicted": p, "elapsed_time": t}
        for a, p, t in zip(actuals, predictions, times)
    ]
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n📁 Saved prediction results to {file_path}")