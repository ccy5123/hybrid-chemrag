# This script initializes a hybrid predictor using the Claude API key and runs an evaluation on the specified data file.
# Ensure that the necessary modules are installed and the environment is set up correctly.

from config import CLAUDE_API_KEY, DATA_FILE
from models.hybrid_predictor import HybridSMILESRAG
from evaluation.evaluator import run_evaluation

def main():
    predictor = HybridSMILESRAG(claude_api_key=CLAUDE_API_KEY)
    run_evaluation(predictor, data_path=DATA_FILE)
    save_results(predictions, actuals, times)

if __name__ == "__main__":
    main()