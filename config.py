# config.py

import os

CLAUDE_API_KEY = os.getenv("ANTHROPIC_API_KEY") or "your-key-here"
DATA_FILE = "data/combined_train_sampled2.jsonl"
