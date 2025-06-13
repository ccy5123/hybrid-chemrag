# hybrid_perdictor.py

import os
import json
import time
import logging
import numpy as np
from typing import List, Dict, Tuple

from models.data_parser import DataParser
from models.utils import (
    create_mol_object, generate_fingerprints,
    calculate_molecular_properties, calculate_r2
)

try:
    from langchain.vectorstores import FAISS
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

import anthropic

logger = logging.getLogger(__name__)

class HybridSMILESRAG:
    def __init__(self, claude_api_key: str, model: str = "claude-sonnet-4-20250514", temperature: float = 0.1):
        self.model = model
        self.temperature = temperature
        self.claude_client = anthropic.Anthropic(api_key=claude_api_key)

        self.train_data = []
        self.parsed_train_data = []
        self.mol_objects = {}
        self.fingerprints = {}
        self.assay_vectorstore = None

        self.cost_tracker = {
            'input_tokens': 0,
            'output_tokens': 0,
            'total_cost': 0.0,
            'api_calls': 0
        }

        if LANGCHAIN_AVAILABLE:
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        else:
            self.embeddings = None

    def load_jsonl_data(self, path: str) -> List[Dict]:
        with open(path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f if line.strip()]

    def simple_train_test_split(self, data: List[Dict], test_size: float = 0.2, random_state: int = 42) -> Tuple[List[Dict], List[Dict]]:
        np.random.seed(random_state)
        shuffled = data.copy()
        np.random.shuffle(shuffled)
        split_idx = int(len(shuffled) * (1 - test_size))
        return shuffled[:split_idx], shuffled[split_idx:]

    def prepare_hybrid_training_data(self, train_data: List[Dict]):
        self.train_data = train_data
        self.parsed_train_data = []
        assay_docs = []

        for idx, item in enumerate(train_data):
            parsed = DataParser.parse_input_text(item['input_text'])
            parsed['logac50'] = int(item['output_text'])
            parsed['idx'] = idx

            if RDKIT_AVAILABLE and parsed['smiles']:
                mol = create_mol_object(parsed['smiles'])
                if mol:
                    parsed['mol'] = mol
                    parsed['fingerprints'] = generate_fingerprints(mol)
                    parsed['molecular_props'] = calculate_molecular_properties(mol)

            if LANGCHAIN_AVAILABLE and parsed['assay_description']:
                content = f"Assay: {parsed['assay_name']}\nDescription: {parsed['assay_description']}"
                doc = Document(
                    page_content=content,
                    metadata={'assay_name': parsed['assay_name'], 'logac50': parsed['logac50'], 'idx': idx}
                )
                assay_docs.append(doc)

            self.parsed_train_data.append(parsed)

        if LANGCHAIN_AVAILABLE and assay_docs:
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            split_docs = splitter.split_documents(assay_docs)
            self.assay_vectorstore = FAISS.from_documents(split_docs, self.embeddings)

    def predict_single(self, query_input: str) -> Tuple[int, str, float]:
        start_time = time.time()
        parsed = DataParser.parse_input_text(query_input)

        # 유사도 검색 생략 (요약용 버전)
        # 프롬프트 작성
        prompt = f"Predict toxicity for:\nAssay: {parsed['assay_name']}\nSMILES: {parsed['smiles']}\nInstruction: {parsed['instruction']}"

        response = self.claude_client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}]
        )

        result_text = response.content[0].text
        self.cost_tracker['input_tokens'] += response.usage.input_tokens
        self.cost_tracker['output_tokens'] += response.usage.output_tokens
        self.cost_tracker['api_calls'] += 1

        pred = self._extract_prediction(result_text)
        elapsed = time.time() - start_time
        return pred, result_text, elapsed

    def _extract_prediction(self, result_text: str) -> int:
        import re
        match = re.search(r'(?:FINAL\s+)?PREDICTION\s*[:=]\s*(\d+)', result_text, re.IGNORECASE)
        if match:
            val = int(match.group(1))
            if 0 <= val <= 100:
                return val
        return 50  # default fallback

    def calculate_cost(self) -> Dict:
        input_cost_per_1k = 0.003
        output_cost_per_1k = 0.015
        input_cost = (self.cost_tracker['input_tokens'] / 1000) * input_cost_per_1k
        output_cost = (self.cost_tracker['output_tokens'] / 1000) * output_cost_per_1k
        total = input_cost + output_cost
        return {
            **self.cost_tracker,
            'input_cost': input_cost,
            'output_cost': output_cost,
            'total_cost': total
        }