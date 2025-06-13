# 🧬 Hybrid ChemRAG

A hybrid RAG (Retrieval-Augmented Generation) system for molecular toxicity prediction, combining chemical structure similarity with experimental assay context using RDKit, LangChain, and Claude Sonnet 4.

---

## 🔍 Overview
This project predicts the toxicity (LogAC50) of molecules based on a given assay and SMILES input. It performs a dual similarity search:

- **Chemical similarity** using molecular fingerprints (RDKit)
- **Assay similarity** using semantic embeddings (LangChain + HuggingFace)

Then, it combines both contexts and prompts Claude Sonnet 4 to make an integrated prediction.

---

## 🚀 Features
- RDKit-based fingerprint similarity (Morgan, MACCS, AtomPair)
- LangChain-based FAISS vector search for text similarity
- Dynamic context weighting for hybrid reasoning
- Claude Sonnet 4 LLM inference via Anthropic API
- MAE, RMSE, token usage, and cost tracking

---

## 📦 Installation
```bash
pip install -r requirements.txt
```

RDKit is best installed via conda:
```bash
conda install -c conda-forge rdkit
```

---

## 📁 Project Structure
```
hybrid-chemrag/
├── main.py                 # Entry point
├── config.py               # API keys & data path
├── requirements.txt
├── .gitignore
├── data/                   # Input data
├── models/                 # Core logic
├── evaluation/             # Evaluation utilities
└── prompts/                # Prompt template (optional)
```

---

## 🧪 Usage
1. Add your Claude API key in `config.py` or via environment variable `ANTHROPIC_API_KEY`
2. Place your `.jsonl` dataset into `data/`
3. Run the pipeline:
```bash
python main.py
```

---

## 📊 Output
You will see:
- Prediction results
- Evaluation metrics (MAE, RMSE, within ±10)
- Claude API token usage and cost

---

## 📜 License
MIT

---

## 🤝 Acknowledgements
- [RDKit](https://www.rdkit.org/)
- [LangChain](https://www.langchain.com/)
- [Anthropic Claude](https://www.anthropic.com/)
- [SentenceTransformers](https://www.sbert.net/)
