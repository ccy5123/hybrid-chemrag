
# config.py 수정 버전
import os

# Claude API 설정
CLAUDE_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-sonnet-4-20250514"
CLAUDE_TEMPERATURE = 0.1

# 데이터 경로
DATA_PATH = "./data/combined_train_sampled2.jsonl"

# 벡터 DB 저장 경로 (새로 추가!)
VECTOR_DB_PATH = "vector_stores/"
ASSAY_VECTORSTORE_PATH = os.path.join(VECTOR_DB_PATH, "assay_vectorstore")
FINGERPRINT_CACHE_PATH = os.path.join(VECTOR_DB_PATH, "fingerprint_cache.pkl")

# 벡터 DB 설정
FORCE_REBUILD_VECTORSTORE = False  # True면 기존 DB 무시하고 새로 생성
SAVE_VECTORSTORE = True            # True면 생성된 DB 저장

# 기존 설정들...
TEST_SIZE = 0.1
RANDOM_STATE = 42
MAX_TOKENS = 3000

# 비용 계산 (per 1K tokens)
INPUT_COST_PER_1K = 0.003
OUTPUT_COST_PER_1K = 0.015

# 유사도 검색 설정
K_ASSAY = 30
K_CHEMICAL = 30

# 분자 지문 가중치
FINGERPRINT_WEIGHTS = {
    'morgan': 0.4,
    'maccs': 0.3,
    'rdkit': 0.2,
    'atompair': 0.1
}

# LangChain 설정
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50