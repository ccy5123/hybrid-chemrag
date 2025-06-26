# models/hybrid_rag.py
import json
import time
import logging
import numpy as np
import random
import anthropic
import re
import math
from typing import List, Dict, Tuple

# LangChain imports
try:
    from langchain.vectorstores import FAISS
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document
    LANGCHAIN_AVAILABLE = True
    print("✅ LangChain loaded successfully!")
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("⚠️ LangChain not available. Install with: pip install langchain sentence-transformers")

from .data_parser import DataParser
from .utils import (
    create_mol_object, generate_fingerprints, calculate_molecular_properties,
    calculate_multi_fingerprint_similarity, combine_similarity_scores
)
from .vector_store_manager import VectorStoreManager

logger = logging.getLogger(__name__)

class HybridSMILESRAG:
    """하이브리드 RAG 시스템: 자연어(LangChain) + 화학적 유사도(RDKit)"""
    
    def __init__(self, claude_api_key: str = None, model: str = "claude-sonnet-4-20250514", temperature: float = 0.1):
        # Claude API 설정
        import config
        self.config = config
        
        self.claude_client = anthropic.Anthropic(api_key=claude_api_key or config.CLAUDE_API_KEY)
        if not (claude_api_key or config.CLAUDE_API_KEY):
            raise ValueError("Claude API key not found. Set in config.py or pass as parameter.")
        
        self.model = model
        self.temperature = temperature
        
        # 데이터 저장소
        self.train_data = []
        self.parsed_train_data = []
        
        # 벡터 스토어 매니저 초기화 (새로 추가!)
        if LANGCHAIN_AVAILABLE:
            self.vector_manager = VectorStoreManager(config)
            logger.info("📁 Vector store manager initialized")
        else:
            self.vector_manager = None
            logger.warning("⚠️ LangChain not available, vector store disabled")
        
        self.assay_vectorstore = None
        
        # RDKit 컴포넌트
        self.mol_objects = {}
        self.fingerprints = {}
        self.fingerprint_cache = {}  # 새로 추가!
        
        # 비용 추적
        self.cost_tracker = {
            'input_tokens': 0,
            'output_tokens': 0,
            'total_cost': 0.0,
            'api_calls': 0
        }
        
        logger.info("🔬 Hybrid RAG System initialized")
    
    def load_jsonl_data(self, file_path: str) -> List[Dict]:
        """JSONL 파일에서 데이터 로드"""
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line.strip())
                    data.append(item)
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
        
        logger.info(f"Loaded {len(data)} samples from {file_path}")
        return data
    
    def simple_train_test_split(self, data: List[Dict], test_size: float = None, 
                               random_state: int = None) -> Tuple[List[Dict], List[Dict]]:
        """간단한 train/test 분할"""
        from config import TEST_SIZE, RANDOM_STATE
        
        if test_size is None:
            test_size = TEST_SIZE
        if random_state is None:
            random_state = RANDOM_STATE
            
        random.seed(random_state)
        np.random.seed(random_state)
        
        shuffled_data = data.copy()
        random.shuffle(shuffled_data)
        
        split_idx = int(len(shuffled_data) * (1 - test_size))
        train_data = shuffled_data[:split_idx]
        test_data = shuffled_data[split_idx:]
        
        logger.info(f"Train set: {len(train_data)} samples")
        logger.info(f"Test set: {len(test_data)} samples")
        
        return train_data, test_data
    
    def prepare_hybrid_training_data(self, train_data: List[Dict]):
        """하이브리드 훈련 데이터 준비 - 벡터 DB 캐싱 포함"""
        logger.info("🏗️ Preparing hybrid training data with caching...")
        
        self.train_data = train_data
        self.parsed_train_data = []
        
        # 벡터스토어 정보 출력
        if self.vector_manager:
            info = self.vector_manager.get_vectorstore_info()
            logger.info("📊 Vector Store Status:")
            logger.info(f"   Assay vectorstore exists: {info['assay_vectorstore_exists']}")
            logger.info(f"   Fingerprint cache exists: {info['fingerprint_cache_exists']}")
            
            if info['assay_vectorstore_exists']:
                logger.info(f"   Vectorstore size: {info.get('vectorstore_size_mb', 0):.1f} MB")
            if info['fingerprint_cache_exists']:
                logger.info(f"   Cache size: {info.get('cache_size_mb', 0):.1f} MB")
        
        # 1. 기본 데이터 파싱
        for idx, item in enumerate(train_data):
            parsed = DataParser.parse_input_text(item['input_text'])
            parsed['logac50'] = int(item['output_text'])
            parsed['idx'] = idx
            parsed['activity_category'] = self._categorize_activity(parsed['logac50'])
            self.parsed_train_data.append(parsed)
            
            if (idx + 1) % 100 == 0:
                logger.info(f"   Parsed {idx + 1}/{len(train_data)} samples...")
        
        # 2. Assay 벡터스토어 로드/생성
        if LANGCHAIN_AVAILABLE and self.vector_manager:
            self.assay_vectorstore = self.vector_manager.load_or_create_assay_vectorstore(train_data)
        
        # 3. 분자 지문 캐시 로드/생성
        if self.vector_manager:
            self.fingerprint_cache = self.vector_manager.load_or_create_fingerprint_cache(self.parsed_train_data)
            
            # 파싱된 데이터에 캐시된 지문 적용
            for parsed in self.parsed_train_data:
                if parsed['smiles'] in self.fingerprint_cache:
                    parsed['fingerprints'] = self.fingerprint_cache[parsed['smiles']]
                    
                    # 분자 객체도 생성 (필요시)
                    if parsed['smiles'] not in self.mol_objects:
                        mol = create_mol_object(parsed['smiles'])
                        if mol is not None:
                            self.mol_objects[parsed['smiles']] = mol
                            parsed['mol'] = mol
                            parsed['molecular_props'] = calculate_molecular_properties(mol)
        
        logger.info(f"✅ Prepared {len(self.parsed_train_data)} training examples")
        
        if self.vector_manager:
            # 최종 통계
            cached_fingerprints = len(self.fingerprint_cache)
            vectorstore_docs = self.assay_vectorstore.index.ntotal if self.assay_vectorstore else 0
            
            logger.info("📈 Caching Statistics:")
            logger.info(f"   Cached fingerprints: {cached_fingerprints}")
            logger.info(f"   Vectorstore documents: {vectorstore_docs}")
    
    def _categorize_activity(self, logac50: int) -> str:
        """활성도 카테고리 분류"""
        if logac50 >= 85:
            return "Very High"
        elif logac50 >= 70:
            return "High"
        elif logac50 >= 50:
            return "Medium"
        elif logac50 >= 30:
            return "Low"
        else:
            return "Very Low"
    
    def hybrid_similarity_search(self, query_input: str, k_assay: int = None, k_chemical: int = None) -> Tuple[List[Dict], List[Dict]]:
        """하이브리드 유사도 검색"""
        from config import K_ASSAY, K_CHEMICAL
        
        if k_assay is None:
            k_assay = K_ASSAY
        if k_chemical is None:
            k_chemical = K_CHEMICAL
            
        parsed_query = DataParser.parse_input_text(query_input)
        
        # 실험 조건 유사도 검색
        similar_assays = []
        if LANGCHAIN_AVAILABLE and self.assay_vectorstore and parsed_query['assay_description']:
            try:
                assay_query = f"{parsed_query['assay_name']} {parsed_query['assay_description']}"
                assay_docs = self.assay_vectorstore.similarity_search_with_score(assay_query, k=k_assay)
                
                for doc, score in assay_docs:
                    similar_assays.append({
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'similarity_score': 1 - score,
                        'assay_name': doc.metadata.get('assay_name', ''),
                        'logac50': doc.metadata.get('logac50', 0)
                    })
                    
                logger.debug(f"Found {len(similar_assays)} similar assays")
            except Exception as e:
                logger.warning(f"Assay search failed: {e}")
        
        # 화학적 유사도 검색
        similar_molecules = []
        if parsed_query['smiles']:
            try:
                query_mol = create_mol_object(parsed_query['smiles'])
                if query_mol is not None:
                    query_fps = generate_fingerprints(query_mol, parsed_query['smiles'])
                    
                    similarities = []
                    for example in self.parsed_train_data:
                        if 'fingerprints' in example and example['fingerprints']:
                            from config import FINGERPRINT_WEIGHTS
                            similarity_scores = calculate_multi_fingerprint_similarity(
                                query_fps, example['fingerprints']
                            )
                            final_similarity = combine_similarity_scores(similarity_scores, FINGERPRINT_WEIGHTS)
                            similarities.append((final_similarity, example, similarity_scores))
                    
                    similarities.sort(key=lambda x: x[0], reverse=True)
                    
                    for sim_score, example, breakdown in similarities[:k_chemical]:
                        similar_molecules.append({
                            'smiles': example['smiles'],
                            'logac50': example['logac50'],
                            'activity_category': example.get('activity_category', ''),
                            'molecular_props': example.get('molecular_props', {}),
                            'similarity_score': sim_score,
                            'similarity_breakdown': breakdown,
                            'assay_name': example.get('assay_name', '')
                        })
                        
                    logger.debug(f"Found {len(similar_molecules)} similar molecules")
            except Exception as e:
                logger.warning(f"Chemical search failed: {e}")
        
        return similar_assays, similar_molecules
    
    def calculate_context_weights(self, similar_assays: List[Dict], similar_molecules: List[Dict]) -> Dict:
        """컨텍스트 가중치 동적 계산"""
        from config import K_ASSAY, K_CHEMICAL
        
        max_assay_sim = max([assay.get('similarity_score', 0) for assay in similar_assays]) if similar_assays else 0
        max_chem_sim = max([mol.get('similarity_score', 0) for mol in similar_molecules]) if similar_molecules else 0
        
        assay_availability = len(similar_assays) / K_ASSAY
        chem_availability = len(similar_molecules) / K_CHEMICAL
        
        # 동적 가중치 계산
        if max_assay_sim > 0.8 and max_chem_sim < 0.5:
            weights = {'assay': 0.75, 'chemical': 0.25}
        elif max_chem_sim > 0.8 and max_assay_sim < 0.5:
            weights = {'assay': 0.25, 'chemical': 0.75}
        elif max_assay_sim > 0.7 and max_chem_sim > 0.7:
            weights = {'assay': 0.5, 'chemical': 0.5}
        else:
            base_assay_weight = 0.4 + (assay_availability * 0.2)
            base_chem_weight = 0.6 - (assay_availability * 0.2)
            weights = {'assay': base_assay_weight, 'chemical': base_chem_weight}
        
        # 정규화
        total = weights['assay'] + weights['chemical']
        weights = {k: v/total for k, v in weights.items()}
        
        logger.debug(f"Context weights: Assay={weights['assay']:.2f}, Chemical={weights['chemical']:.2f}")
        
        return weights
    
    def create_hybrid_prompt(self, query_input: str, similar_assays: List[Dict], 
                           similar_molecules: List[Dict], weights: Dict) -> str:
        """하이브리드 컨텍스트 통합 프롬프트 생성"""
        
        parsed_query = DataParser.parse_input_text(query_input)
        
        # 실험 조건 컨텍스트
        assay_context = ""
        if similar_assays:
            assay_context = f"🧪 EXPERIMENTAL PROTOCOL CONTEXT (Weight: {weights['assay']:.2f}):\n\n"
            for i, assay in enumerate(similar_assays, 1):
                assay_context += f"Similar Assay {i} (Similarity: {assay['similarity_score']:.3f}):\n"
                assay_context += f"  {assay['content']}\n\n"
        else:
            assay_context = "🧪 EXPERIMENTAL PROTOCOL CONTEXT: No similar assays found.\n\n"
        
        # 화학 구조 컨텍스트
        chemical_context = ""
        if similar_molecules:
            chemical_context = f"🧬 CHEMICAL STRUCTURE CONTEXT (Weight: {weights['chemical']:.2f}):\n\n"
            for i, mol in enumerate(similar_molecules, 1):
                chemical_context += f"Similar Molecule {i} (Tanimoto: {mol['similarity_score']:.3f}):\n"
                chemical_context += f"  SMILES: {mol['smiles']}\n"
                chemical_context += f"  LogAC50: {mol['logac50']}\n"
                chemical_context += f"  Activity: {mol['activity_category']}\n"
                
                if 'similarity_breakdown' in mol:
                    breakdown = mol['similarity_breakdown']
                    chemical_context += f"  Fingerprint Details:\n"
                    chemical_context += f"    - Morgan: {breakdown.get('morgan', 0):.3f}\n"
                    chemical_context += f"    - MACCS: {breakdown.get('maccs', 0):.3f}\n"
                    chemical_context += f"    - RDKit: {breakdown.get('rdkit', 0):.3f}\n"
                
                if 'molecular_props' in mol and mol['molecular_props']:
                    props = mol['molecular_props']
                    chemical_context += f"  Properties: MW={props.get('mw', 'N/A'):.1f}, "
                    chemical_context += f"LogP={props.get('logp', 'N/A'):.2f}\n"
                
                chemical_context += "\n"
        else:
            chemical_context = "🧬 CHEMICAL STRUCTURE CONTEXT: No similar molecules found.\n\n"
        
        # 통합 프롬프트
        integrated_prompt = f"""<thinking>
I am performing a hybrid analysis combining experimental protocol knowledge and chemical structure similarity for toxicity prediction.

Query Details:
- Assay: {parsed_query['assay_name']}
- SMILES: {parsed_query['smiles']}

Context Analysis:
- Assay context weight: {weights['assay']:.2f}
- Chemical context weight: {weights['chemical']:.2f}

This weighting suggests I should prioritize {"experimental context" if weights['assay'] > weights['chemical'] else "chemical structure analysis"} while considering both sources of information.

Let me analyze the patterns systematically...
</thinking>

{assay_context}

{chemical_context}

🎯 HYBRID TOXICITY PREDICTION TASK:

Query Input:
- Assay: {parsed_query['assay_name']}
- SMILES: {parsed_query['smiles']}
- Task: {parsed_query['instruction']}

📊 INTEGRATED ANALYSIS FRAMEWORK:

1. **Context Weighting Strategy**:
   - Experimental Protocol Weight: {weights['assay']:.2f}
   - Chemical Structure Weight: {weights['chemical']:.2f}

2. **Primary Analysis Focus**:
   {"Focus on experimental protocol patterns and assay-specific factors" if weights['assay'] > 0.6 else "Focus on chemical structure-activity relationships" if weights['chemical'] > 0.6 else "Balance both experimental and chemical contexts equally"}

3. **Cross-Validation Approach**:
   - Compare patterns from both experimental and chemical contexts
   - Identify consistent vs conflicting predictions
   - Resolve conflicts using the higher-weighted context

4. **Evidence Integration**:
   - Experimental evidence: {"Strong" if len(similar_assays) >= 2 else "Moderate" if len(similar_assays) == 1 else "Weak"}
   - Chemical evidence: {"Strong" if len(similar_molecules) >= 3 else "Moderate" if len(similar_molecules) >= 1 else "Weak"}

🔬 REQUIRED ANALYSIS:

**EXPERIMENTAL CONTEXT ANALYSIS**:
[Analyze the experimental protocol patterns and assay-specific factors]

**CHEMICAL STRUCTURE ANALYSIS**:
[Analyze the molecular structure and chemical similarity patterns]

**INTEGRATED PREDICTION LOGIC**:
[Combine both contexts using the calculated weights]

**FINAL PREDICTION**: [INTEGER 0-100]

**CONFIDENCE ASSESSMENT**: [High/Medium/Low with justification based on context quality]

Remember: Weight your analysis according to the calculated context weights, but always provide reasoning from both experimental and chemical perspectives when available."""

        return integrated_prompt
    
    def predict_single(self, query_input: str) -> Tuple[int, str, float, Dict]:
        """단일 입력에 대한 하이브리드 예측"""
        start_time = time.time()
        
        try:
            # 하이브리드 유사도 검색
            similar_assays, similar_molecules = self.hybrid_similarity_search(query_input)
            
            # 컨텍스트 가중치 계산
            weights = self.calculate_context_weights(similar_assays, similar_molecules)
            
            # 프롬프트 생성
            prompt = self.create_hybrid_prompt(query_input, similar_assays, similar_molecules, weights)
            
            # Claude API 호출
            from config import MAX_TOKENS
            response = self.claude_client.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            
            result_text = response.content[0].text
            
            # 토큰 사용량 추적
            self.cost_tracker['input_tokens'] += response.usage.input_tokens
            self.cost_tracker['output_tokens'] += response.usage.output_tokens
            self.cost_tracker['api_calls'] += 1
            
            # 예측값 추출
            prediction = self._extract_prediction(result_text)
            elapsed_time = time.time() - start_time
            
            # 메타데이터
            metadata = {
                'weights': weights,
                'n_similar_assays': len(similar_assays),
                'n_similar_molecules': len(similar_molecules),
                'max_assay_similarity': max([a.get('similarity_score', 0) for a in similar_assays]) if similar_assays else 0,
                'max_chemical_similarity': max([m.get('similarity_score', 0) for m in similar_molecules]) if similar_molecules else 0
            }
            
            logger.debug(f"Hybrid prediction: {prediction} (assay_weight: {weights['assay']:.2f}, chem_weight: {weights['chemical']:.2f})")
            
            return prediction, result_text, elapsed_time, metadata
            
        except Exception as e:
            logger.error(f"Error in hybrid prediction: {e}")
            return 50, f"Error: {str(e)}", 0.0, {}
    
    def _extract_prediction(self, result_text: str) -> int:
        """LLM 응답에서 예측값 추출"""
        patterns = [
            r'FINAL PREDICTION:\s*(\d{1,3})',
            r'PREDICTION:\s*(\d{1,3})',
            r'Prediction:\s*(\d{1,3})',
            r'LogAC50:\s*(\d{1,3})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, result_text, re.IGNORECASE)
            if match:
                val = int(match.group(1))
                if 0 <= val <= 100:
                    return val
        
        # 백업: 0-100 범위의 첫 번째 숫자
        numbers = re.findall(r'\b(\d{1,3})\b', result_text)
        for num in numbers:
            val = int(num)
            if 0 <= val <= 100:
                return val
        
        logger.warning("Could not extract valid prediction, using default value 50")
        return 50
    
    def calculate_cost(self) -> Dict:
        """비용 계산"""
        from config import INPUT_COST_PER_1K, OUTPUT_COST_PER_1K
        
        input_cost = (self.cost_tracker['input_tokens'] / 1000) * INPUT_COST_PER_1K
        output_cost = (self.cost_tracker['output_tokens'] / 1000) * OUTPUT_COST_PER_1K
        total_cost = input_cost + output_cost
        
        return {
            'input_tokens': self.cost_tracker['input_tokens'],
            'output_tokens': self.cost_tracker['output_tokens'],
            'api_calls': self.cost_tracker['api_calls'],
            'input_cost': input_cost,
            'output_cost': output_cost,
            'total_cost': total_cost
        }