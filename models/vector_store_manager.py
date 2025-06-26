# models/vector_store_manager.py (새 파일!)
import os
import pickle
import logging
from typing import Dict, Optional, List
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """벡터 스토어 관리 클래스 - 저장/로드/캐싱 담당"""
    
    def __init__(self, config):
        self.config = config
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'}
        )
        
        # 디렉토리 생성
        os.makedirs(config.VECTOR_DB_PATH, exist_ok=True)
        
        logger.info(f"📁 Vector store manager initialized")
        logger.info(f"   Storage path: {config.VECTOR_DB_PATH}")
    
    def load_or_create_assay_vectorstore(self, train_data: List[Dict]) -> Optional[FAISS]:
        """Assay 벡터스토어 로드 또는 생성"""
        
        # 1. 기존 벡터스토어 로드 시도
        if not self.config.FORCE_REBUILD_VECTORSTORE and os.path.exists(self.config.ASSAY_VECTORSTORE_PATH):
            try:
                logger.info("📂 Loading existing assay vectorstore...")
                vectorstore = FAISS.load_local(
                    self.config.ASSAY_VECTORSTORE_PATH, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"✅ Loaded existing vectorstore with {vectorstore.index.ntotal} vectors")
                return vectorstore
            except Exception as e:
                logger.warning(f"⚠️ Failed to load existing vectorstore: {e}")
                logger.info("🔄 Will create new vectorstore instead...")
        
        # 2. 새 벡터스토어 생성
        logger.info("🏗️ Creating new assay vectorstore...")
        vectorstore = self._create_assay_vectorstore(train_data)
        
        # 3. 저장
        if vectorstore and self.config.SAVE_VECTORSTORE:
            try:
                logger.info("💾 Saving vectorstore for future use...")
                vectorstore.save_local(self.config.ASSAY_VECTORSTORE_PATH)
                logger.info(f"✅ Vectorstore saved to {self.config.ASSAY_VECTORSTORE_PATH}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to save vectorstore: {e}")
        
        return vectorstore
    
    def _create_assay_vectorstore(self, train_data: List[Dict]) -> Optional[FAISS]:
        """새로운 Assay 벡터스토어 생성"""
        from .data_parser import DataParser
        
        assay_documents = []
        
        for idx, item in enumerate(train_data):
            parsed = DataParser.parse_input_text(item['input_text'])
            logac50 = int(item['output_text'])
            
            if parsed['assay_description']:
                activity_category = self._categorize_activity(logac50)
                
                assay_doc_content = f"""
                Assay: {parsed['assay_name']}
                Description: {parsed['assay_description']}
                Activity: {logac50}
                Category: {activity_category}
                Instructions: {parsed['instruction']}
                """
                
                assay_doc = Document(
                    page_content=assay_doc_content,
                    metadata={
                        'assay_name': parsed['assay_name'],
                        'logac50': logac50,
                        'idx': idx,
                        'activity_category': activity_category
                    }
                )
                assay_documents.append(assay_doc)
            
            if (idx + 1) % 100 == 0:
                logger.info(f"   Processed {idx + 1}/{len(train_data)} documents...")
        
        if not assay_documents:
            logger.warning("⚠️ No assay documents found for vectorstore creation")
            return None
        
        # 문서 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP
        )
        split_docs = text_splitter.split_documents(assay_documents)
        
        # FAISS 벡터스토어 생성
        vectorstore = FAISS.from_documents(split_docs, self.embeddings)
        
        logger.info(f"✅ Created vectorstore with {len(split_docs)} chunks from {len(assay_documents)} documents")
        
        return vectorstore
    
    def load_or_create_fingerprint_cache(self, parsed_train_data: List[Dict]) -> Dict:
        """분자 지문 캐시 로드 또는 생성"""
        
        # 1. 기존 캐시 로드 시도
        if not self.config.FORCE_REBUILD_VECTORSTORE and os.path.exists(self.config.FINGERPRINT_CACHE_PATH):
            try:
                logger.info("📂 Loading existing fingerprint cache...")
                with open(self.config.FINGERPRINT_CACHE_PATH, 'rb') as f:
                    cache = pickle.load(f)
                logger.info(f"✅ Loaded fingerprint cache with {len(cache)} entries")
                return cache
            except Exception as e:
                logger.warning(f"⚠️ Failed to load fingerprint cache: {e}")
                logger.info("🔄 Will create new cache instead...")
        
        # 2. 새 캐시 생성
        logger.info("🏗️ Creating new fingerprint cache...")
        cache = self._create_fingerprint_cache(parsed_train_data)
        
        # 3. 저장
        if self.config.SAVE_VECTORSTORE:
            try:
                logger.info("💾 Saving fingerprint cache for future use...")
                with open(self.config.FINGERPRINT_CACHE_PATH, 'wb') as f:
                    pickle.dump(cache, f)
                logger.info(f"✅ Fingerprint cache saved to {self.config.FINGERPRINT_CACHE_PATH}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to save fingerprint cache: {e}")
        
        return cache
    
    def _create_fingerprint_cache(self, parsed_train_data: List[Dict]) -> Dict:
        """새로운 분자 지문 캐시 생성"""
        from .utils import create_mol_object, generate_fingerprints
        
        cache = {}
        processed = 0
        
        for example in parsed_train_data:
            if 'fingerprints' in example and example['fingerprints']:
                # 이미 처리된 경우
                cache[example['smiles']] = example['fingerprints']
                processed += 1
            elif example['smiles']:
                # 새로 처리
                mol = create_mol_object(example['smiles'])
                if mol is not None:
                    fingerprints = generate_fingerprints(mol, example['smiles'])
                    if fingerprints:
                        cache[example['smiles']] = fingerprints
                        processed += 1
            
            if processed % 100 == 0:
                logger.info(f"   Processed {processed} fingerprints...")
        
        logger.info(f"✅ Created fingerprint cache with {len(cache)} entries")
        return cache
    
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
    
    def get_vectorstore_info(self) -> Dict:
        """벡터스토어 정보 반환"""
        info = {
            'assay_vectorstore_exists': os.path.exists(self.config.ASSAY_VECTORSTORE_PATH),
            'fingerprint_cache_exists': os.path.exists(self.config.FINGERPRINT_CACHE_PATH),
            'vectorstore_path': self.config.ASSAY_VECTORSTORE_PATH,
            'cache_path': self.config.FINGERPRINT_CACHE_PATH
        }
        
        # 파일 크기 정보
        if info['assay_vectorstore_exists']:
            try:
                size = sum(os.path.getsize(os.path.join(self.config.ASSAY_VECTORSTORE_PATH, f)) 
                          for f in os.listdir(self.config.ASSAY_VECTORSTORE_PATH))
                info['vectorstore_size_mb'] = size / (1024 * 1024)
            except:
                info['vectorstore_size_mb'] = 0
        
        if info['fingerprint_cache_exists']:
            try:
                info['cache_size_mb'] = os.path.getsize(self.config.FINGERPRINT_CACHE_PATH) / (1024 * 1024)
            except:
                info['cache_size_mb'] = 0
        
        return info
    
    def clear_all_stores(self):
        """모든 저장된 벡터스토어와 캐시 삭제"""
        import shutil
        
        try:
            if os.path.exists(self.config.ASSAY_VECTORSTORE_PATH):
                shutil.rmtree(self.config.ASSAY_VECTORSTORE_PATH)
                logger.info("🗑️ Deleted assay vectorstore")
            
            if os.path.exists(self.config.FINGERPRINT_CACHE_PATH):
                os.remove(self.config.FINGERPRINT_CACHE_PATH)
                logger.info("🗑️ Deleted fingerprint cache")
                
            logger.info("✅ All vector stores cleared")
        except Exception as e:
            logger.error(f"❌ Error clearing stores: {e}")