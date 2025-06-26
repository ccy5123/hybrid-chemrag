# utils/vector_store_utils.py (새 파일 - 벡터 DB 관리 유틸리티)
import os
import logging
from typing import Dict
import config

logger = logging.getLogger(__name__)

def clear_vector_stores():
    """모든 벡터 스토어 삭제"""
    from models.vector_store_manager import VectorStoreManager
    
    manager = VectorStoreManager(config)
    manager.clear_all_stores()

def get_vector_store_info() -> Dict:
    """벡터 스토어 정보 조회"""
    from models.vector_store_manager import VectorStoreManager
    
    manager = VectorStoreManager(config)
    return manager.get_vectorstore_info()

def print_vector_store_status():
    """벡터 스토어 상태 출력"""
    info = get_vector_store_info()
    
    print("\n📊 Vector Store Status:")
    print(f"   Assay Vectorstore: {'✅ EXISTS' if info['assay_vectorstore_exists'] else '❌ NOT FOUND'}")
    print(f"   Fingerprint Cache: {'✅ EXISTS' if info['fingerprint_cache_exists'] else '❌ NOT FOUND'}")
    
    if info['assay_vectorstore_exists']:
        print(f"   Vectorstore Size: {info.get('vectorstore_size_mb', 0):.1f} MB")
    if info['fingerprint_cache_exists']:
        print(f"   Cache Size: {info.get('cache_size_mb', 0):.1f} MB")
    
    print(f"   Storage Path: {config.VECTOR_DB_PATH}")

if __name__ == "__main__":
    # 스크립트로 실행시 상태 출력
    print_vector_store_status()
