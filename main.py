# main.py 수정 버전 (주요 부분)
import logging
import json
import os
from models.hybrid_rag import HybridSMILESRAG
from evaluation.evaluator import HybridEvaluator
from utils.vector_store_utils import print_vector_store_status
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """메인 실행 함수"""
    try:
        print("🔬 Initializing Hybrid ChemRAG System...")
        print("   • Natural Language Processing: LangChain + Local Embeddings")
        print("   • Chemical Similarity: RDKit + Molecular Fingerprints")
        print("   • Integration: Claude Sonnet 4")
        print("   • Vector DB Caching: FAISS + Pickle")
        
        # 벡터 스토어 상태 출력
        print_vector_store_status()
        
        # 시스템 초기화
        predictor = HybridSMILESRAG()
        
        print(f"\n📚 Loading data from {config.DATA_PATH}...")
        # 데이터 로드
        if not os.path.exists(config.DATA_PATH):
            raise FileNotFoundError(f"Data file not found: {config.DATA_PATH}")
        
        data = predictor.load_jsonl_data(config.DATA_PATH)
        
        print("✂️ Splitting data...")
        # Train/Test 분할
        train_data, test_data = predictor.simple_train_test_split(data)
        
        print("🏗️ Preparing hybrid training data with vector caching...")
        print(f"   • Force rebuild: {config.FORCE_REBUILD_VECTORSTORE}")
        print(f"   • Save stores: {config.SAVE_VECTORSTORE}")
        
        # 훈련 데이터 준비 (캐싱 포함)
        predictor.prepare_hybrid_training_data(train_data)
        
        print("\n🔬 Evaluating with Hybrid RAG System...")
        print("   • Experimental similarity search + Chemical similarity search")
        print("   • Dynamic context weighting + Integrated reasoning")
        
        # 테스트 (일부만 - 비용 고려)
        test_subset = test_data[:10]
        
        print(f"\n🔬 Running predictions on {len(test_subset)} samples...")
        
        # 고급 평가자 사용
        evaluator = HybridEvaluator(predictor)
        results = evaluator.evaluate_test_set(test_subset)
        
        print("\n💰 Calculating costs...")
        cost_breakdown = predictor.calculate_cost()
        
        print("\n📊 Generating comprehensive results...")
        evaluator.print_detailed_results(results, cost_breakdown)
        
        # 결과 저장
        output_data = {
            'model': 'hybrid-chemrag',
            'components': {
                'assay_similarity': 'LangChain + HuggingFace Embeddings',
                'chemical_similarity': 'RDKit + Molecular Fingerprints',
                'integration': 'Claude Sonnet 4'
            },
            'results': results,
            'cost_breakdown': cost_breakdown,
            'settings': {
                'data_file': config.DATA_PATH,
                'test_subset_size': len(test_subset)
            }
        }
        
        with open('hybrid_chemrag_results.json', 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to 'hybrid_chemrag_results.json'")
        
        # 성능 요약
        mae = results['metrics']['mae']
        r2 = results['metrics']['r2']
        within_10 = results['metrics']['within_10_pct']
        avg_assay_weight = results['metrics']['avg_assay_weight']
        avg_chem_weight = results['metrics']['avg_chemical_weight']
        
        print(f"\n🏆 HYBRID SYSTEM PERFORMANCE SUMMARY:")
        print(f"   MAE: {mae:.2f} | R²: {r2:.3f} | Within ±10: {within_10:.1f}%")
        print(f"   Avg Context Weights: Assay={avg_assay_weight:.2f}, Chemical={avg_chem_weight:.2f}")
        
        # 시스템 상태 체크
        from models.utils import RDKIT_AVAILABLE
        from models.hybrid_rag import LANGCHAIN_AVAILABLE
        
        print(f"\n🔧 System Status:")
        if LANGCHAIN_AVAILABLE and RDKIT_AVAILABLE:
            print("   ✅ Full Hybrid System: Both LangChain and RDKit operational!")
            print("   🎯 Optimal performance with dual similarity engines")
        elif RDKIT_AVAILABLE:
            print("   ⚠️ Partial System: RDKit operational, LangChain disabled")
            print("   📝 Install LangChain for assay similarity: pip install langchain sentence-transformers")
        elif LANGCHAIN_AVAILABLE:
            print("   ⚠️ Partial System: LangChain operational, RDKit disabled")
            print("   🧪 Install RDKit for chemical similarity: conda install -c conda-forge rdkit")
        else:
            print("   ❌ Minimal System: Both engines disabled")
            print("   📦 Install dependencies for full functionality")
        
        print("\n🎉 Hybrid ChemRAG Analysis complete!")
        print("💡 Next run will be faster using cached vector stores!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()