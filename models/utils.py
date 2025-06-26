# models/utils.py
import logging
from typing import Dict, Optional, List
import numpy as np

# RDKit imports
try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, MACCSkeys, Descriptors
    RDKIT_AVAILABLE = True
    print("✅ RDKit loaded successfully!")
except ImportError:
    RDKIT_AVAILABLE = False
    print("⚠️ RDKit not available. Install with: conda install -c conda-forge rdkit")

logger = logging.getLogger(__name__)

def create_mol_object(smiles: str):
    """SMILES에서 RDKit 분자 객체 생성"""
    if not RDKIT_AVAILABLE:
        return None
        
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            mol = Chem.AddHs(mol)
            return mol
        return None
    except Exception as e:
        logger.debug(f"Error creating mol object for {smiles}: {e}")
        return None

def generate_fingerprints(mol, smiles: str) -> Dict:
    """다양한 분자 지문 생성"""
    if not RDKIT_AVAILABLE or mol is None:
        return {}
    
    fingerprints = {}
    try:
        fingerprints['morgan'] = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        fingerprints['maccs'] = MACCSkeys.GenMACCSKeys(mol)
        fingerprints['rdkit'] = AllChem.GetRDKitFPGenerator().GetFingerprint(mol)
        fingerprints['atompair'] = AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=2048)
    except Exception as e:
        logger.debug(f"Error generating fingerprints for {smiles}: {e}")
    
    return fingerprints

def calculate_molecular_properties(mol) -> Dict:
    """분자 물성 계산"""
    if not RDKIT_AVAILABLE or mol is None:
        return {}
        
    try:
        props = {
            'mw': Descriptors.MolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'hbd': Descriptors.NumHDonors(mol),
            'hba': Descriptors.NumHAcceptors(mol),
            'tpsa': Descriptors.TPSA(mol),
            'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'aromatic_rings': Descriptors.NumAromaticRings(mol),
            'heavy_atoms': Descriptors.HeavyAtomCount(mol)
        }
        return props
    except Exception:
        return {}

def calculate_r2(y_true: List[float], y_pred: List[float]) -> float:
    """R² Score 계산"""
    y_mean = np.mean(y_true)
    ss_tot = sum([(y - y_mean)**2 for y in y_true])
    ss_res = sum([(t - p)**2 for t, p in zip(y_true, y_pred)])
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

def calculate_multi_fingerprint_similarity(query_fps: Dict, target_fps: Dict) -> Dict:
    """다중 지문을 사용한 유사도 계산"""
    similarities = {}
    fingerprint_types = ['morgan', 'maccs', 'rdkit', 'atompair']
    
    for fp_type in fingerprint_types:
        if fp_type in query_fps and fp_type in target_fps:
            try:
                if RDKIT_AVAILABLE:
                    tanimoto_sim = DataStructs.TanimotoSimilarity(
                        query_fps[fp_type], target_fps[fp_type]
                    )
                    similarities[fp_type] = tanimoto_sim
                else:
                    similarities[fp_type] = 0.0
            except Exception as e:
                logger.debug(f"Error calculating {fp_type} similarity: {e}")
                similarities[fp_type] = 0.0
        else:
            similarities[fp_type] = 0.0
    
    return similarities

def combine_similarity_scores(similarity_scores: Dict, weights: Dict) -> float:
    """여러 지문 유사도를 가중 평균으로 결합"""
    weighted_sum = 0.0
    total_weight = 0.0
    
    for fp_type, weight in weights.items():
        if fp_type in similarity_scores:
            weighted_sum += similarity_scores[fp_type] * weight
            total_weight += weight
    
    return weighted_sum / total_weight if total_weight > 0 else 0.0