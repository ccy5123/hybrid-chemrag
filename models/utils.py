# utils.py

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys, Descriptors
from typing import Dict
import numpy as np
import logging

logger = logging.getLogger(__name__)


def create_mol_object(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            mol = Chem.AddHs(mol)
        return mol
    except Exception as e:
        logger.warning(f"Mol creation failed for {smiles}: {e}")
        return None


def generate_fingerprints(mol) -> Dict:
    try:
        return {
            'morgan': AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048),
            'maccs': MACCSkeys.GenMACCSKeys(mol),
            'rdkit': AllChem.GetRDKitFPGenerator().GetFingerprint(mol),
            'atompair': AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=2048)
        }
    except Exception as e:
        logger.warning(f"Fingerprint generation failed: {e}")
        return {}


def calculate_molecular_properties(mol) -> Dict:
    try:
        return {
            'mw': Descriptors.MolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'hbd': Descriptors.NumHDonors(mol),
            'hba': Descriptors.NumHAcceptors(mol),
            'tpsa': Descriptors.TPSA(mol),
            'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'aromatic_rings': Descriptors.NumAromaticRings(mol),
            'heavy_atoms': Descriptors.HeavyAtomCount(mol)
        }
    except Exception as e:
        logger.warning(f"Property calculation failed: {e}")
        return {}


def calculate_r2(y_true, y_pred) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
