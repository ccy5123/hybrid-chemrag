# models/data_parser.py
import re
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class DataParser:
    """입력 텍스트를 실험 조건과 SMILES로 분리"""
    
    @staticmethod
    def parse_input_text(input_text: str) -> Dict:
        """input_text를 구성 요소로 분리"""
        try:
            # SMILES 추출
            smiles_pattern = r'SMILES:\s*([^\n\r]+)'
            smiles_match = re.search(smiles_pattern, input_text)
            smiles = smiles_match.group(1).strip() if smiles_match else ""
            
            # Assay 이름 추출
            assay_pattern = r'Assay:\s*([^\n\r]+)'
            assay_match = re.search(assay_pattern, input_text)
            assay_name = assay_match.group(1).strip() if assay_match else ""
            
            # 실험 설명 추출 (TOX21... 로 시작하는 부분)
            assay_desc_pattern = r'(TOX21_[^\.]+[^\.]*\.)'
            assay_desc_match = re.search(assay_desc_pattern, input_text, re.DOTALL)
            assay_description = assay_desc_match.group(1).strip() if assay_desc_match else ""
            
            # 전체 지시사항 추출
            instruction_pattern = r'(Given an Assay and SMILES.*?)(?:SMILES:|$)'
            instruction_match = re.search(instruction_pattern, input_text, re.DOTALL)
            instruction = instruction_match.group(1).strip() if instruction_match else ""
            
            return {
                'smiles': smiles,
                'assay_name': assay_name,
                'assay_description': assay_description,
                'instruction': instruction,
                'full_text': input_text
            }
            
        except Exception as e:
            logger.error(f"Error parsing input text: {e}")
            return {
                'smiles': '',
                'assay_name': '',
                'assay_description': '',
                'instruction': '',
                'full_text': input_text
            }
