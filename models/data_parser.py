# data_parser.py

import re
import logging

logger = logging.getLogger(__name__)

class DataParser:
    """입력 텍스트를 Assay, SMILES 등으로 파싱"""

    @staticmethod
    def parse_input_text(input_text: str) -> dict:
        try:
            smiles_pattern = r'SMILES:\s*([^\n\r]+)'
            assay_pattern = r'Assay:\s*([^\n\r]+)'
            desc_pattern = r'(TOX21_[^\.]+[^\.]*\.)'
            instr_pattern = r'(Given an Assay and SMILES.*?)\s*(?:SMILES:|$)'

            smiles = re.search(smiles_pattern, input_text)
            assay = re.search(assay_pattern, input_text)
            desc = re.search(desc_pattern, input_text, re.DOTALL)
            instr = re.search(instr_pattern, input_text, re.DOTALL)

            return {
                'smiles': smiles.group(1).strip() if smiles else '',
                'assay_name': assay.group(1).strip() if assay else '',
                'assay_description': desc.group(1).strip() if desc else '',
                'instruction': instr.group(1).strip() if instr else '',
                'full_text': input_text.strip()
            }
        except Exception as e:
            logger.error(f"Failed to parse input: {e}")
            return {
                'smiles': '',
                'assay_name': '',
                'assay_description': '',
                'instruction': '',
                'full_text': input_text.strip()
            }
