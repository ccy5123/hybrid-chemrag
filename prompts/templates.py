# prompts/templates.py
"""
고급 프롬프트 템플릿 모듈
"""

HYBRID_ANALYSIS_TEMPLATE = """<thinking>
I am performing a hybrid analysis combining experimental protocol knowledge and chemical structure similarity for toxicity prediction.

Query Details:
- Assay: {assay_name}
- SMILES: {smiles}

Context Analysis:
- Assay context weight: {assay_weight:.2f}
- Chemical context weight: {chemical_weight:.2f}

This weighting suggests I should prioritize {"experimental context" if assay_weight > chemical_weight else "chemical structure analysis"} while considering both sources of information.

Let me analyze the patterns systematically...
</thinking>

{assay_context}

{chemical_context}

🎯 HYBRID TOXICITY PREDICTION TASK:

Query Input:
- Assay: {assay_name}
- SMILES: {smiles}
- Task: {instruction}

📊 INTEGRATED ANALYSIS FRAMEWORK:

1. **Context Weighting Strategy**:
   - Experimental Protocol Weight: {assay_weight:.2f}
   - Chemical Structure Weight: {chemical_weight:.2f}

2. **Primary Analysis Focus**:
   {focus_strategy}

3. **Cross-Validation Approach**:
   - Compare patterns from both experimental and chemical contexts
   - Identify consistent vs conflicting predictions
   - Resolve conflicts using the higher-weighted context

4. **Evidence Integration**:
   - Experimental evidence: {experimental_evidence_strength}
   - Chemical evidence: {chemical_evidence_strength}

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

def create_hybrid_prompt(query_input: str, similar_assays: list, similar_molecules: list, weights: dict) -> str:
    """고급 하이브리드 프롬프트 생성"""
    from models.data_parser import DataParser
    
    parsed_query = DataParser.parse_input_text(query_input)
    
    # 실험 조건 컨텍스트 포맷팅
    if similar_assays:
        assay_context = f"🧪 EXPERIMENTAL PROTOCOL CONTEXT (Weight: {weights['assay']:.2f}):\n\n"
        for i, assay in enumerate(similar_assays, 1):
            assay_context += f"Similar Assay {i} (Similarity: {assay['similarity_score']:.3f}):\n"
            assay_context += f"  {assay['content']}\n\n"
    else:
        assay_context = "🧪 EXPERIMENTAL PROTOCOL CONTEXT: No similar assays found.\n\n"
    
    # 화학 구조 컨텍스트 포맷팅
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
    
    # 분석 전략 결정
    if weights['assay'] > 0.6:
        focus_strategy = "Focus on experimental protocol patterns and assay-specific factors"
    elif weights['chemical'] > 0.6:
        focus_strategy = "Focus on chemical structure-activity relationships"
    else:
        focus_strategy = "Balance both experimental and chemical contexts equally"
    
    # 증거 강도 평가
    experimental_evidence_strength = "Strong" if len(similar_assays) >= 2 else "Moderate" if len(similar_assays) == 1 else "Weak"
    chemical_evidence_strength = "Strong" if len(similar_molecules) >= 3 else "Moderate" if len(similar_molecules) >= 1 else "Weak"
    
    return HYBRID_ANALYSIS_TEMPLATE.format(
        assay_name=parsed_query['assay_name'],
        smiles=parsed_query['smiles'],
        instruction=parsed_query['instruction'],
        assay_weight=weights['assay'],
        chemical_weight=weights['chemical'],
        assay_context=assay_context,
        chemical_context=chemical_context,
        focus_strategy=focus_strategy,
        experimental_evidence_strength=experimental_evidence_strength,
        chemical_evidence_strength=chemical_evidence_strength
    )