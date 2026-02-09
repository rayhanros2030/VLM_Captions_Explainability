"""
Prompt generation module for LLaVA-Med integration.
Contains functions to generate prompts in different styles (descriptive, diagnostic, classification)
with optional few-shot examples.
"""

from .config import cfg


def get_prompt(style: str = "descriptive", use_few_shot: bool = False) -> str:
    """Get prompt based on style, optionally with few-shot examples."""
    
    # Few-shot examples for better accuracy
    few_shot_examples = """
Examples of correct diagnoses:
- "Well differentiated tubular adenocarcinoma" → "The histopathology image shows well-differentiated tubular adenocarcinoma with regular glandular structures."
- "Poorly differentiated adenocarcinoma, solid type" → "The histopathology image shows poorly differentiated adenocarcinoma with solid growth pattern and high-grade nuclear atypia."
- "Signet ring cell carcinoma" → "The histopathology image shows signet ring cell carcinoma characterized by cells with prominent cytoplasmic mucin vacuoles."
- "Moderately differentiated tubular adenocarcinoma" → "The histopathology image shows moderately differentiated tubular adenocarcinoma with irregular glandular structures and moderate nuclear atypia."
"""
    
    base_prompts = {
        "descriptive": "You are analyzing a histopathology image. Describe ONLY what you see in the image: the tissue architecture, cell morphology, and pathological features. Do NOT say you cannot see the image. Do NOT provide general information. Describe ONLY the specific features visible in THIS image. Write in complete sentences, not abbreviations or codes.",
        "diagnostic": "You are examining a histopathology image. You MUST provide a descriptive diagnostic classification based ONLY on what you see in the image. Write in complete English sentences, NOT abbreviations, codes, or medical notation. Do NOT say 'I cannot see the image' or provide general information. Start your response with 'The histopathology image shows' and then provide: (1) whether the tissue shows neoplastic (cancerous) changes - if cancer is present, you MUST state it clearly (do NOT say 'no cancer' or 'no evidence of cancer' if cancer is visible), (2) the specific cancer type if present (e.g., 'adenocarcinoma', 'signet ring cell carcinoma', 'carcinoma'), (3) the differentiation level using these EXACT terms: 'well differentiated', 'moderately differentiated', or 'poorly differentiated', and (4) any specific subtype features using EXACT terms: 'tubular', 'papillary', 'mucinous', 'solid', 'non-solid', 'signet ring'. Use the EXACT medical terms from the standard histopathology classification system. If you observe adenocarcinoma, state 'adenocarcinoma' explicitly. If you observe signet ring cells, state 'signet ring cell carcinoma' explicitly. IMPORTANT: Write in full descriptive sentences, NOT abbreviations like 'PD+', 'ypN0', or codes.",
        "classification": "You are examining a histopathology image. You MUST provide the specific diagnostic classification based ONLY on what you see in the image. Write in complete English sentences, NOT abbreviations, codes, or medical notation. Do NOT say 'I cannot see the image' or provide general information. Start your response with 'The histopathology image shows' and then provide the specific subtype classification using EXACT medical terminology including: the cancer type (e.g., 'adenocarcinoma', 'carcinoma'), the differentiation level using EXACT terms ('well differentiated', 'moderately differentiated', or 'poorly differentiated'), and any subtype features using EXACT terms ('tubular', 'papillary', 'signet ring', 'mucinous', 'solid', 'non-solid'). Be specific and use the EXACT medical terminology from standard histopathology classifications. IMPORTANT: Write in full descriptive sentences, NOT abbreviations or codes."
    }
    
    base_prompt = base_prompts.get(style, base_prompts["descriptive"])
    
    # Add few-shot examples if enabled
    if use_few_shot and cfg.USE_FEW_SHOT_EXAMPLES:
        return base_prompt + "\n\n" + few_shot_examples
    
    return base_prompt

