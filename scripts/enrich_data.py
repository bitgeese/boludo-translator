"""
Script to enrich the phrases.csv dataset with additional context using an LLM (simulated).

Adds the following columns:
- Example Sentence (Spanish)
- Example Sentence (English)
- Connotation
- Register

This version adds predefined new terms to the existing dataset.
"""

import pandas as pd
import logging
import time
import os
import random # For simulation

# Assuming this script is run from the project root
DATA_FILE_PATH = "data/phrases.csv"
# No intermediate file needed now, we append and overwrite
# OUTPUT_FILE_PATH = "data/phrases_appended.csv"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- LLM Simulation (Keep the existing simulation logic) ---

def simulate_llm_enrichment(original: str, argentinian: str, explanation: str, formality: str) -> dict:
    """Simulates calling an LLM to get enrichment data."""
    
    # Simulate based on formality/keywords - VERY basic simulation
    connotation = "Neutral"
    register = "Standard"
    
    formality = str(formality).lower()
    explanation = str(explanation).lower()
    argentinian_lower = str(argentinian).lower()

    # Slightly more nuanced simulation based on new terms
    if argentinian_lower == "tomátela!":
        connotation = "Aggressive"
        register = "Vulgar"
    elif argentinian_lower == "zarpado":
         connotation = "Positive/Impressed"
         register = "Slang"
    elif argentinian_lower == "estar en cualquiera":
         connotation = "Negative/Dismissive"
         register = "Colloquial"
    elif argentinian_lower == "transar":
        connotation = "Informal/Suggestive"
        register = "Colloquial"
    elif argentinian_lower == "cheto/a":
        connotation = "Derogatory/Judgmental"
        register = "Slang"
    elif argentinian_lower == "trucho/a":
        connotation = "Negative/Distrustful"
        register = "Colloquial"
    elif argentinian_lower == "¿qué onda?":
        connotation = "Neutral/Inquisitive"
        register = "Colloquial"
    elif argentinian_lower == "afanar":
        connotation = "Negative/Illegal"
        register = "Slang"
    elif "vulgar" in formality or "insult" in explanation or argentinian_lower in ["boludo/a", "pelotudo", "forro", "hijo de puta"]:
        connotation = random.choice(['Derogatory', 'Aggressive', 'Vulgar'])
        register = "Vulgar"
    elif "lunfardo" in explanation or "slang" in explanation:
         connotation = random.choice(['Neutral', 'Informal'])
         register = "Slang"
    elif "casual" in formality or "informal" in explanation or "colloquial" in explanation:
         connotation = random.choice(['Neutral', 'Informal', 'Affectionate', 'Humorous'])
         register = "Colloquial"
    elif "neutral" in formality:
         connotation = "Neutral"
         register = "Standard"

    # Simple template examples - a real LLM would generate more diverse examples
    example_es = f"Ayer usé la frase '{argentinian}' con mi amigo."
    example_en = f"Yesterday I used the phrase '{argentinian}' with my friend."
    if 'greeting' in explanation or '¿qué onda?' in argentinian_lower:
        example_es = f"Le pregunté al pibe, '{argentinian}'"
        example_en = f"I asked the kid, '{argentinian}'"
    if argentinian_lower == "tomátela!":
        example_es = f"Cuando me molestó, le grité '{argentinian}'"
        example_en = f"When he bothered me, I yelled '{argentinian}'"
    if argentinian_lower == "zarpado":
        example_es = f"¡El show estuvo {argentinian}!"
        example_en = f"The show was {argentinian}!"
         
    # Simulate potential LLM inconsistencies or failures
    if random.random() < 0.02: # Simulate 2% failure rate
        return {} 

    return {
        "Example Sentence (Spanish)": example_es,
        "Example Sentence (English)": example_en,
        "Connotation": connotation,
        "Register": register
    }

# --- Main Enrichment Logic --- 
# (Modified to only process the new terms)

def enrich_new_terms_dataframe(df_new: pd.DataFrame) -> pd.DataFrame:
    """Adds enrichment columns to the *new terms* DataFrame by calling the simulated LLM."""
    enriched_rows = []
    total_rows = len(df_new)
    
    for index, row in df_new.iterrows():
        logging.info(f"Processing NEW row {index + 1}/{total_rows} ('{row['Original Phrase/Word']}')")
        
        # Simulate API call
        enrichment_result = simulate_llm_enrichment(
            row.get('Original Phrase/Word', ''),
            row.get('Argentinian Equivalent', ''),
            row.get('Explanation (Context/Usage)', ''),
            row.get('Level of Formality', '')
        )
        
        # Create a new dictionary or Series for the enriched row
        enriched_row = row.to_dict()
        enriched_row['Example Sentence (Spanish)'] = enrichment_result.get('Example Sentence (Spanish)', 'N/A - Generation Failed')
        enriched_row['Example Sentence (English)'] = enrichment_result.get('Example Sentence (English)', 'N/A - Generation Failed')
        enriched_row['Connotation'] = enrichment_result.get('Connotation', 'Unknown')
        enriched_row['Register'] = enrichment_result.get('Register', 'Unknown')
        
        enriched_rows.append(enriched_row)
        
    return pd.DataFrame(enriched_rows)

# Simulated new terms for demonstration
new_terms = [
    {"Original Phrase/Word": "Go away! (strong)", "Argentinian Equivalent": "Tomátela!", "Explanation (Context/Usage)": "Forceful way to tell someone to leave or get lost.", "Region Specificity": "Nationwide", "Level of Formality": "Vulgar"},
    {"Original Phrase/Word": "Awesome / Brilliant", "Argentinian Equivalent": "Zarpado", "Explanation (Context/Usage)": "Youth slang for something impressive, cool, or over-the-top.", "Region Specificity": "Nationwide", "Level of Formality": "Casual"},
    {"Original Phrase/Word": "To be clueless / Out of it", "Argentinian Equivalent": "Estar en cualquiera", "Explanation (Context/Usage)": "Means someone is not paying attention, is mistaken, or talking nonsense.", "Region Specificity": "Nationwide", "Level of Formality": "Casual"},
    {"Original Phrase/Word": "Police patrol car", "Argentinian Equivalent": "Patrullero", "Explanation (Context/Usage)": "Common term for a police car.", "Region Specificity": "Nationwide", "Level of Formality": "Neutral"},
    {"Original Phrase/Word": "To make out / Kiss passionately", "Argentinian Equivalent": "Transar", "Explanation (Context/Usage)": "Slang for making out, can sometimes imply more.", "Region Specificity": "Nationwide", "Level of Formality": "Casual"},
    {"Original Phrase/Word": "Rich kid / Posh person (derogatory)", "Argentinian Equivalent": "Cheto/a", "Explanation (Context/Usage)": "Slang, often derogatory, for someone perceived as posh or snobbish.", "Region Specificity": "Nationwide", "Level of Formality": "Casual"},
    {"Original Phrase/Word": "Fake / Low quality", "Argentinian Equivalent": "Trucho/a", "Explanation (Context/Usage)": "Describes something counterfeit, fake, or of poor quality.", "Region Specificity": "Nationwide", "Level of Formality": "Casual"},
    {"Original Phrase/Word": "What's wrong? / What's up?", "Argentinian Equivalent": "¿Qué onda?", "Explanation (Context/Usage)": "Very common informal greeting or inquiry, similar to 'What's up?' or 'What's the vibe?'.", "Region Specificity": "Nationwide", "Level of Formality": "Casual"},
    {"Original Phrase/Word": "Dickhead / Asshole", "Argentinian Equivalent": "Pelotudo", "Explanation (Context/Usage)": "Common strong insult, can sometimes be used very informally/jokingly between close friends.", "Region Specificity": "Nationwide", "Level of Formality": "Vulgar"},
     {"Original Phrase/Word": "To steal", "Argentinian Equivalent": "Afanar", "Explanation (Context/Usage)": "Common slang verb meaning 'to steal' or 'to rip off'.", "Region Specificity": "Nationwide", "Level of Formality": "Casual"},
]

if __name__ == "__main__":
    logging.info(f"Starting data enrichment script. Reading existing data from {DATA_FILE_PATH}")
    
    if not os.path.exists(DATA_FILE_PATH):
        logging.error(f"Data file not found at {DATA_FILE_PATH}. Exiting.")
        exit(1)
        
    try:
        # Load existing data
        df_existing = pd.read_csv(DATA_FILE_PATH)
        logging.info(f"Successfully read {len(df_existing)} existing rows.")
        
        # Prepare new terms
        df_new_raw = pd.DataFrame(new_terms)
        logging.info(f"Prepared {len(df_new_raw)} new terms for enrichment.")
        
        # Perform enrichment on new terms only
        df_new_enriched = enrich_new_terms_dataframe(df_new_raw)
        
        # Validation (basic check)
        if len(df_new_enriched) != len(df_new_raw):
            logging.error("Error during enrichment: Row count mismatch for new terms.")
            exit(1)
        failed_count = len(df_new_enriched[df_new_enriched['Connotation'] == 'Unknown'])
        if failed_count > 0:
             logging.warning(f"{failed_count} new rows had simulated generation failures.")

        # Combine existing and new enriched data
        logging.info("Appending new enriched terms to existing data.")
        df_combined = pd.concat([df_existing, df_new_enriched], ignore_index=True)
        
        # Remove potential duplicates based on 'Argentinian Equivalent' keeping the first (original if existed)
        original_count = len(df_combined)
        df_combined.drop_duplicates(subset=['Argentinian Equivalent'], keep='first', inplace=True)
        duplicates_removed = original_count - len(df_combined)
        if duplicates_removed > 0:
            logging.warning(f"Removed {duplicates_removed} duplicate rows based on 'Argentinian Equivalent'.")

        # Save combined data back to the original file
        logging.info(f"Saving combined data ({len(df_combined)} rows) back to {DATA_FILE_PATH}")
        df_combined.to_csv(DATA_FILE_PATH, index=False, encoding='utf-8')
        
        logging.info("Enrichment and appending process completed successfully.")

    except FileNotFoundError as e:
        logging.error(f"File error during processing: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True) 