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
    """Simulates calling an LLM to get enrichment data with improved example sentences."""
    
    # --- Determine Connotation and Register (Keep previous logic) ---
    connotation = "Neutral"
    register = "Standard"
    formality = str(formality).lower()
    explanation_lower = str(explanation).lower()
    argentinian_lower = str(argentinian).lower()
    original_lower = str(original).lower()

    # (Keep the existing logic for determining connotation and register...)
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
    elif "vulgar" in formality or "insult" in explanation_lower or argentinian_lower in ["boludo/a", "pelotudo", "forro", "hijo de puta"]:
        connotation = random.choice(['Derogatory', 'Aggressive', 'Vulgar'])
        register = "Vulgar"
    elif "lunfardo" in explanation_lower or "slang" in explanation_lower:
         connotation = random.choice(['Neutral', 'Informal'])
         register = "Slang"
    elif "casual" in formality or "informal" in explanation_lower or "colloquial" in explanation_lower:
         connotation = random.choice(['Neutral', 'Informal', 'Affectionate', 'Humorous'])
         register = "Colloquial"
    elif "neutral" in formality:
         connotation = "Neutral"
         register = "Standard"

    # --- Generate More Diverse Example Sentences ---
    example_es = f"No se me ocurre un buen ejemplo para '{argentinian}'." # Default fallback
    example_en = f"I can't think of a good example for '{argentinian}'."

    # Greetings & Questions
    if "greeting" in explanation_lower or "how are you" in original_lower or "what's up" in original_lower or argentinian_lower == "¿qué onda?":
        templates_es = [
            f"Cuando lo vi, le dije: '{argentinian}'", 
            f"Me cruzé con María y me preguntó '{argentinian}'",
            f"Entré a la tienda y saludé con un '{argentinian}'"
        ]
        templates_en = [
            f"When I saw him, I said: '{argentinian}'",
            f"I ran into Maria and she asked me '{argentinian}'",
            f"I entered the shop and greeted with an '{argentinian}'"
        ]
        idx = random.randrange(len(templates_es))
        example_es, example_en = templates_es[idx], templates_en[idx]
    # Farewells
    elif "goodbye" in original_lower or "farewell" in explanation_lower:
        example_es = f"Al final de la charla, nos despedimos con un '{argentinian}'."
        example_en = f"At the end of the chat, we said goodbye with a '{argentinian}'."
    # Insults / Vulgar terms
    elif register == "Vulgar" and connotation != "Neutral":
        templates_es = [
            f"Se enojó y me gritó: '¡Sos un {argentinian}!'",
            f"No puedo creer lo {argentinian} que fue.",
            f"Dejá de hacerte el {argentinian} y ayudame."
        ]
        templates_en = [
            f"He got angry and yelled at me: 'You're such a {argentinian}!'",
            f"I can't believe how {argentinian} he was.",
            f"Stop being a {argentinian} and help me."
        ]
        # Adjust for specific vulgar terms
        if argentinian_lower == "tomátela!":
             templates_es = [f"Estaba molestando, así que le dije '{argentinian}'"]
             templates_en = [f"He was bothering me, so I told him '{argentinian}'"]
        elif "concha" in argentinian_lower or "puta" in argentinian_lower or "mierda" in argentinian_lower or "carajo" in argentinian_lower:
             templates_es = [f"Se me cayó todo al piso, ¡{argentinian}!", f"¡{argentinian}, qué bronca tengo!"]
             templates_en = [f"Everything fell on the floor, {argentinian}!", f"{argentinian}, I'm so angry!"]
             
        idx = random.randrange(len(templates_es))
        example_es, example_en = templates_es[idx], templates_en[idx]
    # Slang/Colloquial Nouns (money, work, place, thing)
    elif register in ["Slang", "Colloquial"] and ("money" in original_lower or "work" in original_lower or "party" in original_lower or "nightclub" in original_lower or "police" in original_lower or "mess" in explanation_lower or "food" in original_lower or "drink" in original_lower or "car" in original_lower):
        templates_es = [
            f"Necesito conseguir más {argentinian} para el finde.",
            f"Mañana tengo mucho {argentinian}, qué fiaca.",
            f"¿Vamos al {argentinian} esta noche?",
            f"¡Qué {argentinian} se armó en la calle!"
        ]
        templates_en = [
            f"I need to get more {argentinian} for the weekend.",
            f"Tomorrow I have a lot of {argentinian}, what a drag.",
            f"Should we go to the {argentinian} tonight?",
            f"What a {argentinian} started in the street!"
        ]
        idx = random.randrange(len(templates_es))
        example_es, example_en = templates_es[idx], templates_en[idx]
    # Slang/Colloquial Adjectives/Adverbs (cool, fake, bad, good, easy, difficult)
    elif register in ["Slang", "Colloquial"] and ("cool" in original_lower or "awesome" in original_lower or "brilliant" in original_lower or "fake" in original_lower or "lousy" in original_lower or "bad" in original_lower or "easy" in original_lower or "difficult" in original_lower or "drunk" in original_lower or "tired" in original_lower or "clueless" in original_lower):
        templates_es = [
            f"Ese bar nuevo está {argentinian}.",
            f"Compré un reloj {argentinian} en la feria.",
            f"El examen fue {argentinian}, no sé si aprobé.",
            f"Anoche terminé re {argentinian} después de la fiesta.",
            f"Me parece que estás medio {argentinian}, prestá atención."
        ]
        templates_en = [
            f"That new bar is {argentinian}.",
            f"I bought a {argentinian} watch at the fair.",
            f"The exam was {argentinian}, I don't know if I passed.",
            f"Last night I ended up really {argentinian} after the party.",
            f"I think you're a bit {argentinian}, pay attention."
        ]
        # Adjust for zarpado
        if argentinian_lower == "zarpado":
            templates_es = [f"¡El recital estuvo {argentinian}!", f"¡Qué {argentinian} lo que hiciste!"]
            templates_en = [f"The concert was {argentinian}!", f"What you did was {argentinian}!"]

        idx = random.randrange(len(templates_es))
        example_es, example_en = templates_es[idx], templates_en[idx]
    # Slang/Colloquial Verbs (steal, flirt, hang out, work, mess up, understand)
    elif register in ["Slang", "Colloquial"] and ("steal" in original_lower or "flirt" in original_lower or "hang out" in original_lower or "work" in original_lower or "mess up" in original_lower or "understand" in original_lower or "make out" in original_lower or "relax" in original_lower or "pay attention" in original_lower ):
        templates_es = [
            f"Anoche fuimos a {argentinian} con los pibes.",
            f"Tené cuidado que no te {argentinian} el celular.",
            f"Estuvo tratando de {argentinian} toda la noche.",
            f"Mañana tengo que {argentinian} todo el día.",
            f"No te preocupes, yo {argentinian} lo que dijiste.",
            f"Me mandé una macana, la {argentinian} mal."
        ]
        templates_en = [
            f"Last night we went to {argentinian} with the guys.",
            f"Be careful they don't {argentinian} your phone.",
            f"He was trying to {argentinian} all night.",
            f"Tomorrow I have to {argentinian} all day.",
            f"Don't worry, I {argentinian} what you said.",
            f"I made a mistake, I {argentinian} badly."
        ]
        idx = random.randrange(len(templates_es))
        example_es, example_en = templates_es[idx], templates_en[idx]
    # Default / Standard terms
    else:
        # Keep the slightly more varied default from before
        example_es = f"Ayer usé la frase '{argentinian}' con mi amigo."
        example_en = f"Yesterday I used the phrase '{argentinian}' with my friend."
        if 'money' in original_lower:
             example_es = f"Necesito más '{argentinian}' para comprar eso."
             example_en = f"I need more '{argentinian}' to buy that."
         
    # Simulate potential LLM inconsistencies or failures (Keep this)
    if random.random() < 0.02: # Simulate 2% failure rate
        return {
            "Example Sentence (Spanish)": "N/A - Generation Failed",
            "Example Sentence (English)": "N/A - Generation Failed",
            "Connotation": connotation,
            "Register": register
        } 

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