# embeddings.py

import json
from sentence_transformers import SentenceTransformer

def load_data(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    texts = []
    ids = []
    
    for entry in data:
        # Since there's no loctype field, we'll process all entries as states
        # They all seem to be state-level data based on the locationName values
        
        location_name = entry.get('locationName', 'Unknown')
        
        # Extract nested data safely
        area_data = entry.get('area', {})
        total_area = None
        if isinstance(area_data, dict) and 'total' in area_data:
            if isinstance(area_data['total'], dict):
                total_area = area_data['total'].get('totalArea')
            else:
                total_area = area_data.get('total')
        
        rainfall_data = entry.get('rainfall', {})
        rainfall_total = None
        if isinstance(rainfall_data, dict):
            rainfall_total = rainfall_data.get('total')
        
        recharge_data = entry.get('rechargeData', {})
        recharge_total = None
        if isinstance(recharge_data, dict) and 'total' in recharge_data:
            if isinstance(recharge_data['total'], dict):
                recharge_total = recharge_data['total'].get('total')
            else:
                recharge_total = recharge_data.get('total')
        
        draft_data = entry.get('draftData', {})
        draft_total = None
        if isinstance(draft_data, dict) and 'total' in draft_data:
            if isinstance(draft_data['total'], dict):
                draft_total = draft_data['total'].get('total')
            else:
                draft_total = draft_data.get('total')
        
        stage_extraction = entry.get('stageOfExtraction', {})
        stage_total = None
        if isinstance(stage_extraction, dict):
            stage_total = stage_extraction.get('total')
        
        category = entry.get('category', 'Unknown')
        
        # Create text representation
        text = (
            f"District: {location_name}. "
            f"Area: {total_area}. "
            f"Rainfall: {rainfall_total}. "
            f"Recharge: {recharge_total}. "
            f"Draft: {draft_total}. "
            f"Stage of extraction: {stage_total}. "
            f"Category: {category}."
        )
        
        texts.append(text)
        # Use locationUUID as ID, or fallback to a generated ID
        entry_id = entry.get('locationUUID')
        if entry_id is None or entry_id == "":
            # Generate a safe ID using the location name
            safe_name = location_name.replace(" ", "_").replace("-", "_").lower()
            entry_id = f"district_{safe_name}_{len(ids)}"
        ids.append(str(entry_id))  # Ensure it's always a string
    
    return ids, texts

def create_embeddings(texts):
    model = SentenceTransformer('all-mpnet-base-v2')
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings

if __name__ == "__main__":
    json_file = "states/KERALA.json"
    ids, texts = load_data(json_file)
    
    # Print first few texts to verify
    print(f"📝 Sample processed texts:")
    for i, text in enumerate(texts[:3]):
        print(f"  {i+1}. {text}")
    
    embeddings = create_embeddings(texts)
    print(f"✅ Created embeddings for {len(texts)} states.")