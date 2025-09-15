# insert_data.py

from embeddings import load_data, create_embeddings
from pinecone_setup import initialize_pinecone

def insert_embeddings(index, ids, texts, embeddings):
    # Insert embeddings in batches for better performance
    batch_size = 100
    
    for i in range(0, len(embeddings), batch_size):
        batch_ids = ids[i:i + batch_size]
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = embeddings[i:i + batch_size]
        
        # Prepare vectors for upsert
        vectors = []
        for j, vector in enumerate(batch_embeddings):
            metadata = {
                "text": batch_texts[j]
            }
            vectors.append((batch_ids[j], vector.tolist(), metadata))
        
        # Upsert the batch
        index.upsert(vectors)
        print(f"✅ Inserted batch {i//batch_size + 1}: {len(vectors)} entries")
    
    print(f"🎉 Successfully inserted {len(embeddings)} total entries into Pinecone.")

if __name__ == "__main__":
    API_KEY = ""
    json_file = "output/india.json"
    
    print("🚀 Starting data insertion pipeline...")
    
    # Initialize Pinecone
    index = initialize_pinecone(API_KEY)
    
    # Load and process data
    print("📂 Loading data...")
    ids, texts = load_data(json_file)
    print(f"📊 Loaded {len(texts)} entries")
    
    # Create embeddings
    print("🧠 Creating embeddings...")
    embeddings = create_embeddings(texts)
    
    # Insert into Pinecone
    print("📤 Inserting into Pinecone...")
    insert_embeddings(index, ids, texts, embeddings)