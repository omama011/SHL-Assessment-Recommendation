from fastapi import FastAPI, Query
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

app = FastAPI(title="SHL Assessment Recommender API")

# Load model and data
model = SentenceTransformer("./all-MiniLM-L6-v2")
data = pd.read_csv("shl_prepackaged_solutions_detailed.csv")
metadata = data.to_dict(orient="records")

# Precompute embeddings
texts = [
    f"{item.get('Assessment Name', '')} {item.get('Test Type', '')} {item.get('Remote Testing', '')} {item.get('Adaptive/IRT', '')}"
    for item in metadata
]
embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

# Recommendation function
def recommend_assessments(query, model, embeddings, metadata, top_k=10):
    query_embedding = model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
    similarity_scores = np.dot(embeddings, query_embedding)
    top_indices = similarity_scores.argsort()[::-1][:top_k]

    results = []
    for idx in top_indices:
        record = metadata[idx]
        result = {
            "Assessment Name": record.get("Assessment Name", "N/A"),
            "URL": record.get("Link", "N/A"),
            "Remote Testing Support": record.get("Remote Testing", "N/A"),
            "Adaptive/IRT Support": record.get("Adaptive/IRT", "N/A"),
            "Duration": f"{record.get('Completion Time (mins)', 'N/A')} mins",
            "Test Type": record.get("Test Type", "N/A"),
            "Similarity Score": f"{similarity_scores[idx]:.4f}"
        }
        results.append(result)
    return results

# ✅ GET endpoint version
@app.get("/recommend")
def recommend_get(query: str = Query(..., description="Your search query"), top_k: int = Query(10, ge=1, le=50)):
    results = recommend_assessments(query, model, embeddings, metadata, top_k=top_k)
    return {"results": results}
