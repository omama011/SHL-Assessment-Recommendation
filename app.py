import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_resource
def load_model_and_embeddings():
    # Load model from local directory
    model = SentenceTransformer("./all-MiniLM-L6-v2")
    
    # Load metadata
    data = pd.read_csv("shl_prepackaged_solutions_detailed.csv")
    metadata = data.to_dict(orient="records")
    
    # Prepare texts for embeddings
    texts = [item["Assessment Name"] + " " + item.get("Test Type", "") for item in metadata]
    embeddings = model.encode(texts, convert_to_numpy=True)
    
    return model, embeddings, metadata

def search(query, model, embeddings, metadata, top_k=5):
    # Encode the user query
    query_embedding = model.encode([query], convert_to_numpy=True)
    
    # Compute cosine similarity
    similarity_scores = cosine_similarity(query_embedding, embeddings)[0]
    
    # Get top-k results
    top_indices = np.argsort(similarity_scores)[::-1][:top_k]
    results = [metadata[i] for i in top_indices]
    
    return results

def main():
    st.title("SHL Assessment Recommendation")
    st.write("Search for the most relevant assessments from SHL.")

    model, embeddings, metadata = load_model_and_embeddings()

    query = st.text_input("Enter a role or skill you're hiring for:", "")

    if query:
        results = search(query, model, embeddings, metadata)
        st.subheader("Top Recommendations:")

        for i, result in enumerate(results, start=1):
            st.markdown(f"### {i}. {result['Assessment Name']}")
            st.markdown(f"**Test Type**: {result.get('Test Type', 'N/A')}")
            st.markdown(f"**Target Audience**: {result.get('Target Audience', 'N/A')}")
            st.markdown(f"**Assessment Description**: {result.get('Assessment Description', 'N/A')}")
            st.markdown("---")

if __name__ == "__main__":
    main()
