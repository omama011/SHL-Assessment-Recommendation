import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_resource
def load_model_and_embeddings():
    # Load model & tokenizer from local directory
    tokenizer = AutoTokenizer.from_pretrained("./all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("./all-MiniLM-L6-v2")

    # Load metadata
    data = pd.read_csv("shl_prepackaged_solutions_detailed.csv")
    metadata = data.to_dict(orient="records")

    # Create embeddings
    texts = [item["Assessment Name"] + " " + str(item.get("Test Type", "")) for item in metadata]
    embeddings = get_embeddings(texts, tokenizer, model)

    return tokenizer, model, embeddings, metadata

def get_embeddings(texts, tokenizer, model):
    # Create embeddings for a list of texts
    model.eval()
    embeddings = []

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            outputs = model(**inputs)
            # Use mean pooling on last hidden state
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            embeddings.append(embedding)

    return np.array(embeddings)

def search(query, tokenizer, model, embeddings, metadata, top_k=5):
    query_embedding = get_embeddings([query], tokenizer, model)[0]
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    top_indices = similarities.argsort()[::-1][:top_k]
    return [metadata[i] for i in top_indices]

def main():
    st.title("SHL Assessment Recommendation App")

    tokenizer, model, embeddings, metadata = load_model_and_embeddings()

    query = st.text_input("Enter a role, skill, or keyword:")

    if query:
        results = search(query, tokenizer, model, embeddings, metadata)
        st.subheader("Top Recommendations:")
        for i, result in enumerate(results, 1):
            st.markdown(f"### {i}. {result['Assessment Name']}")
            st.markdown(f"**Test Type**: {result.get('Test Type', 'N/A')}")
            st.markdown(f"**Target Audience**: {result.get('Target Audience', 'N/A')}")
            st.markdown(f"**Description**: {result.get('Assessment Description', 'N/A')}")
            st.markdown("---")

if __name__ == "__main__":
    main()
