import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Load model, embeddings and metadata
@st.cache_resource
def load_model_and_embeddings():
    model = SentenceTransformer("./all-MiniLM-L6-v2")  # Load from local folder
    data = pd.read_csv("shl_prepackaged_solutions_detailed.csv")
    metadata = data.to_dict(orient="records")
    texts = [item["Assessment Name"] + " " + item.get("Test Type", "") for item in metadata]
    embeddings = model.encode(texts, convert_to_numpy=True)
    return model, embeddings, metadata

# Recommend assessments using cosine similarity
def recommend_assessments(query, model, embeddings, metadata, top_k=10):
    query_embedding = model.encode(query, convert_to_numpy=True, normalize_embeddings=True)

    # Cosine similarity = dot product since embeddings are normalized
    similarity_scores = np.dot(embeddings, query_embedding)

    # Get indices of top_k scores
    top_indices = similarity_scores.argsort()[::-1][:top_k]

    # Build result list
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

# Streamlit app
def main():
    st.set_page_config(page_title="SHL Assessment Recommender", layout="wide")
    st.title("🔍 SHL Assessment Recommendation Engine")
    st.markdown("Get the best-matching SHL assessments by describing your hiring needs below.")

    # Load model and data
    model, embeddings, metadata = load_model_and_embeddings()

    # User input
    query = st.text_area("💬 Enter your query:", 
        placeholder="E.g. Looking to hire mid-level professionals with Python, SQL, and communication skills.")

    top_k = st.slider("🔢 Number of Recommendations", 1, 20, 10)

    if st.button("🔎 Recommend Assessments"):
        if not query.strip():
            st.warning("Please enter a valid query.")
        else:
            with st.spinner("Finding best matches..."):
                recommendations = recommend_assessments(query, model, embeddings, metadata, top_k=top_k)

            if recommendations:
                st.success(f"Top {top_k} Recommendations:")
                for i, rec in enumerate(recommendations, 1):
                    st.markdown(f"### {i}. {rec['Assessment Name']}")
                    st.markdown(f"- **URL:** {rec['URL']}")
                    st.markdown(f"- **Remote Testing:** {rec['Remote Testing Support']}")
                    st.markdown(f"- **Adaptive/IRT:** {rec['Adaptive/IRT Support']}")
                    st.markdown(f"- **Duration:** {rec['Duration']}")
                    st.markdown(f"- **Test Type:** {rec['Test Type']}")
                    st.markdown(f"- **Similarity Score:** {rec['Similarity Score']}")
                    st.markdown("---")
            else:
                st.warning("No matching assessments found.")

if __name__ == "__main__":
    main()
