import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import json

# Load model and FAISS index
@st.cache_resource
def load_model_and_index():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index("shl_index.faiss")
    return model, index

# Load metadata
@st.cache_data
def load_metadata():
    data = pd.read_csv("shl_prepackaged_solutions_detailed.csv")
    metadata = data.to_dict(orient="records")
    return metadata

# Recommendation logic
def recommend_assessments(query, model, index, metadata, top_k=10):
    query_embedding = model.encode([query])[0].astype("float32")
    distances, indices = index.search(np.array([query_embedding]), top_k)

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx < len(metadata):
            record = metadata[idx]
            result = {
                "Assessment Name": record["Assessment Name"],
                "URL": record.get("Link", "N/A"),
                "Remote Testing Support": record.get("Remote Testing", "N/A"),
                "Adaptive/IRT Support": record.get("Adaptive/IRT", "N/A"),
                "Duration": f"{record.get('Completion Time (mins)', 'N/A')} mins",
                "Test Type": record.get("Test Type", "N/A"),
                "Similarity Score": f"{1 - dist:.4f}"
            }
            results.append(result)
    return results

# Streamlit UI
def main():
    st.set_page_config(page_title="SHL Assessment Recommender", layout="wide")
    st.title("🔍 SHL Assessment Recommendation Engine")
    st.markdown("Get the best-matching SHL assessments by describing your hiring needs below.")

    # Load everything
    model, index = load_model_and_index()
    metadata = load_metadata()

    # User input
    query = st.text_area("💬 Enter your query:", 
        placeholder="E.g. Looking to hire mid-level professionals proficient in Python, SQL and JavaScript. Max 60-minute test.")

    top_k = st.slider("🔢 Number of Recommendations", 1, 20, 10)

    if st.button("🔎 Recommend Assessments"):
        if not query.strip():
            st.warning("Please enter a valid query.")
        else:
            with st.spinner("Generating recommendations..."):
                recommendations = recommend_assessments(query, model, index, metadata, top_k=top_k)

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
                st.warning("No recommendations found.")

if __name__ == "__main__":
    main()
