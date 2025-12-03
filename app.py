import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq

# ---------------------------------------------------
# 🔑 GROQ API KEY — VISIBLE (for testing ONLY!)
# ---------------------------------------------------
GROQ_API_KEY = "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"  # ← REPLACE THIS WITH YOUR ACTUAL KEY



groq_client = Groq(api_key=GROQ_API_KEY)

# ---------------------------------------------------
# 1. Load FAISS index, documents, and embedder (cached)
# ---------------------------------------------------
@st.cache_resource
def load_retrieval_assets():
    embedder = SentenceTransformer("zentom/embedding_model")
    index = faiss.read_index("index.faiss")
    with open("preprocessed_documents.jsonl", "r", encoding="utf-8") as f:
        docs = [json.loads(line) for line in f if line.strip()]
    return embedder, index, docs

embedder, index, documents = load_retrieval_assets()

# ---------------------------------------------------
# 2. Retrieval function
# ---------------------------------------------------
def retrieve(query, k=5):
    q_emb = embedder.encode([query], normalize_embeddings=True)
    scores, idxs = index.search(np.array(q_emb).astype("float32"), k)
    return [documents[i] for i in idxs[0]]

# ---------------------------------------------------
# 3. Generate answer using Groq
# ---------------------------------------------------
def generate_answer(question, retrieved_docs):
    context = "\n\n".join([doc["text"] for doc in retrieved_docs])

    prompt = f"""You are a clinical assistant. Follow these rules strictly:
1. Answer ONLY using facts explicitly stated in the "Context" below.
2. NEVER mention lab tests (e.g., ESR, CRP, AFB smear, culture), imaging findings, or symptoms unless they appear verbatim in the Context.
3. NEVER use phrases like "typical findings", "may include", or "commonly seen".
4. If the Context does not contain sufficient information, respond EXACTLY: "The provided clinical notes do not contain sufficient information."

Context:
{context}

Question: {question}

Answer:"""

    chat_completion = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.1-8b-instant",
        temperature=0.2,
        max_tokens=200,
        top_p=0.9
    )
    return chat_completion.choices[0].message.content.strip()

# ---------------------------------------------------
# 4. UI
# ---------------------------------------------------
st.title("🩺 Clinical RAG System (Groq + Llama-3)")
st.caption("Using Llama-3.1-8B via Groq API")

query = st.text_input("Ask a clinical question:")

if st.button("Run"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("🔍 Retrieving..."):
            chunks = retrieve(query)

        st.subheader("Retrieved Evidence")
        for i, c in enumerate(chunks):
            st.markdown(f"**Diagnosis:** `{c['diagnosis_category']}`")
            st.text_area(f"Note {i+1}", c["text"], height=120, key=f"note_{i}")
            st.markdown("---")

        with st.spinner("🧠 Generating answer..."):
            answer = generate_answer(query, chunks)

        st.subheader("Answer")
        st.success(answer)
