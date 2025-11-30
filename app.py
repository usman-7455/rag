import streamlit as st
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


# ---------------------------------------------------
# 1. Load models & data (cached)
# ---------------------------------------------------
@st.cache_resource
def load_all():

    # Load your uploaded embedding model from HF
    embedder = SentenceTransformer("zentom/embedding_model")

    # Load FAISS index
    index = faiss.read_index("index.faiss")

    # Load documents (JSONL: newline-delimited JSON)
    with open("preprocessed_documents.jsonl", "r", encoding="utf-8") as f:
        docs = [json.loads(line) for line in f if line.strip()]

    # Load local generator model (Phi-3 or any model you want)
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    generator = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    return embedder, index, docs, tokenizer, generator


embedder, index, documents, tokenizer, generator = load_all()


# ---------------------------------------------------
# 2. Retrieval function
# ---------------------------------------------------
def retrieve(query, k=5):
    q_emb = embedder.encode([query], normalize_embeddings=True)
    scores, idxs = index.search(np.array(q_emb).astype("float32"), k)
    return [documents[i] for i in idxs[0]]


# ---------------------------------------------------
# 3. Local generator
# ---------------------------------------------------
def generate_answer(question, retrieved_docs):
    context = "\n\n".join([doc["text"] for doc in retrieved_docs])

    prompt = f"""
You are a clinical question answering AI. Use ONLY the context.

--- CONTEXT ---
{context}
----------------

Question: {question}
Answer (short, clinical, factual):
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(generator.device)

    outputs = generator.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.2
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ---------------------------------------------------
# 4. UI
# ---------------------------------------------------
st.title(" Local Clinical RAG System ")

query = st.text_input("Ask something:")

if st.button("Run"):
    if not query.strip():
        st.warning("Enter a question first.")
    else:
        with st.spinner("Retrieving..."):
            chunks = retrieve(query)

        st.subheader("Retrieved Evidence")
        for i, c in enumerate(chunks):
            st.markdown(f"**Chunk {i+1}:**\n{c['text']}")
            st.markdown("---")

        with st.spinner("Generating answer..."):
            answer = generate_answer(query, chunks)

        st.subheader("Answer")
        st.write(answer)
