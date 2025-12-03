import gradio as gr
import json
import torch
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# ============================================================
#                Load All Files and Models
# ============================================================

# Load preprocessed documents
documents = []
with open("preprocessed_documents.jsonl", "r") as f:
    for line in f:
        documents.append(json.loads(line))

print(f"Loaded {len(documents)} documents.")

# Load FAISS Index
index = faiss.read_index("index.faiss")
print("FAISS index loaded.")

# Load Embedding Model
embedding_model = SentenceTransformer("zentom/embedding_model")
embedding_model.eval()
print("Embedding model loaded.")

# Load Phi-3 Model
tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    trust_remote_code=False,
    padding_side="left",
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

llm = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=False,
    use_safetensors=True
)

print("Phi-3 model loaded successfully.")

# ============================================================
#                       Retrieval
# ============================================================

def retrieve(query: str, k: int = 3):
    query_emb = embedding_model.encode(query, normalize_embeddings=True).astype("float32")
    scores, indices = index.search(np.expand_dims(query_emb, 0), k)

    results = []
    for i in range(k):
        idx = int(indices[0][i])
        results.append({
            "score": float(scores[0][i]),
            "document": documents[idx]
        })
    return results

# ============================================================
#               Prompt for Phi-3 (exact same logic)
# ============================================================

def build_rag_prompt(query: str, retrieved_docs: list, max_length: int = 2000):
    context_parts = []
    for i, doc in enumerate(retrieved_docs):
        note = doc['document']['text']
        context_parts.append(
            f"Document {i+1} (Diagnosis: {doc['document']['diagnosis_category']}):\n{note}\n"
        )

    context = "\n".join(context_parts)
    if len(context) > max_length:
        context = context[:max_length] + "... [truncated]"

    prompt = f"""<|system|>
You are a clinical assistant. Follow these rules strictly:
1. Answer ONLY using facts explicitly stated in the "Context" below.
2. NEVER mention lab tests, imaging, or symptoms unless they appear verbatim in the Context.
3. If the Context lacks relevant info, answer EXACTLY:
   "The provided clinical notes do not contain sufficient information."
<|end|>
<|user|>
Context:
{context}

Question: {query}
<|end|>
<|assistant|>"""

    return prompt

# ============================================================
#                 RAG Pipeline for Gradio
# ============================================================

def rag_pipeline(query):
    retrieved = retrieve(query, k=3)
    prompt = build_rag_prompt(query, retrieved)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=2048
    ).to("cuda")

    with torch.no_grad():
        output_tokens = llm.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )

    answer = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    # Extract assistant part only
    if "<|assistant|>" in answer:
        answer = answer.split("<|assistant|>")[-1].strip()

    # Build retrieved docs display
    retrieved_text = ""
    for i, r in enumerate(retrieved):
        retrieved_text += (
            f"📄 **Document {i+1} — {r['document']['diagnosis_category']}**\n"
            f"**Score:** {r['score']:.3f}\n\n"
            f"```\n{r['document']['text'][:800]}\n```\n\n"
        )

    return answer, retrieved_text

# ============================================================
#                      Gradio UI
# ============================================================

css = """
#answer-box {height: 300px; overflow: auto;}
#docs-box {height: 500px; overflow: auto;}
"""

with gr.Blocks(css=css, title="Clinical RAG System") as demo:

    gr.Markdown("""
    # 🏥 RAG for Diagnostic Reasoning (DiReCT)
    ### Retrieval-Augmented Clinical Question Answering  
    """)

    query = gr.Textbox(
        label="Enter clinical query",
        placeholder="Example: What symptoms appear in COPD exacerbation?",
    )

    run_button = gr.Button("Run RAG")

    answer_box = gr.Textbox(label="Generated Answer", lines=10, elem_id="answer-box")
    docs_box = gr.Markdown(elem_id="docs-box", label="Retrieved Clinical Notes")

    run_button.click(
        fn=rag_pipeline,
        inputs=query,
        outputs=[answer_box, docs_box]
    )

demo.launch()
