#   Imports  
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, logging
import torch
from IPython.display import display, Markdown

# Silence verbose logs
logging.set_verbosity_error()

#   PDF Text Extraction  
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

#   Chunking  
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

#   Embedding  
def embed_chunks(chunks, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    return embeddings, model

#   FAISS Indexing  
def store_embeddings(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

#   Retrieval  
def search_similar_chunks(query, index, chunks, embed_model, k=3):
    query_vec = embed_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, k)
    return [chunks[i] for i in indices[0]]

#   Load LLM  
def load_llm(model_name="microsoft/phi-1_5"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

#   Answer Generation  
def generate_answer(query, context_chunks, tokenizer, model):
    context = "\n".join(context_chunks)
    prompt = f"""You are a helpful study assistant. Use the following notes to answer the question.

            Notes:
            {context}

            Question: {query}
            Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Just return the part after "Answer:"
    return answer.split("Answer:")[-1].strip()

#   Output  
def show_answer(answer):
    print("\n Answer:\n")
    print(answer)

#   MAIN  
if __name__ == "__main__":
    # 1. Load PDF
    pdf_path = input("Enter path to PDF: ").strip()
    raw_text = extract_text_from_pdf(pdf_path)
    print(f"Extracted {len(raw_text)} characters")

    # 2. Chunk
    chunks = chunk_text(raw_text)
    print(f"Chunked into {len(chunks)} segments")

    # 3. Embed
    embeddings, embed_model = embed_chunks(chunks)
    print(f"Embeddings created")

    # 4. Index
    index = store_embeddings(embeddings)
    print("FAISS index ready")

    # 5. Load LLM
    tokenizer, llm_model = load_llm()
    print("LLM loaded")

    # 6. Ask questions!
    while True:
        query = input("\n Ask a question (or type 'exit'): ").strip()
        if query.lower() == 'exit':
            break

        top_chunks = search_similar_chunks(query, index, chunks, embed_model)

        # Fallback if nothing meaningful is found
        if not top_chunks or len(" ".join(top_chunks)) < 10:
            print("\n Note not available in the uploaded materials. Here's a general answer:\n")
        else:
            answer = generate_answer(query, top_chunks, tokenizer, llm_model)
            try:
                show_answer(answer)  # For Jupyter
            except:
                print("\n Answer:\n")
                print(answer)
