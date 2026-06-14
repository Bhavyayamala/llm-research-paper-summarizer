import streamlit as st
import fitz
import chromadb

from transformers import pipeline
from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Research Paper Summarizer",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Research Paper Summarizer & RAG Chatbot")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.header("About")
    st.write(
        """
        Upload a research paper PDF.

        Features:
        - Automatic Summary
        - Semantic Search
        - Question Answering (RAG)
        """
    )

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

@st.cache_resource
def load_models():

    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn"
    )

    embed_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    qa_model = pipeline(
        "text2text-generation",
        model="google/flan-t5-base"
    )

    return summarizer, embed_model, qa_model


summarizer, embed_model, qa_model = load_models()

# --------------------------------------------------
# CHUNKING FUNCTION
# --------------------------------------------------

def split_text(text):

    chunk_size = 500
    overlap = 100

    chunks = []

    for i in range(
        0,
        len(text),
        chunk_size - overlap
    ):
        chunks.append(
            text[i:i + chunk_size]
        )

    return chunks

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Research Paper PDF",
    type=["pdf"]
)

if uploaded_file:

    with st.spinner("Processing PDF..."):

        # Save uploaded file

        pdf_path = "uploaded_paper.pdf"

        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        # Extract text

        doc = fitz.open(pdf_path)

        text = ""

        for page in doc:
            text += page.get_text()

        total_pages = len(doc)

        doc.close()

        # Summary

        summary = summarizer(
            text[:3000],
            max_length=180,
            min_length=60,
            do_sample=False
        )[0]["summary_text"]

        # Chunking

        chunks = split_text(text)

        # ChromaDB

        client = chromadb.Client()

        try:
            client.delete_collection("paper_chunks")
        except:
            pass

        collection = client.create_collection(
            name="paper_chunks"
        )

        for i, chunk in enumerate(chunks):

            embedding = embed_model.encode(
                chunk
            ).tolist()

            collection.add(
                ids=[str(i)],
                embeddings=[embedding],
                documents=[chunk]
            )

    st.success("PDF Processed Successfully!")

    # --------------------------------------------------
    # METRICS
    # --------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Pages",
            total_pages
        )

    with col2:
        st.metric(
            "Chunks Created",
            len(chunks)
        )

    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------

    st.subheader("📌 Paper Summary")

    st.write(summary)

    st.divider()

    # --------------------------------------------------
    # QUESTION ANSWERING
    # --------------------------------------------------

    st.subheader("💬 Ask Questions")

    question = st.text_input(
        "Ask anything about the paper"
    )

    if question:

        query_embedding = embed_model.encode(
            question
        ).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            include=["documents", "distances"]
        )

        context = "\n".join(
            results["documents"][0]
        )

        context = context[:1200]

        prompt = f"""
You are an expert research paper assistant.

Answer ONLY using the provided context.

Give a concise answer in 2-4 sentences.

If the answer is not available in the context, reply:

Information not found in retrieved context.

Context:
{context}

Question:
{question}

Answer:
"""

        answer = qa_model(
            prompt,
            max_new_tokens=120,
            do_sample=False
        )

        st.subheader("Answer")

        st.write(
            answer[0]["generated_text"]
        )

        with st.expander("Retrieved Context"):
            st.write(context)

        with st.expander("Retrieved Chunk IDs"):
            st.write(results["ids"][0])

        with st.expander("Distances"):
            st.write(results["distances"][0])