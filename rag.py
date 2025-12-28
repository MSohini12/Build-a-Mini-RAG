

"""
rag.py
------
Mini RAG pipeline.

Features:
- Multi-document FAISS retrieval
- Local Hugging Face LLM (phi-2)
- Optional OpenRouter LLM (Mistral-7B)
- Strict grounding
- Clean UI output (no prompt echo)
"""

from typing import Dict
from dotenv import load_dotenv
import os
import re

from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from vector import get_retriever

# 1. Environment
load_dotenv()

HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "microsoft/phi-2")
DEVICE = "cpu"

# 2. Local Hugging Face Model
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    HF_MODEL_NAME,
    torch_dtype="auto",
    trust_remote_code=True
)

hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if DEVICE == "cuda" else -1,
    max_new_tokens=256,
    do_sample=False
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)


# 3. Prompt
chat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI assistant for a construction marketplace.\n\n"
            "CRITICAL RULES (MUST FOLLOW):\n"
            "- Use ONLY information explicitly stated in the context.\n"
            "- Do NOT use general knowledge, assumptions, or examples.\n"
            "- Do NOT list common industry factors unless they appear verbatim in the context.\n"
            "- If the context does NOT explicitly answer the question, respond EXACTLY in this format:\n\n"
            "\"The provided documents describe <what is mentioned>.\n"
            "However, they do not specify <what is missing>.\n"
            "I do not have enough information in the provided documents.\""
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
        ),
    ]
)

# 4. Retriever 
retriever = get_retriever(k=3)

def remove_questions_from_context(text: str) -> str:
    """Remove embedded question lines from retrieved context."""
    lines = text.splitlines()
    cleaned = [
        line for line in lines
        if not re.match(r"^\s*(Q:|Question:)", line, re.IGNORECASE)
    ]
    return "\n".join(cleaned)

def format_docs(docs):
    raw = "\n\n".join(doc.page_content for doc in docs)
    return remove_questions_from_context(raw)


rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | chat_prompt
    | llm
    | StrOutputParser()
)

# 5. Output Cleaning
def clean_answer(text: str) -> str:
    """Remove prompt echo and keep only the final answer."""
    parts = re.split(r"\bAnswer:\b", text, flags=re.IGNORECASE)
    if len(parts) > 1:
        text = parts[-1]

    text = re.sub(
        r"(System:|Human:|Context:|Question:).*",
        "",
        text,
        flags=re.DOTALL
    )
    return text.strip()

# 6. RAG Inference 
def generate_answer(query: str, model_type: str = "local") -> Dict:
    """
    Returns:
    - retrieved_context
    - final grounded answer
    """

    from llm_openrouter import generate_openrouter_answer

    # Retrieve documents (for transparency)
    docs = retriever.get_relevant_documents(query)

    retrieved_context = [
        {
            "source": doc.metadata.get("source"),
            "content": doc.page_content,
        }
        for doc in docs
    ]

    # ---------- Local LLM ----------
    if model_type == "local":
        raw_answer = rag_chain.invoke(query)
        answer = clean_answer(raw_answer)

        if not answer:
            answer = (
                "The provided documents describe Indecimal’s processes and policies related to this topic. "
                "However, they do not explicitly specify the information needed to answer the question. "
                "I do not have enough information in the provided documents."
            )


    # ---------- OpenRouter ----------
    elif model_type == "openrouter":
        context_text = "\n\n".join(
            f"Source: {d['source']}\n{d['content']}"
            for d in retrieved_context
        )

        prompt_text = f"""
You are an AI assistant for a construction marketplace.

Answer ONLY using the information in the context.
If the answer is not present, say:
"I do not have enough information in the provided documents."

Context:
{context_text}

Question:
{query}

Answer:
""".strip()

        answer = generate_openrouter_answer(prompt_text)

    else:
        raise ValueError("model_type must be 'local' or 'openrouter'")

    return {
        "query": query,
        "retrieved_context": retrieved_context,
        "answer": answer,
    }

