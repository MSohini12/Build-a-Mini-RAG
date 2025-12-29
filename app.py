
import streamlit as st
from rag import generate_answer
import os

# Page Configuration
st.set_page_config(
    page_title="Indecimal Mini RAG",
    layout="wide"
)

st.title(" Indecimal Construction Assistant")
st.write(
    "Answers are generated **only from internal Indecimal documents** "
    "using a Retrieval-Augmented Generation (RAG) pipeline."
)
# Model Selection
model_choice = st.radio(
    "Select LLM:",
    ["Local (Phi-2)", "OpenRouter (Mistral-7B)"],
    horizontal=True
)

model_type = "local" if "Local" in model_choice else "openrouter"
MODEL_PATH = "./models/phi-2"

if model_type == "local" and not os.path.exists(MODEL_PATH):
    st.info(
        "⚠️ Local Phi-2 model is not available in this deployment. "
        "Automatically switching to OpenRouter (Mistral-7B)."
    )
    model_type = "openrouter"
# User Input

query = st.text_input(
    "Ask a question:",
    placeholder="e.g. How are contractor payments released?"
)

# Run RAG Pipeline
if query:
    with st.spinner("Retrieving documents and generating answer..."):
        result = generate_answer(query, model_type=model_type)

    # Retrieved Context
    st.subheader("📄 Retrieved Context (Ranked by Confidence)")

    for i, chunk in enumerate(result["retrieved_context"], start=1):
        with st.expander(
            f"{i}. Source: {chunk['source']} | Confidence: {chunk['confidence']}"
        ):
            st.markdown(f"**Similarity Distance:** {chunk['distance']}")
            st.write(chunk["content"])

    # Answer Section
    st.subheader("🤖 Answer")

    # Latency
    st.caption(
        f"⏱️ Response time: **{result['latency']:.2f} seconds** "
        f"({model_choice})"
    )

    # Overall Confidence
    if "overall_confidence" in result:
        st.metric(
            "📊 Answer Confidence",
            result["overall_confidence"]
        )

        if result["overall_confidence"] < 0.5:
            st.warning(
                "⚠️ Low confidence answer. The retrieved information may be incomplete."
            )
    # Streaming Answer
    answer_placeholder = st.empty()
    streamed_text = ""

    for token in result["answer"].split():
        streamed_text += token + " "
        answer_placeholder.markdown(streamed_text)
