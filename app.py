
# import streamlit as st
# from rag import generate_answer
# import os

# # Page Configuration
# st.set_page_config(
#     page_title="Indecimal Mini RAG",
#     layout="wide"
# )

# st.title(" Indecimal Construction Assistant")
# st.write(
#     "Answers are generated **only from internal Indecimal documents** "
#     "using a Retrieval-Augmented Generation (RAG) pipeline."
# )
# # Model Selection
# model_choice = st.radio(
#     "Select LLM:",
#     ["Local (Phi-2)", "OpenRouter (Mistral-7B)"],
#     horizontal=True
# )

# model_type = "local" if "Local" in model_choice else "openrouter"
# MODEL_PATH = "./models/phi-2"

# if model_type == "local" and not os.path.exists(MODEL_PATH):
#     st.info(
#         "⚠️ Local Phi-2 model is not available in this deployment. "
#         "Automatically switching to OpenRouter (Mistral-7B)."
#     )
#     model_type = "openrouter"
# # User Input

# query = st.text_input(
#     "Ask a question:",
#     placeholder="e.g. How are contractor payments released?"
# )

# # Run RAG Pipeline
# if query:
#     with st.spinner("Retrieving documents and generating answer..."):
#         result = generate_answer(query, model_type=model_type)

#     # Retrieved Context
#     st.subheader("📄 Retrieved Context (Ranked by Confidence)")

#     for i, chunk in enumerate(result["retrieved_context"], start=1):
#         with st.expander(
#             f"{i}. Source: {chunk['source']} | Confidence: {chunk['confidence']}"
#         ):
#             st.markdown(f"**Similarity Distance:** {chunk['distance']}")
#             st.write(chunk["content"])

#     # Answer Section
#     st.subheader("🤖 Answer")

#     # Latency
#     st.caption(
#         f"⏱️ Response time: **{result['latency']:.2f} seconds** "
#         f"({model_choice})"
#     )

#     # Overall Confidence
#     if "overall_confidence" in result:
#         st.metric(
#             "📊 Answer Confidence",
#             result["overall_confidence"]
#         )

#         if result["overall_confidence"] < 0.5:
#             st.warning(
#                 "⚠️ Low confidence answer. The retrieved information may be incomplete."
#             )
#     # Streaming Answer
#     answer_placeholder = st.empty()
#     streamed_text = ""

#     for token in result["answer"].split():
#         streamed_text += token + " "
#         answer_placeholder.markdown(streamed_text)

import os
import streamlit as st
from rag import generate_answer

# ====================================
# Page Configuration
# ====================================
st.set_page_config(
    page_title="Indecimal Mini RAG Chatbot",
    layout="wide"
)

st.title("🏗️ Indecimal Construction Assistant")

st.caption(
    "💬 Ask questions based **only on internal Indecimal documents**.\n\n"
    "🧹 Use **Clear Chat** to reset the conversation.\n"
    "🚪 Click **Quit Chatbot** when you're done."
)

# ====================================
# Session State Initialization
# ====================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_active" not in st.session_state:
    st.session_state.chat_active = True

# ====================================
# Sidebar Controls
# ====================================
with st.sidebar:
    st.header("🧠 Chat Controls")

    model_choice = st.radio(
        "Select LLM:",
        ["Local (Phi-2)", "OpenRouter (Mistral-7B)"],
        horizontal=False
    )

    model_type = "local" if "Local" in model_choice else "openrouter"

    # Auto fallback if Phi-2 not present
    if model_type == "local" and not os.path.exists("./models/phi-2"):
        st.warning("⚠️ Local Phi-2 not found. Falling back to OpenRouter.")
        model_type = "openrouter"

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()


    if st.button("🚪 Quit Chatbot"):
        st.session_state.chat_active = False

# ====================================
# Stop if Chat Ended
# ====================================
if not st.session_state.chat_active:
    st.success("✅ Chatbot session ended. Refresh the page to start again.")
    st.stop()

# ====================================
# Display Chat History
# ====================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ====================================
# Chat Input
# ====================================
user_query = st.chat_input("Ask a question about Indecimal...")

if user_query:
    # Show user message
    st.session_state.messages.append(
        {"role": "user", "content": user_query}
    )

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = generate_answer(
                query=user_query,
                model_type=model_type
            )

            answer = result["answer"]
            latency = result["latency"]

        st.markdown(answer)
        st.caption(f"⏱️ Response time: **{latency:.2f}s** ({model_choice})")

    # Save assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    # Optional transparency (expandable)
    with st.expander("📄 Retrieved Context (for transparency)"):
        for i, chunk in enumerate(result["retrieved_context"], start=1):
            st.markdown(
                f"**{i}. Source:** {chunk['source']}  \n"
                f"**Confidence:** {chunk['confidence']}  \n"
                f"**Distance:** {chunk['distance']}"
            )
            st.write(chunk["content"])
            st.divider()

