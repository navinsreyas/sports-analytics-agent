import time

import streamlit as st

from agent import senior_agent
from vision_tool import (
    UNSUPPORTED_MESSAGE,
    build_enriched_prompt,
    extract_sports_data,
    is_supported,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Sports Analytics Agent",
    page_icon="🏏",
    layout="centered",
)

st.title("🏏 Senior Sports Analyst")
st.markdown(
    "Ask detailed questions about **IPL Cricket** and **Premier League Football**."
)

# ---------------------------------------------------------------------------
# Sidebar — capabilities + image uploader
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("🔍 Capabilities")
    st.info("**1. Calculator (SQL):** Stats, Scores, Live Standings")
    st.info("**2. Librarian (RAG):** Rules, Definitions (from PDF)")
    st.info("**3. Detective (Graph):** Relationships, Teams Played For")
    st.info("**4. Vision (Gemini):** Read match results, tables & stats cards")

    st.divider()

    st.subheader("📷 Image Upload")
    uploaded_file = st.file_uploader(
        "Upload a sports image (optional)",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )
    st.caption(
        "Supports: match results, league tables, player stats cards, "
        "fixture lists — IPL and Premier League only."
    )

    # Read bytes with .read() BEFORE st.image() can advance the file pointer,
    # then persist them in session_state so they survive the st.chat_input rerun.
    if uploaded_file is not None:
        uploaded_file.seek(0)                          # guarantee pointer is at start
        _raw = uploaded_file.read()                    # read bytes with .read() as required
        st.session_state["_img_bytes"] = _raw
        st.session_state["_img_name"]  = uploaded_file.name
        st.image(
            _raw,                                      # pass bytes directly — avoids re-reading the file object
            caption=f"📎 {uploaded_file.name}",
            use_container_width=True,
        )
    else:
        # File was removed — clear stored bytes immediately.
        st.session_state["_img_bytes"] = None
        st.session_state["_img_name"]  = None

    st.divider()
    st.caption("Powered by LangChain, Neo4j, Neon, Groq & Gemini")

# ---------------------------------------------------------------------------
# Chat history initialisation
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay all previous messages so the history stays visible on reruns.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# When an image is in the uploader, show a gentle reminder so the user
# knows it will be included with their next question.
if uploaded_file is not None:
    st.info(
        "📷 Image ready — type your question below and it will be "
        "analysed alongside the image.",
        icon="ℹ️",
    )

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

# Pull stored image data from session_state once, before the chat handler runs.
# These are guaranteed to be set (or None) by the sidebar block above.
_img_bytes = st.session_state.get("_img_bytes")
_img_name  = st.session_state.get("_img_name")

if prompt := st.chat_input("Ask a sports question..."):

    # ---- Build the message label shown in the chat window ----
    if _img_bytes is not None:
        display_label = f"[📎 {_img_name}] {prompt}"
    else:
        display_label = prompt

    st.chat_message("user").markdown(display_label)
    st.session_state.messages.append({"role": "user", "content": display_label})

    # ---- Generate the response ----
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        try:
            full_response = None  # will be set in one of the branches below

            # ----------------------------------------------------------
            # VISION FLOW — bytes were stored in session_state by the sidebar
            # ----------------------------------------------------------
            if _img_bytes is not None:

                # Step 1: pass the already-read bytes directly to vision_tool.
                with st.spinner("Reading image..."):
                    extracted = extract_sports_data(_img_bytes)

                # Step 2: decide what to do based on the extraction result.
                if extracted is None:
                    # Gemini API itself failed → warn and fall back to text.
                    st.warning(
                        "Could not process the image (Gemini API error). "
                        "Answering based on your text question only.",
                        icon="⚠️",
                    )
                    with st.spinner("Analyzing..."):
                        full_response = senior_agent(prompt)

                elif not is_supported(extracted):
                    # Gemini said it's not a sports image.
                    full_response = UNSUPPORTED_MESSAGE

                else:
                    # Success — build an enriched prompt and pass to the agent.
                    enriched_prompt = build_enriched_prompt(prompt, extracted)
                    with st.spinner("Analyzing..."):
                        full_response = senior_agent(enriched_prompt)

            # ----------------------------------------------------------
            # TEXT-ONLY FLOW — no image, exactly as before
            # ----------------------------------------------------------
            else:
                with st.spinner("Analyzing..."):
                    full_response = senior_agent(prompt)

            # ---- Animate the response word-by-word ----
            displayed = ""
            for word in full_response.split():
                displayed += word + " "
                time.sleep(0.05)
                message_placeholder.markdown(displayed + "▌")

            message_placeholder.markdown(full_response)
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )

        except Exception as e:
            st.error(f"Error: {e}")
