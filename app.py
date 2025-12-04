import re
import time
import json
from typing import List, Dict, Optional, Tuple

import streamlit as st
import google.generativeai as genai

# ---------------------------------------------------------
# CONFIGURATION & UI
# ---------------------------------------------------------
st.set_page_config(page_title="AI Peer Review & Repair (chunking+temps)", layout="wide")

st.title("Agentic Peer Review — chunking & deterministic temps")

# Sidebar: API key + iterations + temperatures
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.secrets.get("GEMINI_API_KEY") or st.text_input(
        "Enter Gemini API Key:", type="password", help="Prefer using st.secrets in deployed apps."
    )
    iterations = st.number_input("Referee-Author Iterations:", min_value=1, max_value=5, value=1)

    st.markdown("### Temperature settings (deterministic per agent)")
    novelty_temp = st.slider("Novelty check temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
    org_temp = st.slider("Organization check temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
    proof_temp = st.slider("Proof-checker temperature", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
    narrative_temp = st.slider("Narrative writer temperature", min_value=0.0, max_value=1.0, value=0.4, step=0.01)
    prooffix_temp = st.slider("Proof fixer temperature", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
    integrator_temp = st.slider("Integrator temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.01)

    st.info("Notes:\n- Use very low temps (0.0–0.1) for proof-related agents to favor determinism.\n- Narrative agents can use higher temps for stylistic variety.")

if not api_key:
    st.warning("Enter your API Key (or set it as GEMINI_API_KEY in st.secrets) to begin.")
    st.stop()

# Configure client
try:
    genai.configure(api_key=api_key)
except Exception as e:
    st.error(f"Failed to configure genai client: {e}")
    st.stop()

# ---------------------------------------------------------
# Robust ask_model wrapper (retry + temperature)
# ---------------------------------------------------------
def ask_model(system_instruction: str, prompt: str, temperature: float = 0.3,
              max_retries: int = 3, timeout_s: int = 30) -> str:
    """
    Wrapper for genai model calls with retries. Returns raw text.
    Keep the wrapper minimal but robust: retries and error messages.
    """
    model = genai.GenerativeModel(model_name="gemini-2.5-flash", system_instruction=system_instruction)

    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(temperature=temperature)
            )
            # response.text is expected; fall back defensively
            text = getattr(response, "text", None) or str(response)
            return text
        except Exception as e:
            backoff = 2 ** attempt
            time.sl
