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
            time.sleep(backoff)
            last_err = e
    # After retries
    return f"ERROR: failed to call model after {max_retries} attempts. Last error: {last_err}"

# ---------------------------------------------------------
# Chunking utilities
# ---------------------------------------------------------
def split_preamble_and_body(tex: str) -> Tuple[str, str]:
    """
    Splits a LaTeX file into preamble (before \\begin{document}) and body (including \\begin{document}...).
    If no \\begin{document} is found, treat everything as body and preamble empty.
    """
    m = re.search(r"\\begin\{document\}", tex)
    if not m:
        return "", tex
    idx = m.start()
    preamble = tex[:idx]
    body = tex[idx:]
    return preamble, body

def split_by_section(body: str) -> List[Dict]:
    """
    Splits body into chunks by top-level \section or \section* occurrences.
    Returns a list of dict: { 'heading': heading_text_or_None, 'chunk': chunk_text }
    Keeps leading content before first \section as its own chunk.
    """
    # Find all \section occurrences and their positions (include the command line)
    pattern = re.compile(r"(\\section\*?\{.*?\})", re.S)
    parts = pattern.split(body)
    chunks = []
    # parts alternates between content and matched section header: [lead, header1, content1, header2, content2, ...]
    if not parts:
        return [{"heading": None, "chunk": body}]
    lead = parts[0]
    if lead.strip():
        chunks.append({"heading": None, "chunk": lead})
    # Now iterate pairs
    for i in range(1, len(parts), 2):
        header = parts[i]
        content = parts[i + 1] if i + 1 < len(parts) else ""
        full_chunk = header + content
        chunks.append({"heading": header, "chunk": full_chunk})
    return chunks

THEOREM_ENVS = ["theorem", "lemma", "proposition", "corollary", "claim", "conjecture", "definition"]

def extract_envs(tex: str, envs: List[str] = THEOREM_ENVS) -> List[Dict]:
    """
    Extract all occurrences of the specified environments.
    Returns list of dict {env, content, label (may be None), start, end}
    """
    results = []
    for env in envs:
        # non-greedy match from begin to end
        pattern = re.compile(rf"(\\begin\{{{env}\}.*?\\end\{{{env}\}})", re.S)
        for m in pattern.finditer(tex):
            block = m.group(1)
            start, end = m.start(1), m.end(1)
            # try to find label inside block
            lab_m = re.search(r"\\label\{([^}]+)\}", block)
            label = lab_m.group(1) if lab_m else None
            results.append({"env": env, "content": block, "label": label, "start": start, "end": end})
    # Sort by start position to preserve order
    results.sort(key=lambda x: x["start"])
    return results

# ---------------------------------------------------------
# Referee Agents (with deterministic temps)
# ---------------------------------------------------------
def agent_referee_novelty(tex_content: str, temperature: float = 0.2) -> str:
    """
    If doc is large, chunk by sections and ask the model per chunk, then aggregate.
    """
    preamble, body = split_preamble_and_body(tex_content)
    sections = split_by_section(body)
    assembled_reports = []
    sys_prompt = """
You are a Senior Editor at a top mathematics journal. For the provided chunk, evaluate:
1) Novelty & significance: Is this chunk presenting novel results or just incremental?
2) Any clear fatal flaws that would make the submission unsuitable.
Output a short JSON object with keys: { "status": "PASS"|"FAIL", "summary": "<1-2 sentences>" }
Only output valid JSON for each chunk.
"""
    # If the doc is small, send whole body
    if len(body) < 8000:
        prompt = f"Evaluate novelty for the following document chunk:\n\n{body}"
        return ask_model(sys_prompt, prompt, temperature=temperature)
    # Otherwise chunk
    for idx, sec in enumerate(sections):
        prompt = f"Chunk {idx+1}/{len(sections)}:\n{sec['chunk']}"
        resp = ask_model(sys_prompt, prompt, temperature=temperature)
        assembled_reports.append(f"--- CHUNK {idx+1} ---\n{resp}")
    return "\n".join(assembled_reports)

def agent_referee_org(tex_content: str, temperature: float = 0.2) -> str:
    """
    Organization check — chunk by sections for long docs.
    """
    preamble, body = split_preamble_and_body(tex_content)
    sections = split_by_section(body)
    sys_prompt = """
You are a Referee. For the chunk provided, list organizational weaknesses, unclear transitions, missing background, or places where definitions should be earlier.
Output numbered bullets or a JSON array of issues.
"""
    if len(body) < 8000:
        return ask_model(sys_prompt, f"Organization critique:\n\n{body}", temperature=temperature)

    issues = []
    for idx, sec in enumerate(sections):
        resp = ask_model(sys_prompt, f"Section chunk {idx+1}:\n\n{sec['chunk']}", temperature=temperature)
        issues.append(f"--- CHUNK {idx+1} ---\n{resp}")
    return "\n".join(issues)

def agent_referee_proofs(tex_content: str, temperature: float = 0.05) -> str:
    """
    Proof checker: extract each theorem-like environment and check it individually.
    Returns aggregated report mapping each env (by label or index) to the model response.
    """
    env_blocks = extract_envs(tex_content)
    if not env_blocks:
        # no theorem-like environments found: optionally run a lighter check on body
        sys_prompt = "You are a Math Referee. There are no theorem-like environments. Check for any explicit 'proof' environments and comment."
        return ask_model(sys_prompt, f"Document body:\n\n{tex_content}", temperature=temperature)

    reports = []
    sys_prompt = """
You are a Math Referee. For the provided single theorem/lemma/proposition block:
1. Identify the theorem statement (confirm it is well-posed).
2. Check the correctness of the proof step-by-step.
3. If flawed, explain exactly WHERE and WHY the logic breaks, and if possible give a corrected sketch.
Return a structured text: start with "ENV_LABEL: <label or INDEX>" then "REPORT:" followed by your findings.
Use low temperature for determinism.
"""
    for idx, b in enumerate(env_blocks):
        label = b["label"] or f"IDX_{idx+1}"
        header = f"ENV_LABEL: {label} (env={b['env']})"
        prompt = f"{header}\n\nBlock content:\n{b['content']}"
        resp = ask_model(sys_prompt, prompt, temperature=temperature)
        reports.append(f"{header}\n{resp}")
    return "\n\n".join(reports)

# ---------------------------------------------------------
# Author Agents (kept similar, using configured temps)
# ---------------------------------------------------------
def agent_author_narrative(full_report: str, tex_content: str, temperature: float = 0.4) -> str:
    sys_prompt = """
You are an Expert Mathematical Writer. Read the referee report and the original text. Rewrite narrative parts: Introduction, Conclusion, and transitions to improve clarity and flow. Output only the improved narrative sections in LaTeX-compatible text (not the whole document).
"""
    prompt = f"Referee Report:\n{full_report}\n\nOriginal Document:\n{tex_content}"
    return ask_model(sys_prompt, prompt, temperature=temperature)

def agent_author_proof_fixer(full_report: str, tex_content: str, temperature: float = 0.05) -> str:
    """
    Fix proofs individually. We will feed the referee proof report; the agent should return one of:
    - NO_CHANGES
    - or a JSON-like list of patches: [{"label": "label", "replacement": "<latex>"}...]
    For simplicity we parse the textual response.
    """
    sys_prompt = """
You are a mathematician specialized in fixing proofs. Read the referee report and the original text. For each theorem/lemma marked as flawed, output a JSON list of patches:
[
  {"label": "<label or IDX_n>", "replacement": "<full \\begin{theorem}...\\end{theorem} text with corrected proof>"},
  ...
]
If there are no changes, output the single token: NO_CHANGES
Only output valid JSON or NO_CHANGES.
"""
    prompt = f"Referee Report:\n{full_report}\n\nOriginal Document:\n{tex_content}"
    resp = ask_model(sys_prompt, prompt, temperature=temperature)
    return resp

def agent_author_integrator(narrative_fix: str, proof_fix: str, original_tex: str, temperature: float = 0.1) -> str:
    """
    Integrator: If proof_fix is JSON list of patches, apply them to original_tex by searching labels or block matches.
    If NO_CHANGES or parsing fails, only apply narrative fixes by inserting an 'Improved Narrative' marker.
    This function is conservative: it will not attempt dangerous global changes.
    """
    # 1) Get preamble/body
    preamble, body = split_preamble_and_body(original_tex)

    # Apply narrative fix: try to replace Introduction environment if present, otherwise prepend a comment block
    updated_body = body
    if narrative_fix and len(narrative_fix.strip()) > 0:
        # naive heuristic: look for \section{Introduction} and replace its first paragraph(s).
        intro_match = re.search(r"(\\section\*?\{[Ii]ntroduction.*?\})(.*?)(\\section\*?\{)", updated_body, re.S)
        if intro_match:
            # replace middle part with narrative_fix (keep the header and next section header)
            new_intro = intro_match.group(1) + "\n% --- Improved Narrative Start ---\n" + narrative_fix + "\n% --- Improved Narrative End ---\n"
            # reconstruct updated_body
            updated_body = updated_body[:intro_match.start()] + new_intro + updated_body[intro_match.end() - len(intro_match.group(3)):]
        else:
            # fallback: prepend improved narrative comment after \begin{document}
            updated_body = updated_body.replace("\\begin{document}", "\\begin{document}\n% --- Improved Narrative ---\n" + narrative_fix + "\n% --- End Improved Narrative ---\n", 1)

    # Apply proof fixes if proof_fix is JSON
    try:
        if proof_fix.strip() != "" and proof_fix.strip() != "NO_CHANGES":
            patches = json.loads(proof_fix)
            # We expect patches to be list of {"label":..., "replacement":...}
            if isinstance(patches, list):
                # Apply each patch by label or by index token
                for p in patches:
                    lbl = p.get("label")
                    repl = p.get("replacement", "")
                    if not lbl or not repl:
                        continue
                    # Try by label
                    label_pattern = re.compile(rf"(\\label\{{{re.escape(lbl)}\}})")
                    m = label_pattern.search(updated_body)
                    if m:
                        # find containing environment around the label (search backwards to \begin{env} and forward to \end{env})
                        start_search = updated_body.rfind("\\begin", 0, m.start())
                        end_search = updated_body.find("\\end", m.end())
                        if start_search != -1 and end_search != -1:
                            # find closing '}' after \end{...}
                            end_env_end = updated_body.find("}", end_search) + 1
                            if end_env_end > -1:
                                updated_body = updated_body[:start_search] + repl + updated_body[end_env_end:]
                                continue
                    # Fallback: attempt to find an environment by index token IDX_n
                    idx_m = re.match(r"IDX_(\d+)", lbl)
                    if idx_m:
                        idx = int(idx_m.group(1)) - 1
                        envs = extract_envs(updated_body)
                        if 0 <= idx < len(envs):
                            env = envs[idx]
                            updated_body = updated_body[:env["start"]] + repl + updated_body[env["end"]:]
                            continue
            # else: if not list, ignore (defensive)
    except Exception as e:
        # If parsing/applying patches fails, return original tex with a warning comment appended
        updated_body = "% WARNING: failed to apply proof patches: " + str(e) + "\n" + updated_body

    # Reconstruct full document
    new_tex = preamble + updated_body
    return new_tex

# ---------------------------------------------------------
# STREAMLIT UI & MAIN LOGIC
# ---------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    initial_tex = st.text_area("1. Paste your LaTeX Paper:", height=300, placeholder="\\documentclass{article}...")

with col2:
    human_referee_input = st.text_area(
        "2. (Optional) Your Referee Report:",
        height=300,
        placeholder="E.g., 'Theorem 3 is incorrect because...' or 'The introduction is too vague.'\n\nIf you fill this in, the first iteration will focus ONLY on addressing your comments."
    )

run = st.button("🚀 Start Review Cycle", type="primary")

if run and initial_tex.strip():
    current_tex = initial_tex

    # Display temps in header so user can confirm them
    st.markdown("**Deterministic temperatures in use:**")
    st.write({
        "novelty_temp": novelty_temp,
        "org_temp": org_temp,
        "proof_temp": proof_temp,
        "narrative_temp": narrative_temp,
        "prooffix_temp": prooffix_temp,
        "integrator_temp": integrator_temp
    })

    for i in range(iterations):
        st.markdown(f"--- \n ### 🔄 Iteration {i+1} / {iterations}")

        full_referee_report = ""
        is_human_round = (i == 0 and human_referee_input.strip() != "")

        # --- REFEREE PHASE ---
        with st.spinner(f"Referee Phase (Round {i+1})"):
            if is_human_round:
                st.info("👤 Using Human Referee Report for this round.")
                full_referee_report = human_referee_input
            else:
                # 1. Novelty
                st.write("🕵️ AI checking novelty (section-chunked if document large)...")
                novelty_report = agent_referee_novelty(current_tex, temperature=novelty_temp)

                # If novelty agent returns a FAIL token at start, short-circuit
                if isinstance(novelty_report, str) and novelty_report.strip().upper().startswith("FAIL"):
                    st.error("⛔ Paper rejected by Gatekeeper Agent.")
                    st.error(novelty_report)
                    st.stop()

                # 2. Organization
                st.write("📐 AI checking organization (section-chunked if large)...")
                org_report = agent_referee_org(current_tex, temperature=org_temp)

                # 3. Proofs (theorem-by-theorem)
                st.write("🧮 AI checking proofs (theorems processed individually)...")
                proof_report = agent_referee_proofs(current_tex, temperature=proof_temp)

                full_referee_report = f"""
NOVELTY REPORT:
{novelty_report}

ORGANIZATION REPORT:
{org_report}

PROOF REPORT:
{proof_report}
"""
                with st.expander("View AI Generated Referee Report"):
                    st.code(full_referee_report)

        # --- AUTHOR PHASE ---
        with st.spinner(f"Author Phase (Round {i+1})"):
            st.write("✍️ Narrative Expert rewriting...")
            narrative_fix = agent_author_narrative(full_referee_report, current_tex, temperature=narrative_temp)

            st.write("🧠 Proof Expert fixing logic (per-referee findings)...")
            proof_fix = agent_author_proof_fixer(full_referee_report, current_tex, temperature=prooffix_temp)

            st.write("🔗 Lead Author integrating changes...")
            updated_tex = agent_author_integrator(narrative_fix, proof_fix, current_tex, temperature=integrator_temp)

            # Update current_tex only if integrator returned something non-empty
            if updated_tex and updated_tex.strip():
                current_tex = updated_tex

            st.success(f"Iteration {i+1} finished.")

    # --- FINAL OUTPUT ---
    st.markdown("---")
    st.subheader("🎉 Final Improved Paper")

    tab1, tab2 = st.tabs(["Rendered Code (preview)", "Raw LaTeX"])
    with tab1:
        st.code(current_tex[:20000] + ("\n\n% --- trimmed preview ---" if len(current_tex) > 20000 else ""), language="latex")
        if len(current_tex) > 20000:
            st.info("Document preview trimmed to 20k chars. Download the .tex to see full result.")
    with tab2:
        st.text_area("Full LaTeX", value=current_tex, height=400)

    st.download_button(
        label="📥 Download .tex file",
        data=current_tex,
        file_name="improved_paper.tex",
        mime="text/plain"
    )
else:
    if run:
        st.warning("Please paste LaTeX content into the left box before starting.")
