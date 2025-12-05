import streamlit as st
import google.generativeai as genai
import re

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(page_title="AI Peer Review & Repair", layout="wide")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Enter Gemini API Key:", type="password")
    iterations = st.number_input("Referee-Author Iterations:", min_value=1, max_value=5, value=1)
    st.info("Note: If you provide a manual report, it consumes the first iteration.")

    if not api_key:
        st.warning("Enter your API Key to begin.")
        st.stop()

genai.configure(api_key=api_key)


def ask_model(system_instruction, prompt, temperature=0.3):
    """Call the Gemini model (thin wrapper)."""
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=system_instruction
    )
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=temperature)
        )
        # Depending on SDK version the attribute might differ; keep as .text for now.
        return getattr(response, "text", str(response))
    except Exception as e:
        return f"Error connecting to API: {e}"


# ---------------------------------------------------------
# REFEREE AGENTS ( The Critics )
# ---------------------------------------------------------

def agent_referee_novelty(tex_content: str) -> str:
    sys_prompt = """
You are a Senior Editor at a top mathematics journal.
1. Evaluate the NOVELTY and INTERESTINGNESS.
2. Check the GENERAL STRUCTURE.
IMPORTANT: If the paper is not novel or nonsense, start with "FAIL: [Reason]".
Otherwise, start with "PASS" and provide a brief report.
"""
    return ask_model(sys_prompt, f"Evaluate this paper:\n{tex_content}", 0.2)


def agent_referee_org(tex_content: str) -> str:
    sys_prompt = "You are a Referee. List organizational weaknesses and flow issues."
    return ask_model(sys_prompt, f"Critique organization:\n{tex_content}", 0.2)


def agent_referee_proofs(tex_content: str) -> str:
    sys_prompt = """
You are a Math Referee.
Identify every Theorem/Proof. Check correctness step-by-step.
If flawed, explain exactly WHERE the logic breaks.
"""
    return ask_model(sys_prompt, f"Verify proofs:\n{tex_content}", 0.1)


# ---------------------------------------------------------
# AUTHOR AGENTS ( The Repair Team )
# ---------------------------------------------------------

def agent_author_narrative(full_report: str, tex_content: str) -> str:
    sys_prompt = """
You are an Expert Mathematical Writer.
Read the Referee Report below. Ignore math correctness comments.
Focus on: Introduction, Conclusion, Structure, and Textual Flow.
Rewrite the narrative parts of the paper based on the feedback.
"""
    return ask_model(sys_prompt, f"Referee Report:\n{full_report}\n\nOriginal Text:\n{tex_content}", 0.4)


def agent_author_proof_fixer(full_report: str, tex_content: str) -> str:
    sys_prompt = """
You are a Mathematician specializing in fixing proofs.
Read the Referee Report below. Look for claims about incorrect proofs or theorems.
If a proof is marked as flawed, rewrite that specific Theorem and Proof.
If no specific math errors are found, output "NO_CHANGES".
"""
    return ask_model(sys_prompt, f"Referee Report:\n{full_report}\n\nOriginal Text:\n{tex_content}", 0.1)


# Helper: safely replace LaTeX environment blocks
def replace_environment_block(tex: str, env: str, new_block: str) -> str:
    """
    Replace the first occurrence of a LaTeX environment "env" in tex
    with new_block. If no such environment is found, returns tex unchanged.
    """
    env_escaped = re.escape(env)
    pattern = r"(\\begin\{" + env_escaped + r"\}.*?\\end\{" + env_escaped + r"\})"
    return re.sub(pattern, new_block, tex, count=1, flags=re.S)


def replace_all_corrected_proofs(tex: str, proof_fix: str) -> str:
    """
    proof_fix is expected to contain one or more LaTeX environment blocks,
    e.g. \begin{theorem}...\end{theorem} or \begin{proof}...\end{proof}.
    For each such block in proof_fix, replace the first occurrence of the
    same environment in tex with that corrected block.
    """
    if proof_fix.strip().upper() == "NO_CHANGES":
        return tex

    # Find all full \begin{...}\end{...} blocks in proof_fix
    corrected_blocks = re.findall(r"(\\begin\{[^\}]+\}.*?\\end\{[^\}]+\})", proof_fix, flags=re.S)
    if not corrected_blocks:
        return tex

    updated_tex = tex
    for block in corrected_blocks:
        m = re.search(r"\\begin\{([^\}]+)\}", block)
        if not m:
            continue
        env_name = m.group(1)
        # Replace the first occurrence of this env in the original tex
        updated_tex = replace_environment_block(updated_tex, env_name, block)

    return updated_tex


def agent_author_integrator(narrative_fix: str, proof_fix: str, original_tex: str) -> str:
    """
    Integrate the improved narrative and corrected proofs into original_tex.

    Heuristics used:
      - Replace the 'Introduction' section if found (first \section{Introduction} ... until next \section or \end{document}).
      - Otherwise, prepend the improved narrative before the document body.
      - Replace corrected proof/theorem environments using replace_all_corrected_proofs().
    """
    merged = original_tex

    # Try to replace the Introduction section (heuristic)
    intro_pattern = re.compile(r"(\\section\{Introduction\}.*?)(?=\\section\{|\\end\{document\})", flags=re.S)
    if intro_pattern.search(merged):
        merged = intro_pattern.sub(narrative_fix, merged, count=1)
    else:
        # If no clear Introduction, try t
