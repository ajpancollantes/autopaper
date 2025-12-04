import streamlit as st
import google.generativeai as genai

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(page_title="AI Peer Review & Repair", layout="wide")
api_key = st.sidebar.text_input("Enter Gemini API Key:", type="password")
iterations = st.sidebar.number_input("Referee-Author Iterations:", min_value=1, max_value=5, value=1)

if not api_key:
    st.warning("Enter your API Key to begin.")
    st.stop()

genai.configure(api_key=api_key)

def ask_model(system_instruction, prompt, temperature=0.3):
    """
    Unified wrapper for Gemini calls.
    Using gemini-2.0-flash (or 1.5-flash) is recommended for speed/cost 
    in iterative loops, but 1.5-pro is better for complex math logic.
    """
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash", # Changed to standard stable model name
        system_instruction=system_instruction
    )
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=temperature
        )
    )
    return response.text

# ---------------------------------------------------------
#  REFEREE AGENTS ( The Critics )
# ---------------------------------------------------------

def agent_referee_novelty(tex_content: str) -> str:
    """Evaluates Novelty, Interestingness, and General Structure."""
    sys_prompt = """
    You are a Senior Editor at a top mathematics journal. 
    1. Evaluate the NOVELTY and INTERESTINGNESS of the paper.
    2. Check the GENERAL STRUCTURE (Does it have Intro, Prelims, Main Results?).
    
    IMPORTANT: If the paper is not novel or structural nonsense, start your response with "FAIL: [Reason]". 
    Otherwise, start with "PASS" and provide a brief report.
    """
    return ask_model(sys_prompt, f"Evaluate this paper:\n{tex_content}", 0.2)

def agent_referee_org(tex_content: str) -> str:
    """Evaluates the organization and flow."""
    sys_prompt = """
    You are a Referee focused on exposition. 
    Evaluate the organization: Section ordering, logical flow between paragraphs, and clarity.
    Output a bulleted list of organizational weaknesses.
    """
    return ask_model(sys_prompt, f"Critique the organization:\n{tex_content}", 0.2)

def agent_referee_proofs(tex_content: str) -> str:
    """
    Acts as the 'Extractor' and 'Verifier' sub-agents.
    Extracts statements/proofs and checks correctness.
    """
    sys_prompt = """
    You are a meticulous Math Referee. 
    1. Identify every Theorem/Lemma and its Proof.
    2. Check the correctness of each proof step-by-step.
    3. If a proof is correct, say "Theorem X: Correct".
    4. If a proof is incorrect, explain exactly WHERE the logic breaks.
    """
    return ask_model(sys_prompt, f"Verify the proofs in this text:\n{tex_content}", 0.1)

# ---------------------------------------------------------
#  AUTHOR AGENTS ( The Repair Team )
# ---------------------------------------------------------

def agent_author_narrative(report: str, tex_content: str) -> str:
    """Expert in narrative, intros, and conclusions."""
    sys_prompt = """
    You are an Expert Mathematical Writer. 
    Based on the Referee's report, rewrite the Introduction, Conclusion, and transitional text.
    Do NOT change the Theorems or Proofs (math content), only the narrative structure.
    """
    return ask_model(sys_prompt, f"Referee Report:\n{report}\n\nOriginal Text:\n{tex_content}", 0.4)

def agent_author_proof_fixer(proof_report: str, tex_content: str) -> str:
    """Expert in writing proofs. Only fixes flawed proofs."""
    sys_prompt = """
    You are a Mathematician specializing in fixing proofs.
    Read the Referee's proof report. If a proof is marked as flawed, rewrite that specific Theorem and Proof entirely.
    If the proof was correct, leave it exactly as is.
    Output the corrected LaTeX segments for the flawed parts.
    """
    return ask_model(sys_prompt, f"Proof Report:\n{proof_report}\n\nOriginal Text:\n{tex_content}", 0.1)

def agent_author_integrator(narrative_fix: str, proof_fix: str, original_tex: str) -> str:
    """Integrates responses into a coherent whole."""
    sys_prompt = """
    You are the Lead Author. 
    Integrate the improved narrative text and the corrected proofs into a single, coherent LaTeX document.
    Ensure all packages and document structure are preserved.
    """
    query = f"""
    1. Improved Narrative: {narrative_fix}
    2. Corrected Proofs: {proof_fix}
    3. Original Source: {original_tex}
    
    Output the full, compiled, valid LaTeX file.
    """
    return ask_model(sys_prompt, query, 0.1)

# ---------------------------------------------------------
# STREAMLIT UI & LOGIC
# ---------------------------------------------------------

st.title("🤖🎓 Agentic Peer Review System")
st.markdown("### Upload your LaTeX -> Referee checks -> Author fixes")

initial_tex = st.text_area("Paste your LaTeX Paper:", height=300, placeholder="\\documentclass{article}...")
run = st.button("🚀 Start Review Cycle")

if run and initial_tex.strip():
    current_tex = initial_tex
    
    # Progress container
    log = st.empty()

    for i in range(iterations):
        st.markdown(f"--- \n ### 🔄 Iteration {i+1} / {iterations}")
        
        # --- REFEREE PHASE ---
        with st.status(f"Referee Phase (Round {i+1})", expanded=True):
            
            # Step 1: Novelty & Structure
            st.write("🕵️ checking novelty...")
            novelty_report = agent_referee_novelty(current_tex)
            
            if novelty_report.strip().upper().startswith("FAIL"):
                st.error("⛔ Paper rejected by Gatekeeper Agent.")
                st.error(novelty_report)
                st.stop() # Return to human
            
            st.success("Novelty/Structure passed.")
            with st.expander("Novelty Report"):
                st.write(novelty_report)

            # Step 2: Organization
            st.write("📐 checking organization...")
            org_report = agent_referee_org(current_tex)
            
            # Step 3: Proofs
            st.write("🧮 checking proofs...")
            proof_report = agent_referee_proofs(current_tex)
            with st.expander("Proof Analysis"):
                st.write(proof_report)

            full_referee_report = f"""
            NOVELTY REPORT: {novelty_report}
            ORGANIZATION REPORT: {org_report}
            PROOF REPORT: {proof_report}
            """

        # --- AUTHOR PHASE ---
        with st.status(f"Author Phase (Round {i+1})", expanded=True):
            st.write("✍️ Narrative Expert rewriting...")
            narrative_fix = agent_author_narrative(full_referee_report, current_tex)
            
            st.write("🧠 Proof Expert fixing logic...")
            proof_fix = agent_author_proof_fixer(proof_report, current_tex)
            
            st.write("🔗 Lead Author integrating...")
            current_tex = agent_author_integrator(narrative_fix, proof_fix, current_tex)
            
            st.success(f"Iteration {i+1} complete.")

    # --- FINAL OUTPUT ---
    st.markdown("---")
    st.subheader("🎉 Final Improved Paper")
    st.code(current_tex, language="latex")
    
    st.download_button(
        label="📥 Download .tex file",
        data=current_tex,
        file_name="improved_paper.tex",
        mime="text/plain"
    )
