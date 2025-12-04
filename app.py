import streamlit as st
import google.generativeai as genai

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
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=system_instruction
    )
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=temperature)
        )
        return response.text
    except Exception as e:
        return f"Error connecting to API: {e}"

# ---------------------------------------------------------
#  REFEREE AGENTS ( The Critics )
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
#  AUTHOR AGENTS ( The Repair Team )
# ---------------------------------------------------------

def agent_author_narrative(full_report: str, tex_content: str) -> str:
    """Extracts narrative instructions from ANY report (Human or AI) and applies them."""
    sys_prompt = """
    You are an Expert Mathematical Writer. 
    Read the Referee Report below. Ignore math correctness comments.
    Focus on: Introduction, Conclusion, Structure, and Textual Flow.
    Rewrite the narrative parts of the paper based on the feedback.
    """
    return ask_model(sys_prompt, f"Referee Report:\n{full_report}\n\nOriginal Text:\n{tex_content}", 0.4)

def agent_author_proof_fixer(full_report: str, tex_content: str) -> str:
    """Extracts math errors from ANY report (Human or AI) and fixes them."""
    sys_prompt = """
    You are a Mathematician specializing in fixing proofs.
    Read the Referee Report below. Look for claims about incorrect proofs or theorems.
    If a proof is marked as flawed, rewrite that specific Theorem and Proof.
    If no specific math errors are found, output "NO_CHANGES".
    """
    return ask_model(sys_prompt, f"Referee Report:\n{full_report}\n\nOriginal Text:\n{tex_content}", 0.1)

def agent_author_integrator(narrative_fix: str, proof_fix: str, original_tex: str) -> str:
    sys_prompt = """
    You are the Lead Author. 
    Integrate the improved narrative text and the corrected proofs into the original document.
    1. Replace old narrative with 'Improved Narrative'.
    2. If 'Corrected Proofs' is not NO_CHANGES, replace the specific theorems/proofs.
    Output the full, compiled, valid LaTeX file.
    """
    query = f"1. Improved Narrative: {narrative_fix}\n2. Corrected Proofs: {proof_fix}\n3. Original Source: {original_tex}"
    return ask_model(sys_prompt, query, 0.1)

# ---------------------------------------------------------
# STREAMLIT UI & LOGIC
# ---------------------------------------------------------

st.title("🤖🎓 Agentic Peer Review System")

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
    
    for i in range(iterations):
        st.markdown(f"--- \n ### 🔄 Iteration {i+1} / {iterations}")
        
        full_referee_report = ""
        is_human_round = (i == 0 and human_referee_input.strip() != "")

        # --- REFEREE PHASE ---
        with st.status(f"Referee Phase (Round {i+1})", expanded=True) as status:
            
            if is_human_round:
                st.info("👤 Using Human Referee Report for this round.")
                full_referee_report = human_referee_input
                # We skip the AI novelty check because the human explicitly asked for fixes.
            
            else:
                # 1. Novelty
                st.write("🕵️ AI checking novelty...")
                novelty_report = agent_referee_novelty(current_tex)
                
                if novelty_report.strip().upper().startswith("FAIL"):
                    st.error("⛔ Paper rejected by Gatekeeper Agent.")
                    st.error(novelty_report)
                    status.update(label="Referee Phase Failed", state="error")
                    st.stop()
                
                # 2. Organization & Proofs
                st.write("📐 AI checking organization...")
                org_report = agent_referee_org(current_tex)
                
                st.write("🧮 AI checking proofs...")
                proof_report = agent_referee_proofs(current_tex)

                full_referee_report = f"""
                NOVELTY REPORT: {novelty_report}
                ORGANIZATION REPORT: {org_report}
                PROOF REPORT: {proof_report}
                """
                
                with st.expander("View AI Generated Report"):
                    st.write(full_referee_report)
            
            status.update(label="Referee Phase Complete", state="complete")

        # --- AUTHOR PHASE ---
        with st.status(f"Author Phase (Round {i+1})", expanded=True) as status:
            st.write("✍️ Narrative Expert rewriting...")
            # We pass the full report; the agent is instructed to extract relevant parts
            narrative_fix = agent_author_narrative(full_referee_report, current_tex)
            
            st.write("🧠 Proof Expert fixing logic...")
            proof_fix = agent_author_proof_fixer(full_referee_report, current_tex)
            
            st.write("🔗 Lead Author integrating...")
            current_tex = agent_author_integrator(narrative_fix, proof_fix, current_tex)
            
            status.update(label="Author Phase Complete", state="complete")
            st.success(f"Iteration {i+1} finished.")

    # --- FINAL OUTPUT ---
    st.markdown("---")
    st.subheader("🎉 Final Improved Paper")
    
    tab1, tab2 = st.tabs(["Rendered Code", "Raw LaTeX"])
    with tab1:
        st.code(current_tex, language="latex")
    with tab2:
        st.text(current_tex)
    
    st.download_button(
        label="📥 Download .tex file",
        data=current_tex,
        file_name="improved_paper.tex",
        mime="text/plain"
    )
