import streamlit as st
import google.generativeai as genai
from typing import List, Dict
import json

# --- CONFIGURATION ---
st.set_page_config(page_title="Math Research Copilot", layout="wide")

# Sidebar for API Key
api_key = st.sidebar.text_input("Enter Google Gemini API Key", type="password")

if not api_key:
    st.warning("Please enter your API Key in the sidebar to start.")
    st.stop()

# Configure Gemini
genai.configure(api_key=api_key)

# --- DEFINING THE AGENTS ---

def get_gemini_response(prompt, temperature=0.7):
    model = genai.GenerativeModel('gemini-1.5-flash') # 'Flash' is fast and free
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(temperature=temperature)
    )
    return response.text

def agent_generator(context: str, current_ideas: List[str]) -> str:
    prompt = f"""
    You are a creative mathematical researcher.
    Original Context: "{context}"
    Current Accepted Ideas: {current_ideas}
    
    TASK: Propose ONE new, novel research follow-up or conjecture based on the context.
    Focus on creative connections. Keep it concise (3-4 sentences).
    """
    return get_gemini_response(prompt, temperature=0.9)

def agent_critic(idea: str, context: str) -> Dict:
    prompt = f"""
    You are a strict mathematics reviewer.
    Context: "{context}"
    Proposed Idea: "{idea}"
    
    Evaluate this idea on:
    1. Novelty (Is it new?)
    2. Correctness (Does it make mathematical sense?)
    3. Feasibility (Is it solvable?)
    
    Output ONLY valid JSON in this format:
    {{
        "score": <integer 1-10>,
        "critique": "<short text critique>",
        "improved_version": "<rewrite the idea to be more rigorous>"
    }}
    """
    try:
        response = get_gemini_response(prompt, temperature=0.1)
        # Clean up JSON formatting if the model adds markdown
        cleaned_response = response.replace("```json", "").replace("```", "")
        return json.loads(cleaned_response)
    except:
        return {"score": 0, "critique": "Error parsing critic response", "improved_version": idea}

# --- THE APP UI ---

st.title("🎓 Autonomous Research Copilot")
st.markdown("Paste your math abstract/notes below. The AI will brainstorm and refine ideas recursively.")

original_text = st.text_area("Your Notes/Abstract:", height=150)
iterations = st.slider("Number of Brainstorming Loops", 1, 5, 3)

if st.button("Start Research Loop") and original_text:
    
    research_log = []
    final_ideas = []
    
    status_box = st.empty()
    
    # --- THE LOOP ---
    for i in range(iterations):
        status_box.info(f"🔄 Iteration {i+1}/{iterations}: Generating ideas...")
        
        # 1. Generator generates an idea
        raw_idea = agent_generator(original_text, final_ideas)
        
        # 2. Critic evaluates it
        status_box.info(f"⚖️ Iteration {i+1}/{iterations}: Critic is reviewing...")
        review = agent_critic(raw_idea, original_text)
        
        # 3. Decision Logic
        if review['score'] >= 7:
            status_box.success(f"✅ Idea Accepted! (Score: {review['score']}/10)")
            final_ideas.append(review['improved_version'])
            # Update context for next loop so it doesn't repeat
            original_text += f"\n\n[Expansion {i+1}]: {review['improved_version']}"
        else:
            status_box.warning(f"❌ Idea Rejected (Score: {review['score']}/10): {review['critique']}")
        
        # Log for display
        research_log.append({
            "iteration": i+1,
            "raw_idea": raw_idea,
            "score": review['score'],
            "critique": review['critique'],
            "accepted": review['score'] >= 7
        })

    status_box.empty()
    
    # --- RESULTS DISPLAY ---
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Final Expanded Paper Plan")
        st.write(original_text)
        
    with col2:
        st.subheader("🕵️ Process Log")
        for log in research_log:
            with st.expander(f"Loop {log['iteration']} - Score: {log['score']}"):
                st.markdown(f"**Draft:** {log['raw_idea']}")
                st.markdown(f"**Critique:** {log['critique']}")
                if log['accepted']:
                    st.success("Added to paper")
                else:
                    st.error("Discarded")
