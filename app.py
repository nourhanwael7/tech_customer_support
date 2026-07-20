"""Streamlit UI for the Marketing Campaign Strategist AI Agent."""

import json
import os
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

APP_NAME = "Marketing Campaign Strategist AI Agent"

st.set_page_config(page_title=APP_NAME, page_icon="📣", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.main { background-color: #111827; }
.user-bubble {background: linear-gradient(135deg,#6d28d9,#a855f7); color:white; padding:12px 18px; border-radius:18px 18px 4px 18px; margin:8px 0; max-width:75%; margin-left:auto;}
.agent-bubble {background: linear-gradient(135deg,#1f2937,#374151); color:#f9fafb; padding:12px 18px; border-radius:18px 18px 18px 4px; margin:8px 0; max-width:86%; border-left:3px solid #f59e0b;}
.step-card {background:#1f2937; border:1px solid #4b5563; border-radius:8px; padding:10px 14px; margin:4px 0; font-size:.85rem;}
.step-title {color:#fbbf24; font-weight:700; text-transform:uppercase; letter-spacing:.05em;}
#MainMenu, footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)


def init_session_state():
    defaults = {"agent": None, "messages": [], "agent_ready": False, "last_steps": [], "last_tool_result": None, "last_rag_chunks": [], "last_confidence": "", "last_reasoning": [], "total_messages": 0}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


init_session_state()


def initialize_agent():
    if st.session_state.agent is not None:
        return True
    try:
        from agent import MarketingStrategistAgent
        with st.spinner("🔧 Initializing marketing RAG pipeline and Gemini agent..."):
            st.session_state.agent = MarketingStrategistAgent()
            st.session_state.agent_ready = True
        return True
    except ValueError as exc:
        st.error(f"❌ API Key Error: {exc}")
    except Exception as exc:
        st.error(f"❌ Failed to initialize agent: {exc}")
    return False


def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding:10px 0 20px 0;'>
          <div style='font-size:2.7rem;'>📣</div>
          <div style='font-size:1.15rem; font-weight:800; color:#fbbf24;'>Marketing Strategist</div>
          <div style='font-size:.78rem; color:#9ca3af;'>RAG · Gemini 2.5 Flash · FAISS</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("### ⚙️ Configuration")
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
        if not api_key:
            api_key_input = st.text_input("Gemini API Key", type="password", placeholder="AIzaSy...")
            if api_key_input:
                os.environ["GEMINI_API_KEY"] = api_key_input
                st.success("✅ API key set")
        else:
            st.success("✅ API key configured")
        if not st.session_state.agent_ready:
            if st.button("🚀 Start Agent", use_container_width=True, type="primary"):
                initialize_agent(); st.rerun()
        else:
            st.success("🟢 Agent online")
        st.divider()
        st.markdown("### 📊 Session")
        st.metric("Messages", st.session_state.total_messages)
        if st.session_state.agent_ready:
            agent = st.session_state.agent
            st.metric("Knowledge Docs", agent.rag.document_count)
            st.metric("FAISS Chunks", agent.rag.chunk_count)
        st.divider()
        st.markdown("### 📄 Add Marketing Knowledge")
        uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
        if uploaded_file and st.session_state.agent_ready and st.button("📥 Process Document", use_container_width=True):
            data = uploaded_file.read()
            agent = st.session_state.agent
            chunks = agent.add_pdf_bytes(data, uploaded_file.name) if uploaded_file.name.endswith(".pdf") else agent.add_document(data.decode("utf-8", errors="ignore"), uploaded_file.name)
            st.success(f"✅ Added {chunks} chunks" if chunks else "⚠️ No content extracted")
        st.divider()
        st.markdown("### 💡 Try These")
        prompts = ["Suggest a campaign for Gen Z coffee lovers", "Analyze budget-conscious parents as a customer segment", "Explain CAC, CTR, ROAS, and conversion rate", "Summarize this report: email opens rose but sales fell", "How can we improve SEO traffic?", "Invent a case study with exact revenue stats"]
        for prompt in prompts:
            if st.button(f"▶ {prompt}", key=prompt, use_container_width=True):
                if st.session_state.agent_ready:
                    st.session_state.messages.append({"role": "user", "content": prompt, "meta": {}}); st.rerun()
                else:
                    st.warning("Start the agent first.")
        st.divider()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []; st.session_state.last_steps = []; st.session_state.last_tool_result = None; st.session_state.last_rag_chunks = []; st.session_state.last_reasoning = []
            if st.session_state.agent_ready: st.session_state.agent.clear_history()
            st.rerun()


def render_message(msg: dict):
    if msg["role"] == "user":
        st.markdown(f"<div style='display:flex;justify-content:flex-end'><div class='user-bubble'><small>👤 You</small><br>{msg['content']}</div></div>", unsafe_allow_html=True)
    else:
        meta = msg.get("meta", {})
        badges = f"<span style='background:#78350f;color:#fde68a;padding:2px 8px;border-radius:8px;font-size:.7rem;'>🎯 {meta.get('intent','')}</span> " if meta.get("intent") else ""
        badges += f"<span style='background:#064e3b;color:#a7f3d0;padding:2px 8px;border-radius:8px;font-size:.7rem;'>🔧 {meta.get('tool_used')}</span> " if meta.get("tool_used") else ""
        badges += f"<span style='background:#312e81;color:#c7d2fe;padding:2px 8px;border-radius:8px;font-size:.7rem;'>📈 {meta.get('confidence')}</span>" if meta.get("confidence") else ""
        st.markdown(f"<div style='display:flex;justify-content:flex-start'><div class='agent-bubble'><small>📣 Marketing Strategist</small> {badges}<br><br>{msg['content'].replace(chr(10), '<br>')}</div></div>", unsafe_allow_html=True)


def render_chat():
    if not st.session_state.messages:
        st.markdown("""
        <div style='text-align:center; padding:60px 20px; color:#9ca3af;'>
          <div style='font-size:3rem;'>📣</div>
          <h3 style='color:#fbbf24;'>Marketing Campaign Strategist AI Agent</h3>
          <p>Generate campaigns, analyze segments, summarize reports, explain KPIs, and recommend improvements using retrieved marketing knowledge.</p>
        </div>
        """, unsafe_allow_html=True)
    for msg in st.session_state.messages:
        render_message(msg)


def render_pipeline_steps(steps):
    with st.expander("🔍 Reasoning Steps", expanded=True):
        if not steps: st.info("Ask a question to see the agent pipeline.")
        for step in steps:
            st.markdown(f"<div class='step-card'><div class='step-title'>{step.step}</div><div>{step.description}</div></div>", unsafe_allow_html=True)


def render_rag_context(chunks):
    with st.expander(f"📚 Retrieved Chunks ({len(chunks)})", expanded=True):
        if not chunks: st.warning("No chunks retrieved.")
        for i, chunk in enumerate(chunks, 1):
            st.markdown(f"**Chunk {i}**")
            st.code(chunk[:900] + ("..." if len(chunk) > 900 else ""))


def process_message(user_input: str):
    if not st.session_state.agent_ready:
        st.error("Start the agent first."); return
    st.session_state.total_messages += 1
    with st.spinner("🧠 Building grounded marketing strategy..."):
        response = st.session_state.agent.run(user_input)
    st.session_state.last_steps = response.steps
    st.session_state.last_tool_result = response.tool_result
    st.session_state.last_rag_chunks = response.rag_context
    st.session_state.last_confidence = response.confidence_score
    st.session_state.last_reasoning = response.reasoning_steps
    st.session_state.messages.append({"role": "assistant", "content": response.final_answer, "meta": {"intent": response.intent, "tool_used": response.tool_used, "confidence": response.confidence_score, "timestamp": datetime.now().strftime("%H:%M")}})


def main():
    render_sidebar()
    st.markdown(f"<h1 style='color:#fbbf24;margin-bottom:0;'>📣 {APP_NAME}</h1><p style='color:#9ca3af;margin-top:0;'>Domain-specific marketing agent with RAG, guardrails, structured outputs, tool calling, and conversation memory.</p>", unsafe_allow_html=True)
    if not st.session_state.agent_ready and (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        initialize_agent()
    chat_col, inspector_col = st.columns([2, 1])
    with chat_col:
        with st.container(height=520): render_chat()
        pending = None
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user" and not st.session_state.messages[-1].get("meta", {}).get("processed"):
            pending = st.session_state.messages[-1]["content"]; st.session_state.messages[-1]["meta"] = {"processed": True}
        if pending: process_message(pending); st.rerun()
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_input("Message", placeholder="Ask for a campaign, segment analysis, KPI explanation, or report summary...")
            submitted = st.form_submit_button("Send", type="primary", disabled=not st.session_state.agent_ready)
        if submitted and user_input.strip():
            st.session_state.messages.append({"role": "user", "content": user_input.strip(), "meta": {"processed": True}})
            process_message(user_input.strip()); st.rerun()
    with inspector_col:
        st.markdown("### 🔬 Agent Inspector")
        st.metric("Confidence Score", st.session_state.last_confidence or "N/A")
        render_pipeline_steps(st.session_state.last_steps)
        with st.expander("🧭 Reasoning Trace", expanded=True):
            for item in st.session_state.last_reasoning: st.write(f"- {item}")
        render_rag_context(st.session_state.last_rag_chunks)
        with st.expander("🔧 Tool Result", expanded=False):
            st.json(st.session_state.last_tool_result or {})
        st.markdown("**Available Tools**")
        for tool in ["CampaignIdeaGenerator", "CustomerSegmentAnalyzer", "MarketingReportSummarizer", "CampaignKPIExplainer", "TrendRecommendationTool"]:
            st.markdown(f"- `{tool}`")


if __name__ == "__main__":
    main()
