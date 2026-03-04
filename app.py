"""
app.py — TechStore Multi-Step AI Customer Support Agent
Streamlit UI with chat interface, PDF upload, admin panel, and conversation history.
"""

import os
import sys
import time
import json
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page Config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="TechStore AI Support",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Main background */
.main { background-color: #0f1117; }

/* Chat message bubbles */
.user-bubble {
    background: linear-gradient(135deg, #1e40af, #3b82f6);
    color: white;
    padding: 12px 18px;
    border-radius: 18px 18px 4px 18px;
    margin: 8px 0;
    max-width: 75%;
    margin-left: auto;
    box-shadow: 0 2px 8px rgba(59,130,246,0.3);
}
.agent-bubble {
    background: linear-gradient(135deg, #1e293b, #334155);
    color: #e2e8f0;
    padding: 12px 18px;
    border-radius: 18px 18px 18px 4px;
    margin: 8px 0;
    max-width: 80%;
    border-left: 3px solid #3b82f6;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}

/* Step pipeline cards */
.step-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 4px 0;
    font-size: 0.85rem;
}
.step-card .step-title {
    color: #60a5fa;
    font-weight: 600;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Metric cards */
.metric-card {
    background: #1e293b;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
    border: 1px solid #334155;
}

/* Sidebar styling */
.sidebar-section {
    background: #1e293b;
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 12px;
}

/* Status badges */
.badge-shipped { background: #1d4ed8; color: white; padding: 2px 10px; border-radius: 12px; font-size: 0.75rem; }
.badge-delivered { background: #15803d; color: white; padding: 2px 10px; border-radius: 12px; font-size: 0.75rem; }
.badge-processing { background: #92400e; color: white; padding: 2px 10px; border-radius: 12px; font-size: 0.75rem; }
.badge-cancelled { background: #991b1b; color: white; padding: 2px 10px; border-radius: 12px; font-size: 0.75rem; }

/* Input area */
.stTextInput > div > input {
    background: #1e293b;
    border: 1px solid #475569;
    color: white;
    border-radius: 8px;
}

/* Hide Streamlit default elements */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Session State Initialization ───────────────────────────────────────────────

def init_session_state():
    defaults = {
        "agent": None,
        "messages": [],           # chat history [{role, content, meta}]
        "agent_ready": False,
        "api_key_set": False,
        "last_steps": [],
        "last_tool_result": None,
        "last_rag_chunks": [],
        "doc_count": 0,
        "total_messages": 0,
        "tickets_created": 0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


# ── Agent Initialization ───────────────────────────────────────────────────────

def initialize_agent():
    """Initialize the agent (cached in session state)."""
    if st.session_state.agent is not None:
        return True
    try:
        from agent import CustomerSupportAgent
        with st.spinner("🔧 Initializing AI Agent & loading knowledge base..."):
            agent = CustomerSupportAgent()
            st.session_state.agent = agent
            st.session_state.agent_ready = True
            st.session_state.doc_count = agent.rag.document_count
        return True
    except ValueError as e:
        st.error(f"❌ API Key Error: {e}")
        return False
    except Exception as e:
        st.error(f"❌ Failed to initialize agent: {e}")
        return False


# ── Sidebar ────────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        # Logo / Header
        st.markdown("""
        <div style='text-align:center; padding: 10px 0 20px 0;'>
            <div style='font-size:2.5rem;'>🤖</div>
            <div style='font-size:1.2rem; font-weight:700; color:#60a5fa;'>TechStore AI</div>
            <div style='font-size:0.75rem; color:#64748b;'>Customer Support Agent</div>
        </div>
        """, unsafe_allow_html=True)

        # API Key Section
        st.markdown("### ⚙️ Configuration")
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
        if not api_key:
            api_key_input = st.text_input(
                "Gemini API Key",
                type="password",
                placeholder="AIzaSy...",
                help="Enter your Gemini API key. Get one at aistudio.google.com"
            )
            if api_key_input:
                os.environ["GEMINI_API_KEY"] = api_key_input
                st.success("✅ API Key set!")
                st.session_state.api_key_set = True
        else:
            st.success(f"✅ API Key configured")

        # Initialize Agent Button
        if not st.session_state.agent_ready:
            if st.button("🚀 Start Agent", use_container_width=True, type="primary"):
                initialize_agent()
                st.rerun()
        else:
            st.success("🟢 Agent Online")

        st.divider()

        # Stats
        st.markdown("### 📊 Session Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Messages", st.session_state.total_messages)
        with col2:
            st.metric("Tickets", st.session_state.tickets_created)

        if st.session_state.agent_ready:
            agent = st.session_state.agent
            st.metric("Docs in KB", agent.rag.document_count)
            st.metric("Chunks", agent.rag.chunk_count)

        st.divider()

        # PDF Upload Section
        st.markdown("### 📄 Upload Knowledge Documents")
        uploaded_file = st.file_uploader(
            "Upload PDF or TXT",
            type=["pdf", "txt"],
            help="Add documents to the AI's knowledge base"
        )

        if uploaded_file and st.session_state.agent_ready:
            if st.button("📥 Process Document", use_container_width=True):
                agent = st.session_state.agent
                file_bytes = uploaded_file.read()

                with st.spinner(f"Processing {uploaded_file.name}..."):
                    if uploaded_file.name.endswith(".pdf"):
                        chunks = agent.add_pdf_bytes(file_bytes, source=uploaded_file.name)
                    else:
                        text = file_bytes.decode("utf-8", errors="ignore")
                        chunks = agent.add_document(text, source=uploaded_file.name)

                if chunks > 0:
                    st.success(f"✅ Added {chunks} chunks from {uploaded_file.name}")
                    st.session_state.doc_count = agent.rag.document_count
                else:
                    st.warning("⚠️ No content extracted from document.")

        st.divider()

        # Knowledge Base Info
        if st.session_state.agent_ready:
            agent = st.session_state.agent
            sources = agent.rag.get_sources()
            if sources:
                st.markdown("**📚 Loaded Sources:**")
                for src in sources:
                    st.markdown(f"  - `{src}`")

        st.divider()

        # Admin Panel
        st.markdown("### 🔐 Admin Panel")

        if st.button("📊 Generate PDF Report", use_container_width=True):
            if st.session_state.agent_ready:
                st.session_state.messages.append({
                    "role": "user",
                    "content": "Generate a support ticket report",
                    "meta": {}
                })
                st.rerun()
            else:
                st.warning("Start the agent first.")

        # Show PDF download if last tool was generate_report
        last_result = st.session_state.get("last_tool_result")
        if last_result and last_result.get("pdf_path"):
            pdf_path = last_result["pdf_path"]
            if os.path.exists(pdf_path):
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download PDF Report",
                        data=f.read(),
                        file_name=os.path.basename(pdf_path),
                        mime="application/pdf",
                        use_container_width=True,
                    )

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_steps = []
            st.session_state.last_tool_result = None
            st.session_state.last_rag_chunks = []
            if st.session_state.agent_ready:
                st.session_state.agent.clear_history()
            st.rerun()

        st.divider()

        # Quick Test Prompts
        st.markdown("### 💡 Try These")
        sample_prompts = [
            "Track order ORD-001",
            "Send me an email about order ORD-013",
            "What's your return policy?",
            "I have a broken laptop, help me",
            "Generate a PDF report",
            "Check order for sarah.j@email.com",
        ]
        for prompt in sample_prompts:
            if st.button(f"▶ {prompt}", key=f"quick_{prompt}", use_container_width=True):
                if st.session_state.agent_ready:
                    st.session_state.messages.append({
                        "role": "user",
                        "content": prompt,
                        "meta": {}
                    })
                    st.rerun()
                else:
                    st.warning("Start the agent first.")


# ── Chat Display ───────────────────────────────────────────────────────────────

def render_message(msg: dict):
    role = msg["role"]
    content = msg["content"]

    if role == "user":
        st.markdown(f"""
        <div style='display:flex; justify-content:flex-end; margin:6px 0;'>
            <div class='user-bubble'>
                <small style='opacity:0.7;'>👤 You</small><br>
                {content}
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif role == "assistant":
        meta = msg.get("meta", {})
        intent = meta.get("intent", "")
        tool_used = meta.get("tool_used", "")

        badges = ""
        if intent:
            badges += f"<span style='background:#1e3a5f;color:#93c5fd;padding:1px 8px;border-radius:8px;font-size:0.7rem;margin-right:4px;'>🎯 {intent}</span>"
        if tool_used:
            badges += f"<span style='background:#1a3a2a;color:#86efac;padding:1px 8px;border-radius:8px;font-size:0.7rem;'>🔧 {tool_used}</span>"

        st.markdown(f"""
        <div style='display:flex; justify-content:flex-start; margin:6px 0;'>
            <div class='agent-bubble'>
                <small style='opacity:0.6;'>🤖 Alex (AI Agent)</small>&nbsp;{badges}<br><br>
                {content.replace(chr(10), '<br>')}
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif role == "system":
        st.info(content)


def render_chat():
    """Render all chat messages."""
    if not st.session_state.messages:
        st.markdown("""
        <div style='text-align:center; padding: 60px 20px; color: #64748b;'>
            <div style='font-size:3rem;'>💬</div>
            <h3 style='color:#94a3b8;'>Welcome to TechStore Support</h3>
            <p>I'm Alex, your AI customer support agent.<br>
            Ask me about orders, shipping, returns, or any product questions!</p>
            <p style='font-size:0.85rem;'>
            Try: <em>"Track my order ORD-001"</em> or <em>"What's your return policy?"</em>
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            render_message(msg)


# ── Agent Step Visualization ───────────────────────────────────────────────────

def render_pipeline_steps(steps: list):
    """Render the agent's pipeline steps in an expander."""
    if not steps:
        return

    with st.expander("🔍 Agent Pipeline Steps", expanded=False):
        for step in steps:
            icon = "✅" if step.result and step.result != "skipped" else "⏭️"
            st.markdown(f"""
            <div class='step-card'>
                <div class='step-title'>{icon} {step.step}</div>
                <div>{step.description}</div>
            </div>
            """, unsafe_allow_html=True)


def render_rag_context(chunks: list):
    """Show retrieved RAG context."""
    if not chunks:
        return

    with st.expander(f"📚 RAG Context ({len(chunks)} chunks retrieved)", expanded=False):
        for i, chunk in enumerate(chunks, 1):
            with st.container():
                st.markdown(f"**Chunk {i}:**")
                st.code(chunk[:400] + ("..." if len(chunk) > 400 else ""), language=None)


def render_tool_result(tool_name: str, result: dict):
    """Show tool execution result, with PDF download if applicable."""
    if not result:
        return

    with st.expander(f"🔧 Tool Result: `{tool_name}`", expanded=False):
        # PDF download button
        pdf_path = result.get("pdf_path")
        if pdf_path and os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label=f"⬇️ Download {os.path.basename(pdf_path)}",
                    data=f.read(),
                    file_name=os.path.basename(pdf_path),
                    mime="application/pdf",
                    use_container_width=True,
                )
            st.success(f"PDF saved: `{os.path.basename(pdf_path)}`")

        # Email mode badge
        if result.get("mode"):
            mode = result["mode"]
            badge_color = "#15803d" if mode == "real" else "#1d4ed8" if mode == "simulated" else "#991b1b"
            st.markdown(
                f'<span style="background:{badge_color};color:white;padding:2px 10px;'
                f'border-radius:8px;font-size:0.75rem;">✉️ {mode.upper()}</span>',
                unsafe_allow_html=True
            )
            st.write("")

        st.json(result)


# ── Process User Message ───────────────────────────────────────────────────────

def process_message(user_input: str):
    """Run the agent pipeline and update session state."""
    if not st.session_state.agent_ready:
        st.error("Agent not initialized. Please start the agent first.")
        return

    agent = st.session_state.agent
    st.session_state.total_messages += 1

    # Show typing indicator
    with st.spinner("🤔 Alex is thinking..."):
        try:
            response = agent.run(user_input)
        except Exception as e:
            st.error(f"❌ Agent error: {str(e)}")
            return

    # Track tickets
    if response.tool_used == "create_support_ticket":
        if response.tool_result and response.tool_result.get("success"):
            st.session_state.tickets_created += 1

    # Save steps for display
    st.session_state.last_steps = response.steps
    st.session_state.last_tool_result = response.tool_result
    st.session_state.last_rag_chunks = response.rag_context

    # Add assistant response to messages
    st.session_state.messages.append({
        "role": "assistant",
        "content": response.final_answer,
        "meta": {
            "intent": response.intent,
            "tool_used": response.tool_used,
            "timestamp": datetime.now().strftime("%H:%M"),
        }
    })


# ── Main Layout ────────────────────────────────────────────────────────────────

def main():
    render_sidebar()

    # Main content area
    st.markdown("""
    <div style='padding:10px 0 20px 0;'>
        <h1 style='color:#60a5fa; margin:0;'>🤖 TechStore AI Support Agent</h1>
        <p style='color:#64748b; margin:0;'>Powered by Gemini 2.5 Flash · Multi-Step RAG Pipeline</p>
    </div>
    """, unsafe_allow_html=True)

    # Auto-initialize if API key is present
    if not st.session_state.agent_ready:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if api_key:
            initialize_agent()

    # Two-column layout: Chat | Inspector
    chat_col, inspector_col = st.columns([2, 1])

    with chat_col:
        # Chat container
        chat_container = st.container(height=500)
        with chat_container:
            render_chat()

        # Handle pending user messages (from sidebar quick prompts)
        pending = None
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            last_msg = st.session_state.messages[-1]
            if not last_msg.get("meta", {}).get("processed"):
                pending = last_msg["content"]
                # Mark as processed
                st.session_state.messages[-1]["meta"] = {"processed": True}

        if pending:
            process_message(pending)
            st.rerun()

        # Chat input
        with st.form(key="chat_form", clear_on_submit=True):
            input_col, btn_col = st.columns([5, 1])
            with input_col:
                user_input = st.text_input(
                    "Message",
                    placeholder="Type your message... (e.g. 'Track order ORD-001')",
                    label_visibility="collapsed",
                )
            with btn_col:
                submitted = st.form_submit_button(
                    "Send",
                    use_container_width=True,
                    type="primary",
                    disabled=not st.session_state.agent_ready,
                )

        if submitted and user_input.strip():
            # Add user message
            st.session_state.messages.append({
                "role": "user",
                "content": user_input.strip(),
                "meta": {"processed": True}
            })
            process_message(user_input.strip())
            st.rerun()

    with inspector_col:
        st.markdown("### 🔬 Agent Inspector")

        if not st.session_state.agent_ready:
            st.info("Start the agent to see pipeline details here.")
        else:
            # Pipeline steps
            render_pipeline_steps(st.session_state.last_steps)

            # RAG context
            render_rag_context(st.session_state.last_rag_chunks)

            # Tool result
            if st.session_state.last_tool_result and st.session_state.messages:
                last_meta = next(
                    (m.get("meta", {}) for m in reversed(st.session_state.messages)
                     if m["role"] == "assistant"),
                    {}
                )
                tool_name = last_meta.get("tool_used", "tool")
                render_tool_result(tool_name, st.session_state.last_tool_result)

            # Quick status
            st.markdown("---")
            st.markdown("**📋 Knowledge Base Status**")
            agent = st.session_state.agent
            if agent.rag.is_ready:
                st.success(f"✅ {agent.rag.chunk_count} chunks indexed")
                sources = agent.rag.get_sources()
                for src in sources:
                    st.markdown(f"  📄 `{src}`")
            else:
                st.warning("Knowledge base empty")

            st.markdown("---")
            st.markdown("**🔧 Available Tools**")
            tools_info = [
                ("check_order_status", "📦", "Look up orders"),
                ("create_support_ticket", "🎫", "Create tickets"),
                ("generate_report", "📊", "Admin reports"),
                ("simulate_send_email", "✉️", "Send emails"),
            ]
            for tool_name, icon, desc in tools_info:
                st.markdown(f"{icon} `{tool_name}` — {desc}")

    # ── History Tab ────────────────────────────────────────────────────────────
    if st.session_state.messages:
        st.markdown("---")
        with st.expander("📜 Full Conversation History (Export)", expanded=False):
            history_data = [
                {
                    "role": m["role"],
                    "content": m["content"],
                    "intent": m.get("meta", {}).get("intent", ""),
                    "tool": m.get("meta", {}).get("tool_used", ""),
                }
                for m in st.session_state.messages
            ]
            history_json = json.dumps(history_data, indent=2)
            st.json(history_data)
            st.download_button(
                "⬇️ Download History (JSON)",
                data=history_json,
                file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )


if __name__ == "__main__":
    main()
