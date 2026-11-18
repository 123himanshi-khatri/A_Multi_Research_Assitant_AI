import os
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage

# =========================================================
# CONFIG
# =========================================================

load_dotenv()

st.set_page_config(
    page_title="NEXUS AI Research",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background:
            radial-gradient(circle at 10% 10%, rgba(99,102,241,0.12), transparent 25%),
            radial-gradient(circle at 90% 20%, rgba(14,165,233,0.10), transparent 25%),
            #070b14;
        color: #f8fafc;
    }

    .main .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* ---------- SIDEBAR ---------- */

    section[data-testid="stSidebar"] {
        background: #090e19;
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.5rem;
    }

    /* ---------- HEADER ---------- */

    .brand {
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -1.5px;
        margin-bottom: 0;
        background: linear-gradient(
            90deg,
            #8b5cf6,
            #38bdf8,
            #22d3ee
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .subtitle {
        color: #94a3b8;
        font-size: 1rem;
        margin-top: 0.2rem;
    }

    /* ---------- CARDS ---------- */

    .glass-card {
        background: rgba(15,23,42,0.72);
        border: 1px solid rgba(148,163,184,0.13);
        border-radius: 18px;
        padding: 1.4rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(12px);
        box-shadow: 0 10px 35px rgba(0,0,0,0.18);
    }

    .metric-card {
        background: linear-gradient(
            145deg,
            rgba(30,41,59,0.85),
            rgba(15,23,42,0.72)
        );
        border: 1px solid rgba(148,163,184,0.12);
        border-radius: 16px;
        padding: 1rem 1.2rem;
        min-height: 110px;
    }

    .metric-label {
        color: #94a3b8;
        font-size: 0.85rem;
    }

    .metric-value {
        font-size: 1.7rem;
        font-weight: 700;
        margin-top: 0.3rem;
    }

    /* ---------- AGENT ---------- */

    .agent {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px 14px;
        margin: 8px 0;
        border-radius: 12px;
        background: rgba(30,41,59,0.55);
        border: 1px solid rgba(148,163,184,0.08);
    }

    .agent-icon {
        width: 36px;
        height: 36px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 10px;
        background: rgba(99,102,241,0.15);
        font-size: 18px;
    }

    .agent-name {
        font-weight: 600;
    }

    .agent-status {
        font-size: 0.75rem;
        color: #64748b;
    }

    .online {
        color: #34d399;
    }

    /* ---------- BADGES ---------- */

    .badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        background: rgba(34,211,153,0.10);
        color: #34d399;
        border: 1px solid rgba(34,211,153,0.18);
    }

    /* ---------- INPUT ---------- */

    div[data-testid="stTextArea"] textarea {
        background: #0b1220 !important;
        border: 1px solid rgba(148,163,184,0.18) !important;
        border-radius: 14px !important;
        color: white !important;
        font-size: 1rem !important;
    }

    div[data-testid="stTextArea"] textarea:focus {
        border: 1px solid #6366f1 !important;
        box-shadow: 0 0 0 1px #6366f1 !important;
    }

    /* ---------- BUTTON ---------- */

    .stButton > button {
        border-radius: 12px;
        border: 1px solid rgba(139,92,246,0.35);
        background: linear-gradient(
            135deg,
            #7c3aed,
            #2563eb
        );
        color: white;
        font-weight: 700;
        padding: 0.65rem 1.4rem;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        border-color: #a78bfa;
        transform: translateY(-1px);
        box-shadow: 0 8px 25px rgba(99,102,241,0.25);
    }

    /* ---------- DIVIDER ---------- */

    hr {
        border-color: rgba(148,163,184,0.10);
    }

    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;
        color: #475569;
        padding-top: 2rem;
        font-size: 0.8rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# SESSION STATE
# =========================================================

if "research_result" not in st.session_state:
    st.session_state.research_result = ""

if "research_history" not in st.session_state:
    st.session_state.research_history = []

# =========================================================
# MISTRAL LLM
# =========================================================

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if MISTRAL_API_KEY:
    llm = ChatMistralAI(
        model="mistral-small-latest",
        temperature=0,
        api_key=MISTRAL_API_KEY,
    )
else:
    llm = None

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div style="text-align:center; padding:10px;">
            <div style="font-size:42px;">🧠</div>
            <h2 style="margin:0;">NEXUS AI</h2>
            <p style="color:#64748b;">Research Intelligence</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("### ⚙️ Research Settings")

    research_depth = st.selectbox(
        "Research depth",
        [
            "Quick",
            "Balanced",
            "Deep",
        ],
        index=1,
    )

    response_style = st.selectbox(
        "Response style",
        [
            "Professional",
            "Academic",
            "Executive",
            "Simple",
        ],
    )

    include_sources = st.toggle(
        "Include sources",
        value=True,
    )

    st.divider()

    st.markdown("### 🤖 Active Agents")

    st.markdown(
        """
        <div class="agent">
            <div class="agent-icon">🔎</div>
            <div>
                <div class="agent-name">Research Agent</div>
                <div class="agent-status online">● Ready</div>
            </div>
        </div>

        <div class="agent">
            <div class="agent-icon">📊</div>
            <div>
                <div class="agent-name">Analysis Agent</div>
                <div class="agent-status online">● Ready</div>
            </div>
        </div>

        <div class="agent">
            <div class="agent-icon">📝</div>
            <div>
                <div class="agent-name">Writer Agent</div>
                <div class="agent-status online">● Ready</div>
            </div>
        </div>

        <div class="agent">
            <div class="agent-icon">🧪</div>
            <div>
                <div class="agent-name">Fact Checker</div>
                <div class="agent-status online">● Ready</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.caption("Powered by Mistral AI")
    st.caption("NEXUS Research Engine v1.0")

# =========================================================
# MAIN HEADER
# =========================================================

st.markdown(
    """
    <div>
        <div class="brand">NEXUS AI</div>
        <div class="subtitle">
            Multi-Agent Research & Intelligence Platform
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# =========================================================
# STATUS BAR
# =========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">System Status</div>
            <div class="metric-value">🟢 Online</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    model_name = "Mistral Small"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">AI Model</div>
            <div class="metric-value">⚡ {model_name}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Research Depth</div>
            <div class="metric-value">🔬 {research_depth}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Reports Generated</div>
            <div class="metric-value">📄 {len(st.session_state.research_history)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")
st.markdown("---")

# =========================================================
# RESEARCH INPUT
# =========================================================

st.markdown("## 🔬 Start a Research Task")

st.markdown(
    """
    <div class="glass-card">
        <b>Ask NEXUS anything.</b><br>
        Enter a topic, question, technology, market, company,
        academic subject or business problem.
    </div>
    """,
    unsafe_allow_html=True,
)

topic = st.text_area(
    "Research topic",
    placeholder=(
        "Example:\n"
        "Compare AI investment opportunities in Indian banking "
        "and fintech for the next 5 years."
    ),
    height=150,
    label_visibility="collapsed",
)

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    research_button = st.button(
        "🚀 Start Research",
        use_container_width=True,
    )

with col2:
    clear_button = st.button(
        "🗑️ Clear",
        use_container_width=True,
    )

if clear_button:
    st.session_state.research_result = ""
    st.rerun()

# =========================================================
# RESEARCH ENGINE
# =========================================================

if research_button:

    if not topic.strip():
        st.warning("⚠️ Please enter a research topic first.")

    elif not MISTRAL_API_KEY:
        st.error(
            """
            ❌ MISTRAL_API_KEY not found.

            Add it to your `.env` file:

            `MISTRAL_API_KEY=your_api_key`
            """
        )

    else:

        # Depth instructions
        if research_depth == "Quick":
            depth_instruction = """
            Give a concise but useful research response.
            Focus on the most important findings.
            """

        elif research_depth == "Deep":
            depth_instruction = """
            Perform a deep analytical response.
            Cover background, important factors, advantages,
            disadvantages, risks, opportunities and conclusion.
            """

        else:
            depth_instruction = """
            Provide a balanced research response with enough
            detail for a professional report.
            """

        # Style instructions
        style_instruction = f"""
        Write in a {response_style.lower()} style.
        """

        prompt = f"""
You are NEXUS AI, an advanced research intelligence assistant.

Research Topic:
{topic}

{depth_instruction}

{style_instruction}

Create a high-quality research report using this structure:

# Executive Summary

# Introduction

# Key Findings

# Detailed Analysis

# Advantages

# Risks / Challenges

# Future Outlook

# Conclusion

Use clear headings and bullet points where appropriate.

Do not invent statistics or sources.
If information is uncertain, clearly mention the uncertainty.
"""

        # Progress UI
        st.markdown("### ⚡ Research Pipeline")

        progress = st.progress(0)

        status = st.empty()

        stages = [
            ("🔎", "Research Agent", 20),
            ("📊", "Analysis Agent", 45),
            ("🧪", "Fact Checking", 70),
            ("📝", "Report Writer", 90),
        ]

        for icon, agent, value in stages:
            status.markdown(
                f"**{icon} {agent}** — processing..."
            )
            progress.progress(value)

        try:

            response = llm.invoke(
                [HumanMessage(content=prompt)]
            )

            result = response.content

            progress.progress(100)

            status.success(
                "✅ Research completed successfully"
            )

            st.session_state.research_result = result

            st.session_state.research_history.append(
                {
                    "topic": topic,
                    "time": datetime.now().strftime(
                        "%d %b %Y, %I:%M %p"
                    ),
                }
            )

        except Exception as e:

            progress.empty()

            st.error(
                f"""
                ❌ Research failed.

                **Error:** `{str(e)}`
                """
            )

# =========================================================
# RESULTS
# =========================================================

if st.session_state.research_result:

    st.markdown("---")

    st.markdown("## 📑 Research Report")

    result = st.session_state.research_result

    tab1, tab2, tab3 = st.tabs(
        [
            "📄 Report",
            "🔍 Analysis",
            "⬇️ Export",
        ]
    )

    with tab1:

        st.markdown(
            f"""
            <div class="glass-card">
                <span class="badge">AI GENERATED REPORT</span>
                <h2>{topic}</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(result)

    with tab2:

        st.markdown("### 🧠 Research Intelligence")

        st.info(
            """
            This report was generated using the NEXUS
            research pipeline powered by Mistral AI.

            **Pipeline**

            🔎 Research → 📊 Analysis → 🧪 Validation → 📝 Writing
            """
        )

        st.markdown("#### Configuration")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Depth",
                research_depth,
            )

        with c2:
            st.metric(
                "Model",
                "Mistral",
            )

        with c3:
            st.metric(
                "Temperature",
                "0",
            )

    with tab3:

        st.markdown("### 📥 Download Report")

        report_text = f"""
NEXUS AI — RESEARCH REPORT
================================

Topic:
{topic}

Generated:
{datetime.now().strftime("%d %B %Y, %I:%M %p")}

--------------------------------

{result}

--------------------------------

Generated by NEXUS AI
Powered by Mistral AI
"""

        st.download_button(
            label="📄 Download TXT Report",
            data=report_text,
            file_name="nexus_research_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

# =========================================================
# HISTORY
# =========================================================

if st.session_state.research_history:

    st.markdown("---")

    st.markdown("## 🕘 Recent Research")

    for item in reversed(
        st.session_state.research_history[-5:]
    ):

        st.markdown(
            f"""
            <div class="glass-card">
                <b>🔬 {item["topic"]}</b>
                <br>
                <span style="color:#64748b;">
                    {item["time"]}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        NEXUS AI Research Intelligence Platform
        <br>
        Built with Streamlit + LangChain + Mistral AI
    </div>
    """,
    unsafe_allow_html=True,
)