"""
frontend/dashboard.py — MayaGuard Streamlit monitoring dashboard.

Run with:
    streamlit run frontend/dashboard.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import httpx
import streamlit as st
import plotly.graph_objects as go

API_BASE = "http://localhost:8080/api/v1"

# ── Helpers ────────────────────────────────────────────────────────────────────

def risk_color(score: float) -> str:
    if score >= 0.75:
        return "#e74c3c"   # red
    if score >= 0.6:
        return "#e67e22"   # orange
    if score >= 0.35:
        return "#f1c40f"   # yellow
    return "#2ecc71"       # green


def risk_label(score: float) -> str:
    if score >= 0.75:
        return "🔴 CRITICAL"
    if score >= 0.6:
        return "🔴 HIGH"
    if score >= 0.35:
        return "🟡 MEDIUM"
    return "🟢 LOW"


def gauge(value: float, title: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(value * 100, 1),
            title={"text": title, "font": {"size": 13}},
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": risk_color(value)},
                "steps": [
                    {"range": [0, 35], "color": "#d5f5e3"},
                    {"range": [35, 60], "color": "#fef9e7"},
                    {"range": [60, 100], "color": "#fadbd8"},
                ],
            },
        )
    )
    fig.update_layout(height=220, margin=dict(t=30, b=10, l=20, r=20))
    return fig


# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MayaGuard",
    page_icon="🛡️",
    layout="wide",
)

# ── Session state ──────────────────────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history: list[dict] = []

# ── Header ─────────────────────────────────────────────────────────────────────

st.title("🛡️ MayaGuard — Hallucination-Aware AI Monitor")
st.caption("Phase 1 Core Platform — domain-agnostic hallucination detection")

st.divider()

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")
    adapter = st.selectbox("Adapter", ["default"], index=0)
    top_k = st.slider("Retrieval top-k", 1, 10, 5)
    st.divider()
    st.subheader("Stats (this session)")
    total_q = len(st.session_state.history)
    if total_q:
        scores = [h["risk_score"] for h in st.session_state.history]
        high_risk = sum(1 for s in scores if s >= 0.6)
        st.metric("Queries", total_q)
        st.metric("High-risk", f"{high_risk} ({high_risk/total_q*100:.0f}%)")
        st.metric("Avg risk score", f"{sum(scores)/total_q:.2f}")
    else:
        st.info("No queries yet.")

# ── Query input ────────────────────────────────────────────────────────────────

col_input, col_btn = st.columns([5, 1])
with col_input:
    query = st.text_input("Enter your query", placeholder="Ask anything…")
with col_btn:
    st.write("")
    submitted = st.button("🔍 Submit", use_container_width=True)

# ── Submit ─────────────────────────────────────────────────────────────────────

if submitted and query.strip():
    with st.spinner("Running MayaGuard pipeline…"):
        try:
            resp = httpx.post(
                f"{API_BASE}/query",
                json={"query": query, "adapter": adapter},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.RequestError as exc:
            st.error(f"Could not reach the API: {exc}")
            data = None
        except Exception as exc:
            st.error(f"Error: {exc}")
            data = None

    if data:
        report = data["hallucination_report"]
        st.session_state.history.append(
            {
                "ts": datetime.now(tz=timezone.utc).isoformat(),
                "query": query,
                "action": data["action_taken"],
                "risk_score": report["risk_score"],
                "faithfulness": report["faithfulness_score"],
                "latency_ms": data["latency_ms"],
            }
        )

        # ── Result header ──────────────────────────────────────────
        risk_score = report["risk_score"]
        st.subheader(f"Result  {risk_label(risk_score)}")
        st.caption(
            f"Adapter: **{data['adapter_used']}** · "
            f"Action: **{data['action_taken'].upper()}** · "
            f"Latency: **{data['latency_ms']:.0f} ms**"
        )

        # ── Gauges ────────────────────────────────────────────────
        g1, g2, g3 = st.columns(3)
        g1.plotly_chart(gauge(risk_score, "Hallucination Risk"), use_container_width=True)
        g2.plotly_chart(
            gauge(report["faithfulness_score"], "Faithfulness"), use_container_width=True
        )
        g3.plotly_chart(
            gauge(1.0 - report["self_reflection_confidence"], "Reflection Uncertainty"),
            use_container_width=True,
        )

        # ── Safe answer ────────────────────────────────────────────
        st.subheader("Safe Answer")
        st.write(data["safe_answer"])

        # ── Claim analysis ─────────────────────────────────────────
        with st.expander("📋 Claim-level Analysis"):
            verdicts = report.get("claim_verdicts", [])
            if verdicts:
                for v in verdicts:
                    icon = "✅" if v["supported"] else "❌"
                    st.markdown(
                        f"{icon} **{v['claim']['text']}**  \n"
                        f"&nbsp;&nbsp;Confidence: `{v['confidence']:.2f}` — {v['explanation']}"
                    )
            else:
                st.info("No individual claim data.")

        # ── Sources ────────────────────────────────────────────────
        with st.expander("📚 Retrieved Sources"):
            docs = report.get("retrieved_documents", [])
            if docs:
                for d in docs:
                    st.markdown(
                        f"**{d['source']}** (score `{d['score']:.2f}`)  \n{d['content'][:300]}…"
                    )
            else:
                st.info("No sources retrieved.")

        # ── Self-critique ──────────────────────────────────────────
        with st.expander("🤔 Self-Reflection Critique"):
            st.write(report.get("self_critique") or "No critique available.")

# ── History table ──────────────────────────────────────────────────────────────

if st.session_state.history:
    st.divider()
    st.subheader("Session History")
    st.dataframe(
        st.session_state.history,
        use_container_width=True,
        column_config={
            "risk_score": st.column_config.ProgressColumn(
                "Risk Score", min_value=0, max_value=1, format="%.2f"
            ),
            "faithfulness": st.column_config.ProgressColumn(
                "Faithfulness", min_value=0, max_value=1, format="%.2f"
            ),
        },
    )
