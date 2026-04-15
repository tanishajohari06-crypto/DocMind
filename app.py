import streamlit as st
import fitz
import numpy as np
import faiss
import os
import time
import json
from groq import Groq
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind — Multi-LLM RAG Comparison",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700&IBM+Plex+Mono:wght@400;600;700&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:        #ddeeff;
    --surface:   #c8e0f8;
    --border:    #a8c8e8;
    --accent:    #00ff88;
    --accent2:   #ff6b35;
    --accent3:   #7eb8f7;
    --text:      #1a2a4a;
    --muted:     #4a6080;
    --card:      #e8f4ff;
    --llm1:      #00aa55;
    --llm2:      #ff6b35;
    --llm3:      #2a7ab8;
    --llm4:      #c8980a;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'IBM Plex Sans', sans-serif;
}
[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer { visibility: hidden; }
h1, h2, h3, h4 { font-family: 'Orbitron', sans-serif !important; }

.hero {
    padding: 2rem 0 1rem 0;
    text-align: center;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
.hero h1 {
    font-size: 2.4rem;
    font-weight: 700;
    letter-spacing: -2px;
    color: #3b1f8c;
    margin-bottom: 0.2rem;
}
.hero .subtitle {
    color: var(--muted);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}

/* Comparison Grid */
.comparison-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin: 1rem 0;
}
.llm-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
}
.llm-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.llm-card.llm1::before { background: var(--llm1); }
.llm-card.llm2::before { background: var(--llm2); }
.llm-card.llm3::before { background: var(--llm3); }
.llm-card.llm4::before { background: var(--llm4); }

.llm-card .model-name {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.6rem;
    font-weight: 600;
}
.llm-card.llm1 .model-name { color: var(--llm1); }
.llm-card.llm2 .model-name { color: var(--llm2); }
.llm-card.llm3 .model-name { color: var(--llm3); }
.llm-card.llm4 .model-name { color: var(--llm4); }

.llm-card .answer-text {
    color: var(--text);
    font-size: 0.9rem;
    line-height: 1.7;
    margin-bottom: 0.8rem;
}
.llm-card .meta-row {
    display: flex;
    gap: 0.8rem;
    flex-wrap: wrap;
}
.meta-chip {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 0.18rem 0.6rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: var(--muted);
}
.meta-chip.good { border-color: var(--accent); color: var(--accent); }
.meta-chip.warn { border-color: var(--accent2); color: var(--accent2); }

/* Stats Row */
.stats-row {
    display: flex;
    gap: 1rem;
    margin: 1rem 0;
}
.stat-card {
    flex: 1;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.stat-card .number {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    color: var(--accent);
    font-weight: 700;
}
.stat-card .label {
    color: var(--muted);
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.2rem;
    font-family: 'IBM Plex Mono', monospace;
}

/* Winner Badge */
.winner-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(0,255,136,0.08);
    border: 1px solid var(--accent);
    border-radius: 20px;
    padding: 0.3rem 0.9rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: var(--accent);
    margin: 0.5rem 0;
}

/* Score Table */
.score-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    margin: 1rem 0;
}
.score-table th {
    color: var(--muted);
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 0.5rem 0.8rem;
    text-align: left;
    border-bottom: 1px solid var(--border);
}
.score-table td {
    padding: 0.6rem 0.8rem;
    border-bottom: 1px solid var(--border);
    color: var(--text);
}
.score-table tr:hover td { background: var(--surface); }
.rank-1 { color: var(--accent) !important; font-weight: 600; }
.rank-2 { color: var(--llm3) !important; }
.rank-3 { color: var(--llm4) !important; }

/* Source Cards */
.source-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.84rem;
}
.src-header {
    font-family: 'IBM Plex Mono', monospace;
    color: var(--accent2);
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.3rem;
}
.src-text { color: var(--muted); line-height: 1.5; }

/* Inputs */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(0,255,136,0.1) !important;
}

.stButton > button {
    background: var(--accent) !important;
    color: #000 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.8rem !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.05em !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

[data-testid="stFileUploader"] {
    background: var(--card) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 10px !important;
}

.sidebar-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--muted);
    margin-bottom: 0.4rem;
}
.stAlert { border-radius: 8px !important; }
hr { border-color: var(--border) !important; }

.tab-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
    margin-bottom: 0.8rem;
}

.error-box {
    background: rgba(255, 107, 53, 0.08);
    border: 1px solid var(--accent2);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-size: 0.84rem;
    color: var(--accent2);
    font-family: 'IBM Plex Mono', monospace;
}

.bar-container {
    background: var(--surface);
    border-radius: 4px;
    height: 8px;
    overflow: hidden;
    margin-top: 0.3rem;
}
.bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.5s ease;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MODEL CONFIG — all available on Groq
# ─────────────────────────────────────────────
MODELS = {
    "llama-3.3-70b-versatile": {
        "label":       "LLaMA 3.3 70B",
        "family":      "Meta LLaMA",
        "params":      "70B",
        "context":     "128K tokens",
        "strength":    "Best overall reasoning",
        "color_class": "llm1",
        "color_var":   "#00ff88",
    },
    "meta-llama/llama-4-scout-17b-16e-instruct": {
        "label":       "Llama 4 Scout 17B",
        "family":      "Meta LLaMA",
        "params":      "17B (16E MoE)",
        "context":     "131K tokens",
        "strength":    "Fast multimodal model",
        "color_class": "llm2",
        "color_var":   "#ff6b35",
    },
    "qwen/qwen3-32b": {
        "label":       "Qwen3 32B",
        "family":      "Alibaba Cloud",
        "params":      "32B",
        "context":     "131K tokens",
        "strength":    "Strong reasoning & multilingual",
        "color_class": "llm3",
        "color_var":   "#7eb8f7",
    },
    "openai/gpt-oss-20b": {
        "label":       "GPT-OSS 20B",
        "family":      "OpenAI",
        "params":      "20B MoE",
        "context":     "131K tokens",
        "strength":    "Ultra fast open-weight model",
        "color_class": "llm4",
        "color_var":   "#f5c842",
    },
}

# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
for key, val in {
    "index": None, "chunks": None, "embed_model": None,
    "groq_client": None, "history": [], "ready": False,
    "doc_stats": {}, "doc_names": [], "selected_models": list(MODELS.keys()),
    "comparison_results": []
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ─────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────
def load_pdfs(uploaded_files):
    pages = []
    for uf in uploaded_files:
        data = uf.read()
        doc  = fitz.open(stream=data, filetype="pdf")
        for page_num, page in enumerate(doc):
            text = page.get_text().strip()
            if text:
                pages.append({"text": text, "source": uf.name, "page": page_num + 1})
    return pages

def split_chunks(pages, size=500, overlap=50):
    chunks = []
    for p in pages:
        text, start = p["text"], 0
        while start < len(text):
            chunks.append({"text": text[start:start+size], "source": p["source"], "page": p["page"]})
            start += size - overlap
    return chunks

@st.cache_resource
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def build_index(chunks, model):
    texts = [c["text"] for c in chunks]
    embs  = model.encode(texts, show_progress_bar=False).astype("float32")
    idx   = faiss.IndexFlatL2(embs.shape[1])
    idx.add(embs)
    return idx

def retrieve_chunks(query, top_k=6, diverse=True):
    qv   = st.session_state.embed_model.encode([query]).astype("float32")
    if diverse:
        D, I = st.session_state.index.search(qv, min(50, st.session_state.index.ntotal))
        best = {}
        for dist, idx in zip(D[0], I[0]):
            chunk = st.session_state.chunks[idx]
            src   = chunk["source"]
            if src not in best or dist < best[src]["distance"]:
                best[src] = {**chunk, "distance": round(float(dist), 4)}
        return sorted(best.values(), key=lambda x: x["distance"])
    else:
        D, I = st.session_state.index.search(qv, top_k)
        return [{**st.session_state.chunks[i], "distance": round(float(d), 4)} for d, i in zip(D[0], I[0])]

def build_prompt(query, hits):
    context = "\n\n".join(
        f"[Source {i+1}: {h['source']}, Page {h['page']}]\n{h['text']}"
        for i, h in enumerate(hits)
    )
    return f"""You are a precise research assistant. Answer using ONLY the context below.
Cite which source(s) support your answer. Be thorough but concise.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

def query_single_model(model_id, prompt, client):
    try:
        t_start = time.time()
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.2,
        )
        elapsed = round(time.time() - t_start, 2)
        answer  = resp.choices[0].message.content.strip()
        tokens  = resp.usage.completion_tokens if resp.usage else 0
        return {"answer": answer, "time": elapsed, "tokens": tokens, "error": None}
    except Exception as e:
        return {"answer": "", "time": 0, "tokens": 0, "error": str(e)}

def estimate_length_score(text):
    words = len(text.split())
    if words < 30:  return "short", "warn"
    if words > 250: return "detailed", "good"
    return "concise", "good"

def words_per_second(tokens, secs):
    if secs == 0: return "—"
    return f"{round(tokens/secs, 1)} tok/s"

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("##  DocMind")
    st.markdown("<div class='sidebar-label'>Groq API Key</div>", unsafe_allow_html=True)
    api_key = st.text_input("", type="password", placeholder="gsk_...", label_visibility="collapsed")
    if api_key:
        st.session_state.groq_client = Groq(api_key=api_key)
        st.success("✓ API key set")

    st.markdown("---")
    st.markdown("<div class='sidebar-label'>Models to Compare</div>", unsafe_allow_html=True)
    selected = []
    for mid, mcfg in MODELS.items():
        checked = st.checkbox(
            f"{mcfg['label']} ({mcfg['params']})",
            value=mid in st.session_state.selected_models,
            key=f"chk_{mid}"
        )
        if checked:
            selected.append(mid)
    st.session_state.selected_models = selected if selected else [list(MODELS.keys())[0]]

    st.markdown("---")
    st.markdown("<div class='sidebar-label'>Retrieval Settings</div>", unsafe_allow_html=True)
    diverse_mode = st.toggle("One best chunk per document", value=True)
    if not diverse_mode:
        top_k = st.slider("Top-K chunks", 2, 15, 6)
    else:
        top_k = 6

    chunk_size = st.slider("Chunk size (chars)", 200, 1000, 500, step=50)
    overlap    = st.slider("Chunk overlap", 0, 150, 50, step=10)

    st.markdown("---")
    st.markdown("<div class='sidebar-label'>Upload Documents</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)

    if uploaded and api_key:
        if st.button(" Build Index"):
            with st.spinner("Indexing documents..."):
                pages  = load_pdfs(uploaded)
                chunks = split_chunks(pages, chunk_size, overlap)
                model  = load_embed_model()
                index  = build_index(chunks, model)
                st.session_state.update({
                    "index": index, "chunks": chunks, "embed_model": model,
                    "ready": True,
                    "doc_names": list({u.name for u in uploaded}),
                    "doc_stats": {"docs": len(uploaded), "pages": len(pages), "chunks": len(chunks)}
                })
            st.success("Index ready ✓")

    if st.session_state.doc_names:
        st.markdown("---")
        st.markdown("<div class='sidebar-label'>Loaded Documents</div>", unsafe_allow_html=True)
        for name in st.session_state.doc_names:
            st.markdown(f"📄 `{name}`")

# ─────────────────────────────────────────────
#  MAIN AREA
# ─────────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <h1>DocMind</h1>
    <p class='subtitle'>Multi-LLM RAG Comparison System · Gen AI Assessment</p>
</div>
""", unsafe_allow_html=True)

# Model info table
with st.expander(" Model Comparison Reference", expanded=False):
    cols = st.columns(len(MODELS))
    for col, (mid, mcfg) in zip(cols, MODELS.items()):
        active = mid in st.session_state.selected_models
        with col:
            st.markdown(f"""
            <div class='llm-card {mcfg["color_class"]}' style='opacity:{"1" if active else "0.4"}'>
                <div class='model-name'>{mcfg["label"]}</div>
                <div style='font-size:0.78rem;color:var(--muted);margin-bottom:0.3rem'>{mcfg["family"]}</div>
                <div style='font-size:0.8rem;margin-bottom:0.2rem'> {mcfg["params"]}</div>
                <div style='font-size:0.8rem;margin-bottom:0.2rem'> {mcfg["context"]}</div>
                <div style='font-size:0.78rem;color:var(--muted)'>{mcfg["strength"]}</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")

if st.session_state.ready:
    s = st.session_state.doc_stats
    st.markdown(f"""
    <div class='stats-row'>
        <div class='stat-card'><div class='number'>{s['docs']}</div><div class='label'>Documents</div></div>
        <div class='stat-card'><div class='number'>{s['pages']}</div><div class='label'>Pages</div></div>
        <div class='stat-card'><div class='number'>{s['chunks']}</div><div class='label'>Chunks</div></div>
        <div class='stat-card'><div class='number'>{len(st.session_state.selected_models)}</div><div class='label'>Models Active</div></div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  QUERY SECTION
# ─────────────────────────────────────────────
if not api_key:
    st.info("Paste your Groq API key in the sidebar.")
elif not uploaded:
    st.info("Upload one or more PDFs, then click **Build Index**.")
elif not st.session_state.ready:
    st.info("Click ** Build Index** in the sidebar.")
else:
    tab1, tab2, tab3 = st.tabs([" Compare Models", "Analytics", " History"])

    # ── TAB 1: Compare ──────────────────────────────
    with tab1:
        st.markdown("### Ask a Question — See All Models Answer")
        query = st.text_input("", placeholder="e.g. Summarize the main findings across all documents", label_visibility="collapsed")

        col1, col2 = st.columns([1, 5])
        with col1:
            run_btn = st.button(" Run Comparison")

        if run_btn and query and st.session_state.selected_models:
            hits   = retrieve_chunks(query, top_k, diverse_mode)
            prompt = build_prompt(query, hits)
            results = {}

            progress = st.progress(0, text="Querying models...")
            for i, mid in enumerate(st.session_state.selected_models):
                progress.progress((i) / len(st.session_state.selected_models), text=f"Querying {MODELS[mid]['label']}...")
                results[mid] = query_single_model(mid, prompt, st.session_state.groq_client)
            progress.progress(1.0, text="Done ✓")
            time.sleep(0.3)
            progress.empty()

            # Sort by speed for ranking
            successful = {k: v for k, v in results.items() if not v["error"]}
            ranked     = sorted(successful.keys(), key=lambda k: successful[k]["time"])

            # ── Winner badge ──
            if ranked:
                fastest = ranked[0]
                st.markdown(f"""
                <div class='winner-badge'>
                     Fastest: {MODELS[fastest]['label']} — {successful[fastest]['time']}s
                </div>
                """, unsafe_allow_html=True)

            # ── Comparison grid ──
            n_models = len(st.session_state.selected_models)
            if n_models == 1:
                grid_cols = st.columns(1)
            elif n_models == 2:
                grid_cols = st.columns(2)
            elif n_models == 3:
                grid_cols = st.columns(3)
            else:
                grid_cols = st.columns(2)

            model_list = st.session_state.selected_models
            for idx, mid in enumerate(model_list):
                col_idx = idx % len(grid_cols)
                mcfg    = MODELS[mid]
                res     = results[mid]
                rank    = ranked.index(mid) + 1 if mid in ranked else "—"
                length_label, length_cls = estimate_length_score(res["answer"]) if not res["error"] else ("—", "warn")

                with grid_cols[col_idx]:
                    if res["error"]:
                        st.markdown(f"""
                        <div class='llm-card {mcfg["color_class"]}'>
                            <div class='model-name'>{mcfg["label"]}</div>
                            <div class='error-box'>⚠ {res["error"][:120]}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        wps = words_per_second(res["tokens"], res["time"])
                        answer_preview = res["answer"][:800] + ("..." if len(res["answer"]) > 800 else "")
                        st.markdown(f"""
                        <div class='llm-card {mcfg["color_class"]}'>
                            <div class='model-name'>#{rank} — {mcfg["label"]}</div>
                            <div class='answer-text'>{answer_preview}</div>
                            <div class='meta-row'>
                                <span class='meta-chip good'>⏱ {res['time']}s</span>
                                <span class='meta-chip'>{res['tokens']} tokens</span>
                                <span class='meta-chip'>{wps}</span>
                                <span class='meta-chip {length_cls}'>{length_label}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

            # ── Scoring Table ──
            st.markdown("---")
            st.markdown("####  Head-to-Head Metrics")

            max_time   = max((r["time"] for r in successful.values()), default=1) or 1
            max_tokens = max((r["tokens"] for r in successful.values()), default=1) or 1

            table_html = """
            <table class='score-table'>
            <thead><tr>
                <th>Rank</th><th>Model</th><th>Family</th><th>Params</th>
                <th>Response Time</th><th>Tokens Out</th><th>Speed</th><th>Length</th>
            </tr></thead><tbody>
            """
            for rank_pos, mid in enumerate(ranked, 1):
                mcfg = MODELS[mid]
                res  = successful[mid]
                wps  = words_per_second(res["tokens"], res["time"])
                ll, _= estimate_length_score(res["answer"])
                rc   = f"rank-{rank_pos}" if rank_pos <= 3 else ""
                bar_w = int((1 - res["time"] / max_time) * 100)
                table_html += f"""
                <tr>
                    <td class='{rc}'>#{rank_pos}</td>
                    <td class='{rc}'>{mcfg["label"]}</td>
                    <td>{mcfg["family"]}</td>
                    <td>{mcfg["params"]}</td>
                    <td>
                        {res['time']}s
                        <div class='bar-container'><div class='bar-fill' style='width:{bar_w}%;background:{mcfg["color_var"]}'></div></div>
                    </td>
                    <td>{res['tokens']}</td>
                    <td>{wps}</td>
                    <td>{ll}</td>
                </tr>
                """
            table_html += "</tbody></table>"
            st.markdown(table_html, unsafe_allow_html=True)

            # ── Full answers expander ──
            st.markdown("---")
            st.markdown("####  Full Answers")
            for mid in st.session_state.selected_models:
                mcfg = MODELS[mid]
                res  = results[mid]
                if not res["error"]:
                    with st.expander(f"{mcfg['label']} — Full Answer"):
                        st.markdown(res["answer"])

            # ── Retrieved Sources ──
            st.markdown("---")
            st.markdown(f"####  Retrieved Context ({len(hits)} chunks from {len(set(h['source'] for h in hits))} documents)")
            for i, h in enumerate(hits):
                st.markdown(f"""
                <div class='source-card'>
                    <div class='src-header'>Chunk {i+1} — {h['source']} · Page {h['page']} · Dist: {h['distance']}</div>
                    <div class='src-text'>{h['text'][:350]}{'...' if len(h['text']) > 350 else ''}</div>
                </div>
                """, unsafe_allow_html=True)

            # Save to history
            st.session_state.comparison_results.append({
                "query": query, "results": results, "hits": hits,
                "ranked": ranked, "timestamp": time.strftime("%H:%M:%S")
            })

    # ── TAB 2: Analytics ────────────────────────────
    with tab2:
        st.markdown("### Cumulative Performance Analytics")
        if not st.session_state.comparison_results:
            st.info("Run at least one comparison to see analytics.")
        else:
            # Aggregate stats per model
            agg = {mid: {"times": [], "tokens": [], "queries": 0} for mid in MODELS}
            for run in st.session_state.comparison_results:
                for mid, res in run["results"].items():
                    if not res["error"]:
                        agg[mid]["times"].append(res["time"])
                        agg[mid]["tokens"].append(res["tokens"])
                        agg[mid]["queries"] += 1

            st.markdown("#### Average Response Time by Model")
            times_data = {mid: (sum(d["times"]) / len(d["times"])) for mid, d in agg.items() if d["times"]}
            if times_data:
                fastest_mid = min(times_data, key=times_data.get)
                max_t = max(times_data.values()) or 1
                for mid, avg_t in sorted(times_data.items(), key=lambda x: x[1]):
                    mcfg = MODELS[mid]
                    bar_pct = int((1 - avg_t / max_t) * 100)
                    badge = "  FASTEST" if mid == fastest_mid else ""
                    st.markdown(f"""
                    <div style='margin:0.6rem 0'>
                        <div style='font-family:IBM Plex Mono,monospace;font-size:0.78rem;color:{mcfg["color_var"]};margin-bottom:0.3rem'>
                            {mcfg["label"]}{badge} — avg {avg_t:.2f}s over {agg[mid]["queries"]} queries
                        </div>
                        <div class='bar-container' style='height:12px'>
                            <div class='bar-fill' style='width:{bar_pct}%;background:{mcfg["color_var"]}'></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("#### Average Tokens Generated")
            tokens_data = {mid: (sum(d["tokens"]) / len(d["tokens"])) for mid, d in agg.items() if d["tokens"]}
            if tokens_data:
                max_tok = max(tokens_data.values()) or 1
                for mid, avg_tok in sorted(tokens_data.items(), key=lambda x: -x[1]):
                    mcfg = MODELS[mid]
                    bar_pct = int((avg_tok / max_tok) * 100)
                    st.markdown(f"""
                    <div style='margin:0.6rem 0'>
                        <div style='font-family:IBM Plex Mono,monospace;font-size:0.78rem;color:{mcfg["color_var"]};margin-bottom:0.3rem'>
                            {mcfg["label"]} — avg {avg_tok:.0f} tokens
                        </div>
                        <div class='bar-container' style='height:12px'>
                            <div class='bar-fill' style='width:{bar_pct}%;background:{mcfg["color_var"]}'></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("#### Win Rate (Fastest Per Query)")
            wins = {mid: 0 for mid in MODELS}
            for run in st.session_state.comparison_results:
                if run["ranked"]:
                    wins[run["ranked"][0]] += 1
            total_runs = len(st.session_state.comparison_results)
            for mid, w in sorted(wins.items(), key=lambda x: -x[1]):
                if w > 0:
                    mcfg = MODELS[mid]
                    pct  = int((w / total_runs) * 100)
                    st.markdown(f"""
                    <div style='margin:0.5rem 0;font-family:IBM Plex Mono,monospace;font-size:0.8rem'>
                        <span style='color:{mcfg["color_var"]}'>{mcfg["label"]}</span>
                        &nbsp;—&nbsp; {w}/{total_runs} wins ({pct}%)
                    </div>
                    """, unsafe_allow_html=True)

    # ── TAB 3: History ──────────────────────────────
    with tab3:
        st.markdown("### Query History")
        if not st.session_state.comparison_results:
            st.info("No queries yet.")
        else:
            for run in reversed(st.session_state.comparison_results):
                with st.expander(f"[{run['timestamp']}] {run['query'][:80]}"):
                    st.markdown(f"**Fastest:** {MODELS[run['ranked'][0]]['label'] if run['ranked'] else '—'}")
                    for mid in run["results"]:
                        res  = run["results"][mid]
                        mcfg = MODELS[mid]
                        if not res["error"]:
                            st.markdown(f"**{mcfg['label']}** ({res['time']}s)")
                            st.markdown(res["answer"][:400] + ("..." if len(res["answer"]) > 400 else ""))
                            st.markdown("---")
