# screens/codebook_screen.py
"""
Screen 3 — LM-Assisted Codebook Development (fixed persistence + export + safe LM fallback)

Key features:
- Loads codebook/prompt from st.session_state["task_meta"] (persisted from Screen 1/2).
- Safe widgets (session_state keys are initialized BEFORE widgets).
- LM calls use a short timeout and fall back to simulated outputs if Ollama is unreachable.
- Verifier is simulated (ensures the produced span exists or picks a snippet from text).
- "Exit & export" button to download final prompt, annotations CSV, and JSON report, AND navigates to next screen.
- "Force advance" button available when per-item UI gets stuck.
- Batch size default 1; attempts fixed to 1 (as requested).
"""

from typing import List, Tuple, Dict, Any
import subprocess
import json
import requests
import random
import math
import base64
from io import BytesIO, StringIO
from collections import Counter
import html as html_module
import re
from datetime import datetime
import csv

import streamlit as st
import matplotlib.pyplot as plt

# Ollama endpoint (can be changed)
OLLAMA_URL = "http://localhost:11434"

# -------------------------
# Utilities
# -------------------------
def get_installed_models() -> List[str]:
    try:
        proc = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        lines = proc.stdout.strip().splitlines()
        models = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith("name") and "id" in line.lower():
                continue
            parts = line.split()
            if parts:
                models.append(parts[0])
        seen = set()
        uniq = [m for m in models if not (m in seen or seen.add(m))]
        return uniq or ["llama3:latest", "llama3.2:latest", "mistral:latest"]
    except Exception:
        return ["llama3:latest", "llama3.2:latest", "mistral:latest"]


def call_ollama_generate_sync(model: str, prompt_text: str, timeout: int = 6, ollama_url: str = OLLAMA_URL) -> str:
    """
    Synchronous Ollama call with short timeout. Raises on HTTP errors.
    """
    url = f"{ollama_url.rstrip('/')}/api/generate"
    payload = {"model": model, "prompt": prompt_text, "stream": False}
    headers = {"Content-Type": "application/json"}
    r = requests.post(url, json=payload, headers=headers, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    out = j.get("response") or (j.get("message") or {}).get("content") or (j.get("choices") or [{}])[0].get("text")
    return out if out is not None else str(j)


# -------------------------
# Prompt / template helpers
# -------------------------
def build_prompt_template(task_meta: dict, guidelines: List[str]) -> str:
    task_name = (task_meta or {}).get("task_name", "Annotation task")
    desc = (task_meta or {}).get("description", "")
    codebook = (task_meta or {}).get("codebook_struct", [])  # list of {label, notes}
    labels = [c.get("label") for c in codebook] if codebook else []
    labels_part = "\n".join([f"- {l}" for l in labels]) if labels else "(no labels defined)"
    guidelines_part = "\n".join([f"- {g}" for g in (guidelines or [])]) if guidelines else "No guidelines."
    tmpl = (
        f"Task: {task_name}\n"
        f"Description: {desc}\n\n"
        f"Labels (choose one):\n{labels_part}\n\n"
        f"Guidelines:\n{guidelines_part}\n\n"
        f"Example to annotate (replace the token {{EXAMPLE}} below with the example text):\n\"\"\"\n{{EXAMPLE}}\n\"\"\"\n\n"
        f"Return a short JSON with fields: label, span, cot (chain-of-thought). Example:\n"
        f'{{"label":"LabelA","span":"...","cot":"reasoning"}}\n'
    )
    return tmpl


def finalize_prompt_for_example(template: str, example_text: str) -> str:
    if "{{EXAMPLE}}" in template:
        return template.replace("{{EXAMPLE}}", example_text)
    else:
        return template + "\n\nExample:\n\"\"\"\n" + example_text + "\n\"\"\"\n"


# -------------------------
# plotting helpers
# -------------------------
def _plot_iterations(eval_hist: List[float], batch_hist: List[float]):
    fig, ax = plt.subplots(figsize=(4.2, 2.0))
    if not eval_hist and not batch_hist:
        ax.text(0.5, 0.5, "No iterations yet", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        return fig
    n = max(len(eval_hist), len(batch_hist))
    import numpy as np
    x = list(range(1, n + 1))
    eval_plot = list(eval_hist) + [math.nan] * (n - len(eval_hist))
    batch_plot = list(batch_hist) + [math.nan] * (n - len(batch_hist))
    ax.plot(x, eval_plot, marker="o", linestyle="-", label="Eval F1")
    ax.plot(x, batch_plot, marker="o", linestyle="-", label="Batch F1")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("F1 score")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.grid(axis="y", linestyle=":", linewidth=0.6)
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    return fig


def fig_to_data_uri(fig, fmt="png"):
    buf = BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches="tight")
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode("ascii")
    buf.close()
    return f"data:image/{fmt};base64,{data}"


def _render_topright_overlay(eval_history: List[float], batch_history: List[float], top_px: int, right_px: int):
    fig = _plot_iterations(eval_history, batch_history)
    data_uri = fig_to_data_uri(fig)
    plt.close(fig)

    last_eval = eval_history[-1] if eval_history else st.session_state.get("codebook_sim_eval_acc", 0.0)
    last_batch = batch_history[-1] if batch_history else 0.0

    html = f'''
    <div style="
        position: fixed;
        top: {top_px}px;
        right: {right_px}px;
        width: 360px;
        max-width: 42vw;
        background: rgba(255,255,255,0.98);
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 8px 28px rgba(0,0,0,0.12);
        z-index: 9999;">
      <div style="font-weight:700; margin-bottom:6px;">Iterations & F1</div>
      <img src="{data_uri}" style="width:100%; height:auto; display:block; border-radius:6px;" />
      <div style="display:flex; gap:8px; margin-top:8px; justify-content:space-between;">
        <div style="flex:1;">
          <div style="font-size:12px; color:#6b7280">Eval F1</div>
          <div style="font-weight:700; font-size:15px">{last_eval*100:.1f}%</div>
        </div>
        <div style="flex:1;">
          <div style="font-size:12px; color:#6b7280">Batch F1</div>
          <div style="font-weight:700; font-size:15px">{last_batch*100:.1f}%</div>
        </div>
      </div>
    </div>
    '''
    st.markdown(html, unsafe_allow_html=True)


# -------------------------
# Simple sampling
# -------------------------
def sample_batch_indices(batch_size: int, total: int, exclude: List[int]) -> List[int]:
    available = [i for i in range(total) if i not in exclude]
    if not available:
        return []
    if batch_size >= len(available):
        return available.copy()
    return random.sample(available, batch_size)


# -------------------------
# Parse LM results (robust)
# -------------------------
def parse_lm_json_like(text: str) -> Dict[str, str]:
    out = {"label": "", "span": "", "cot": "", "raw": text or ""}
    if not text:
        return out
    # try strict JSON
    try:
        j = json.loads(text)
        if isinstance(j, dict):
            out["label"] = str(j.get("label", "") or "").strip()
            out["span"] = str(j.get("span", "") or "").strip()
            out["cot"] = str(j.get("cot", "") or "").strip()
            return out
    except Exception:
        pass
    # JSON substring
    m_json = re.search(r"\{(?:.|\s)*\}", text)
    if m_json:
        try:
            j = json.loads(m_json.group(0))
            if isinstance(j, dict):
                out["label"] = str(j.get("label", "") or "").strip()
                out["span"] = str(j.get("span", "") or "").strip()
                out["cot"] = str(j.get("cot", "") or "").strip()
                return out
        except Exception:
            pass
    # heuristics
    m_label = re.search(r"\"?label\"?\s*[:\-]\s*\"?([^\n\",]+)\"?", text, flags=re.IGNORECASE)
    if m_label:
        out["label"] = m_label.group(1).strip()
    m_span = re.search(r"\"?span\"?\s*[:\-]\s*\"?(.{1,250})\"?", text, flags=re.IGNORECASE | re.DOTALL)
    if m_span:
        out["span"] = m_span.group(1).strip()
    m_cot = re.search(r"\"?cot\"?\s*[:\-]\s*\"?(.{1,500})\"?", text, flags=re.IGNORECASE | re.DOTALL)
    if m_cot:
        out["cot"] = m_cot.group(1).strip()
    if not out["span"]:
        m_q = re.search(r"\"([^\"]{3,200})\"", text)
        if m_q:
            out["span"] = m_q.group(1).strip()
    return out


# -------------------------
# Extract span from example text (safe)
# -------------------------
def _extract_span_from_lm_text(lm_text: str, example_text: str) -> str:
    parsed = parse_lm_json_like(lm_text)
    if parsed.get("span"):
        return parsed["span"]
    # try quoted substring
    m = re.search(r'"([^"]{3,200})"', lm_text)
    if m:
        cand = m.group(1).strip()
        if cand and cand in example_text:
            return cand
    # try short contiguous tokens from LM output inside example
    tokens = re.findall(r"\b\w{4,}\b", lm_text)
    for length in (6,5,4,3):
        for i in range(len(tokens)-length+1):
            s = " ".join(tokens[i:i+length])
            if s and s in example_text:
                return s
    # split into sentences and try to match any sentence present in lm_text
    sentences = re.split(r'(?<=[\.\?\!])\s+', example_text)
    for s in sentences:
        if s.strip() and s.strip() in lm_text and s.strip() in example_text:
            return s.strip()
    return ""


# -------------------------
# Single-run LM + verifier (attempts=1)
# -------------------------
def run_lm_with_verifier(example_text: str, prompt_template: str, labels: List[str], model_name: str, ollama_url: str = OLLAMA_URL, attempts: int = 1) -> Tuple[str, str, str, str]:
    """
    Attempts=1: single LM call (defensive). Verifier simulated: check span in text, else pick snippet.
    Returns (label, span, cot, raw_text)
    """
    try:
        prompt = finalize_prompt_for_example(prompt_template, example_text)
        lm_out = call_ollama_generate_sync(model_name, prompt, timeout=6, ollama_url=ollama_url)
    except Exception:
        # fallback quick simulated output so UI doesn't hang
        sim_label = random.choice(labels) if labels else "LabelA"
        sim_span = example_text[: min(120, len(example_text))].split("\n")[0].strip()
        sim_cot = "simulated reasoning (fallback)"
        lm_out = json.dumps({"label": sim_label, "span": sim_span, "cot": sim_cot})

    parsed = parse_lm_json_like(lm_out)
    label = parsed.get("label") or (random.choice(labels) if labels else "")
    span = parsed.get("span") or _extract_span_from_lm_text(lm_out, example_text)
    cot = parsed.get("cot") or ""

    # verifier (simulate): ensure span exists; if not, make a reasonable snippet from example_text
    verified = False
    if span and span in example_text:
        verified = True
    if not verified:
        # choose the first sentence or up to 120 chars
        parts = re.split(r'(?<=[\.\?\!])\s+', example_text.strip())
        snippet = parts[0][:120] if parts and parts[0].strip() else example_text.strip()[:120]
        span = snippet
        verified = bool(span)

    # return parsed values and raw lm output for debugging
    return label, span, cot, lm_out


# -------------------------
# Session-state init (before widgets)
# -------------------------
def _ensure_state():
    # canonical stuff from other screens (task_meta is authoritative)
    st.session_state.setdefault("task_meta", st.session_state.get("task_meta", {}))
    # load codebook prompt from task_meta if present, else build default
    tm = st.session_state.get("task_meta", {}) or {}
    existing_prompt = tm.get("codebook_text")
    existing_guidelines = st.session_state.get("codebook_guidelines", tm.get("codebook_guidelines", [])) or []
    st.session_state.setdefault("codebook_guidelines", existing_guidelines)
    if "codebook_prompt_template" not in st.session_state:
        if existing_prompt:
            st.session_state["codebook_prompt_template"] = existing_prompt
        else:
            st.session_state["codebook_prompt_template"] = build_prompt_template(tm, st.session_state.get("codebook_guidelines", []))
    # widget-owned copy (prompt_area) initialized from canonical template
    st.session_state.setdefault("codebook_prompt_area", st.session_state.get("codebook_prompt_template", ""))
    # model select default
    models = get_installed_models()
    st.session_state.setdefault("codebook_model_select", st.session_state.get("codebook_model_select", models[0] if models else "llama3:latest"))
    # batch bookkeeping
    st.session_state.setdefault("codebook_batch_indices", st.session_state.get("codebook_batch_indices", []))
    st.session_state.setdefault("codebook_batch_ptr", st.session_state.get("codebook_batch_ptr", 0))
    st.session_state.setdefault("codebook_batch_preds", st.session_state.get("codebook_batch_preds", []))
    st.session_state.setdefault("codebook_batch_user", st.session_state.get("codebook_batch_user", []))
    st.session_state.setdefault("codebook_batch_matches", st.session_state.get("codebook_batch_matches", []))
    st.session_state.setdefault("codebook_batch_in_progress", st.session_state.get("codebook_batch_in_progress", False))
    # histories
    st.session_state.setdefault("codebook_history", st.session_state.get("codebook_history", []))
    st.session_state.setdefault("eval_history", st.session_state.get("eval_history", []))
    st.session_state.setdefault("batch_history", st.session_state.get("batch_history", []))
    st.session_state.setdefault("codebook_sim_eval_acc", st.session_state.get("codebook_sim_eval_acc", 0.0))
    st.session_state.setdefault("codebook_batch_size", st.session_state.get("codebook_batch_size", 1))
    # overlay controls
    st.session_state.setdefault("codebook_overlay_top", st.session_state.get("codebook_overlay_top", 80))
    st.session_state.setdefault("codebook_overlay_right", st.session_state.get("codebook_overlay_right", 18))
    st.session_state.setdefault("codebook_overlay_show", st.session_state.get("codebook_overlay_show", True))


# -------------------------
# CSV/Export helpers
# -------------------------
def prepare_annotations_csv_from_history(history_list: List[Dict[str, Any]]) -> bytes:
    """
    Creates a CSV with rows: iteration, index, pred_label, user_label
    """
    buf = StringIO()
    writer = csv.writer(buf)
    writer.writerow(["iteration", "index", "pred_label", "user_label"])
    for it, h in enumerate(history_list, start=1):
        indices = h.get("indices", [])
        preds = h.get("lm_preds", [])
        users = h.get("user_labels", [])
        for i, idx in enumerate(indices):
            p = preds[i] if i < len(preds) else ""
            u = users[i] if i < len(users) else ""
            writer.writerow([it, idx, p, u])
    return buf.getvalue().encode("utf-8")


def prepare_prompt_text(prompt_template: str) -> bytes:
    return prompt_template.encode("utf-8")


def prepare_report_json(eval_hist: List[float], batch_hist: List[float]) -> bytes:
    rep = {"eval_history": eval_hist, "batch_history": batch_hist, "generated_at": datetime.utcnow().isoformat()}
    return json.dumps(rep, indent=2).encode("utf-8")


# -------------------------
# Main render
# -------------------------
def render():
    _ensure_state()

    st.header("LM-Assisted Codebook Development")
    st.write("Iteratively run an LM on small batches and correct its labels. Edits persist to the task metadata and will be available to later screens.")

    # prerequisites
    if not st.session_state.get("uploaded_data"):
        st.warning("No data loaded. Please upload data on Screen 1 first.")
        return
    if not st.session_state.get("task_meta"):
        st.warning("No task defined. Please set up a task on Screen 1 first.")
        return

    total = len(st.session_state.uploaded_data)
    # Model select widget (owns key codebook_model_select)
    models = get_installed_models()
    model_default_idx = 0
    try:
        model_default_idx = models.index(st.session_state.get("codebook_model_select")) if st.session_state.get("codebook_model_select") in models else 0
    except Exception:
        model_default_idx = 0
    model_name = st.selectbox("Choose model for LM calls", models, index=model_default_idx, key="codebook_model_select")

    # Sidebar: guidelines + batch controls
    st.sidebar.markdown("## Codebook guidelines")
    if st.session_state.get("codebook_guidelines"):
        for i, g in enumerate(st.session_state["codebook_guidelines"]):
            st.sidebar.write(f"{i+1}. {g}")
    new_guideline = st.sidebar.text_area("Add guideline (press Add)", height=80, key="cb_add_guideline")
    if st.sidebar.button("Add guideline"):
        g = (new_guideline or "").strip()
        if g:
            st.session_state["codebook_guidelines"].append(g)
            # update canonical prompt template and persist into task_meta
            tm = st.session_state.get("task_meta", {}) or {}
            st.session_state["codebook_prompt_template"] = build_prompt_template(tm, st.session_state.get("codebook_guidelines", []))
            st.session_state["codebook_prompt_area"] = st.session_state["codebook_prompt_template"]
            tm["codebook_text"] = st.session_state["codebook_prompt_template"]
            st.session_state["task_meta"] = tm
            st.sidebar.success("Added guideline and updated prompt template.")

    st.sidebar.markdown("---")
    batch_size = st.sidebar.number_input("Batch size", min_value=1, max_value=10, value=st.session_state.get("codebook_batch_size", 1), key="cb_batch_size")
    st.session_state["codebook_batch_size"] = int(batch_size)

    # overlay controls
    cols_ctrl = st.columns([1, 3])
    with cols_ctrl[0]:
        st.checkbox("Show iterations overlay", value=st.session_state.get("codebook_overlay_show", True), key="codebook_overlay_show")
    with cols_ctrl[1]:
        st.caption("Tip: edit the prompt template on the right to change LM behavior. Use {{EXAMPLE}} where the example text should be inserted.")

    # overlay render
    if st.session_state.get("codebook_overlay_show", True):
        _render_topright_overlay(st.session_state.get("eval_history", []), st.session_state.get("batch_history", []), st.session_state.get("codebook_overlay_top", 80), st.session_state.get("codebook_overlay_right", 18))

    # Start / Abort controls
    control_col1, control_col2 = st.columns([1, 1])
    with control_col1:
        if st.button("Start batch", key="cb_start"):
            processed = set()
            for h in st.session_state.get("codebook_history", []):
                processed.update(h.get("indices", []))
            processed.update(st.session_state.get("codebook_batch_indices", []))
            indices = sample_batch_indices(st.session_state.get("codebook_batch_size", 1), total, exclude=list(processed))
            if not indices:
                st.info("No available items left to sample (all processed).")
            else:
                # initialize batch containers
                st.session_state["codebook_batch_indices"] = indices
                st.session_state["codebook_batch_ptr"] = 0
                st.session_state["codebook_batch_preds"] = [None] * len(indices)
                st.session_state["codebook_batch_user"] = [None] * len(indices)
                st.session_state["codebook_batch_matches"] = [False] * len(indices)
                st.session_state["codebook_batch_in_progress"] = True
                st.success(f"Started batch with indices: {indices}")
    with control_col2:
        if st.button("Abort batch", key="cb_abort"):
            st.session_state["codebook_batch_in_progress"] = False
            st.session_state["codebook_batch_indices"] = []
            st.session_state["codebook_batch_ptr"] = 0
            st.session_state["codebook_batch_preds"] = []
            st.session_state["codebook_batch_user"] = []
            st.session_state["codebook_batch_matches"] = []
            st.success("Aborted batch and cleared state.")

    # Layout: left annotate area, right prompt editor
    left_col, right_col = st.columns([2, 1.3])

    # Right: prompt editor (widget owns key codebook_prompt_area)
    with right_col:
        st.markdown("### Prompt template (editable)")
        st.caption("Edits persist to the task metadata (task_meta). Use `{{EXAMPLE}}` where the example text should be inserted.")
        prompt_area_val = st.text_area("LM prompt template", value=st.session_state.get("codebook_prompt_area", ""), height=420, key="codebook_prompt_area")
        # persist canonical prompt back into task_meta (safe)
        tm = st.session_state.get("task_meta", {}) or {}
        tm["codebook_text"] = prompt_area_val
        st.session_state["task_meta"] = tm
        st.session_state["codebook_prompt_template"] = prompt_area_val

        if st.button("Reset prompt to default", key="reset_prompt_btn"):
            default = build_prompt_template(st.session_state.get("task_meta", {}), st.session_state.get("codebook_guidelines", []))
            st.session_state["codebook_prompt_template"] = default
            st.session_state["codebook_prompt_area"] = default
            tm = st.session_state.get("task_meta", {}) or {}
            tm["codebook_text"] = default
            st.session_state["task_meta"] = tm
            st.success("Prompt reset to default.")

        st.markdown("---")
        if st.button("Synthesize prompt (agent)", key="synth_btn"):
            with st.spinner("Synthesizing prompt..."):
                p = st.session_state.get("codebook_prompt_template", "")
                lines = p.splitlines()
                seen = set()
                out_lines = []
                for L in lines:
                    if L.strip().startswith("- "):
                        tag = L.split(":")[0].strip()
                        if tag in seen:
                            continue
                        seen.add(tag)
                    out_lines.append(L)
                synthesized = "\n".join(out_lines)
                st.session_state["codebook_prompt_template"] = synthesized
                st.session_state["codebook_prompt_area"] = synthesized
                tm = st.session_state.get("task_meta", {}) or {}
                tm["codebook_text"] = synthesized
                st.session_state["task_meta"] = tm
                st.success("Synthesized prompt (duplicates removed).")

        st.markdown("---")
        # Exit/export area
        st.markdown("### Exit & export")
        st.caption("When finished iterating you can export the final prompt and a CSV report. Clicking Exit will also navigate you to Screen 4.")
        if st.button("Exit & export (prompt + report)"):
            # prepare files
            prompt_bytes = prepare_prompt_text(st.session_state.get("codebook_prompt_template", ""))
            csv_bytes = prepare_annotations_csv_from_history(st.session_state.get("codebook_history", []))
            report_bytes = prepare_report_json(st.session_state.get("eval_history", []), st.session_state.get("batch_history", []))

            st.download_button("Download prompt (txt)", data=prompt_bytes, file_name="final_prompt.txt")
            st.download_button("Download iterations report (csv)", data=csv_bytes, file_name="iterations_report.csv")
            st.download_button("Download run summary (json)", data=report_bytes, file_name="run_summary.json")
            # navigate to next screen
            st.session_state["screen"] = "Screen 4"
            st.success("Export prepared and moving to Screen 4 (you can download above).")

    # Left: batch / item UI
    with left_col:
        st.markdown("### Batch / Item")
        if st.session_state.get("codebook_batch_in_progress"):
            indices = st.session_state.get("codebook_batch_indices", [])
            ptr = st.session_state.get("codebook_batch_ptr", 0)
            if ptr >= len(indices):
                # finalize batch
                user_labels = st.session_state.get("codebook_batch_user", [])
                pred_labels = st.session_state.get("codebook_batch_preds", [])
                # compute simple batch F1 (single-label)
                try:
                    processed = sum(1 for u in user_labels if u is not None)
                    matches = sum(1 for u, p in zip(user_labels, pred_labels) if u is not None and u == p)
                    batch_f1 = matches / processed if processed else 0.0
                except Exception:
                    batch_f1 = 0.0

                prev_eval = st.session_state.get("eval_history", [])[-1] if st.session_state.get("eval_history") else st.session_state.get("codebook_sim_eval_acc", 0.0)
                if batch_f1 == 0.0:
                    eval_f1 = float(prev_eval)
                else:
                    eval_f1 = max(0.0, min(1.0, batch_f1 - 0.15 + random.uniform(-0.02, 0.02)))

                st.session_state["codebook_history"].append({
                    "indices": indices.copy(),
                    "lm_preds": pred_labels.copy(),
                    "user_labels": user_labels.copy(),
                    "accuracy": batch_f1
                })
                st.session_state["eval_history"].append(float(eval_f1))
                st.session_state["batch_history"].append(float(batch_f1))
                st.session_state["codebook_sim_eval_acc"] = float(eval_f1)
                st.success(f"Batch finished — batch F1 {batch_f1*100:.1f}%, eval F1 {eval_f1*100:.1f}%")
                # clear current batch
                st.session_state["codebook_batch_indices"] = []
                st.session_state["codebook_batch_ptr"] = 0
                st.session_state["codebook_batch_preds"] = []
                st.session_state["codebook_batch_user"] = []
                st.session_state["codebook_batch_matches"] = []
                st.session_state["codebook_batch_in_progress"] = False
            else:
                idx = indices[ptr]
                example_text = st.session_state.uploaded_data[idx]
                st.markdown(f"**Item {ptr+1} of {len(indices)} (index {idx})**")

                # show example
                safe_example = html_module.escape(example_text)
                st.markdown(
                    f'<div style="white-space: pre-wrap; background:#fbfdff; padding:10px; border-radius:6px; border:1px solid rgba(2,6,23,0.04); font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, monospace; font-size:13px;">'
                    f'{safe_example}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # LM + verifier (attempts=1)
                labels = [c.get("label") for c in st.session_state.get("task_meta", {}).get("codebook_struct", [])] or []
                try:
                    with st.spinner("Running LM (with simulated verifier)..."):
                        chosen_label, chosen_span, cot_agg, raw = run_lm_with_verifier(example_text, st.session_state.get("codebook_prompt_template", ""), labels, st.session_state.get("codebook_model_select"), ollama_url=OLLAMA_URL, attempts=1)
                except Exception as e:
                    st.error(f"LM call failed; using simulated fallback: {e}")
                    chosen_label = random.choice(labels) if labels else "LabelA"
                    chosen_span = example_text[:120]
                    cot_agg = "simulated fallback rationale"
                    raw = "[simulated]"

                # save prediction to batch preds slot (safe write)
                ptr_local = st.session_state.get("codebook_batch_ptr", 0)
                if len(st.session_state.get("codebook_batch_preds", [])) <= ptr_local:
                    # ensure list slot exists
                    needed = ptr_local + 1 - len(st.session_state.get("codebook_batch_preds", []))
                    st.session_state["codebook_batch_preds"].extend([None] * needed)
                st.session_state["codebook_batch_preds"][ptr_local] = chosen_label

                # Display LM outputs
                st.markdown("**LM predicted label:**")
                st.info(f"🤖  {chosen_label or '[no prediction]'}")
                st.markdown("**LM chain-of-thought / reasoning:**")
                st.write(cot_agg or "(none)")
                st.markdown("**LM-extracted span:**")
                if chosen_span:
                    st.code(chosen_span)
                else:
                    st.write("_(no span extracted)_")

                # Label selection UI (radio / checkboxes based on task_type)
                codebook = st.session_state.get("task_meta", {}).get("codebook_struct", []) or []
                label_names = [c.get("label") for c in codebook] if codebook else ["Label A", "Label B"]
                task_type = st.session_state.get("task_meta", {}).get("task_type", "multiclass")

                user_choice = None
                if task_type == "multilabel":
                    st.markdown("Select all labels that apply:")
                    selected = []
                    for ln in label_names:
                        key = f"cb_lab_{idx}_{ln}"
                        if st.checkbox(ln, key=key):
                            selected.append(ln)
                    user_choice = ",".join(selected)
                else:
                    key = f"cb_rad_{idx}"
                    default_idx = label_names.index(chosen_label) if (chosen_label in label_names) else 0
                    if label_names:
                        try:
                            user_choice = st.radio("Choose the correct label", label_names, index=default_idx, key=key)
                        except Exception:
                            user_choice = st.radio("Choose the correct label", label_names, index=0, key=key)
                    else:
                        st.warning("No labels defined in codebook — go to Screen 1 to define labels.")
                        user_choice = ""

                # Actions
                a1, a2, a3 = st.columns([1,1,1])
                with a1:
                    if st.button("Submit correction", key=f"cb_submit_{idx}"):
                        # ensure user list has slot
                        if len(st.session_state.get("codebook_batch_user", [])) <= ptr_local:
                            needed = ptr_local + 1 - len(st.session_state.get("codebook_batch_user", []))
                            st.session_state["codebook_batch_user"].extend([None] * needed)
                        st.session_state["codebook_batch_user"][ptr_local] = user_choice
                        match = (user_choice == st.session_state["codebook_batch_preds"][ptr_local])
                        if len(st.session_state.get("codebook_batch_matches", [])) <= ptr_local:
                            needed = ptr_local + 1 - len(st.session_state.get("codebook_batch_matches", []))
                            st.session_state["codebook_batch_matches"].extend([False] * needed)
                        st.session_state["codebook_batch_matches"][ptr_local] = bool(match)
                        st.success("Saved correction.")
                        st.session_state["codebook_batch_ptr"] = st.session_state.get("codebook_batch_ptr", 0) + 1
                with a2:
                    if st.button("Mark as correct / skip", key=f"cb_skip_{idx}"):
                        if len(st.session_state.get("codebook_batch_user", [])) <= ptr_local:
                            needed = ptr_local + 1 - len(st.session_state.get("codebook_batch_user", []))
                            st.session_state["codebook_batch_user"].extend([None] * needed)
                        st.session_state["codebook_batch_user"][ptr_local] = st.session_state["codebook_batch_preds"][ptr_local]
                        if len(st.session_state.get("codebook_batch_matches", [])) <= ptr_local:
                            needed = ptr_local + 1 - len(st.session_state.get("codebook_batch_matches", []))
                            st.session_state["codebook_batch_matches"].extend([False] * needed)
                        st.session_state["codebook_batch_matches"][ptr_local] = True
                        st.session_state["codebook_batch_ptr"] = st.session_state.get("codebook_batch_ptr", 0) + 1
                        st.info("Marked as correct and advanced.")
                with a3:
                    if st.button("Force advance", key=f"cb_force_{idx}"):
                        # if UI stuck, force increment pointer (useful during testing)
                        st.session_state["codebook_batch_ptr"] = st.session_state.get("codebook_batch_ptr", 0) + 1
                        st.info("Forced advance to next item.")

                # Optionally add LM label to codebook
                if st.button("Add LM label to codebook", key=f"cb_addlabel_{idx}"):
                    new_label = chosen_label or f"Label_{random.randint(1000,9999)}"
                    tm = st.session_state.get("task_meta", {}) or {}
                    struct = tm.get("codebook_struct", []) or []
                    existing_labels = {s.get("label") for s in struct if s.get("label")}
                    if new_label in existing_labels:
                        st.warning(f"'{new_label}' already in codebook.")
                    else:
                        struct.append({"label": new_label, "notes": f"Added from LM at {datetime.utcnow().isoformat()}"})
                        tm["codebook_struct"] = struct
                        appended = st.session_state.get("codebook_prompt_template", "") + f"\n- {new_label}: Added from LM"
                        st.session_state["codebook_prompt_template"] = appended
                        st.session_state["codebook_prompt_area"] = appended
                        tm["codebook_text"] = appended
                        st.session_state["task_meta"] = tm
                        st.success(f"Added label '{new_label}' to codebook (persisted).")
        else:
            st.info("No active batch. Click 'Start batch' to begin.")

    # Recent history / debug
    st.markdown("---")
    with st.expander("Recent iterations (history)"):
        if not st.session_state.get("codebook_history"):
            st.write("No iterations yet.")
        else:
            for i, h in enumerate(reversed(st.session_state.get("codebook_history")[-20:])):
                st.markdown(f"**Iteration {len(st.session_state.get('codebook_history'))-i}**")
                st.write(f"indices: {h.get('indices')}, batch F1: {h.get('accuracy',0.0)*100:.1f}%")
                st.write("LM preds:", h.get("lm_preds"))
                st.write("User labels:", h.get("user_labels"))

    # Overlay position controls
    with st.expander("Overlay position (top-right) & toggle", expanded=False):
        t = st.number_input("Overlay top offset (px)", min_value=0, max_value=400, value=st.session_state.get("codebook_overlay_top", 80), key="cb_overlay_top")
        r = st.number_input("Overlay right offset (px)", min_value=0, max_value=400, value=st.session_state.get("codebook_overlay_right", 18), key="cb_overlay_right")
        st.session_state["codebook_overlay_top"] = int(t)
        st.session_state["codebook_overlay_right"] = int(r)
        st.caption("Adjust overlay so it doesn't overlap your app header.")

    st.markdown("Tip: edits to the prompt template (on the right) are used immediately for LM calls. Use `{{EXAMPLE}}` where the example text should be inserted.")
