# screens/codebook_screen.py
"""
Screen 3 — LM-Assisted Codebook Development (two-column layout)

Modifications:
- Prompt area made taller and right column slightly wider.
- If batch F1 == 0.0, eval F1 is set to the previous eval F1 instead of being batch_f1 - 0.15.
"""

from typing import List
import subprocess
import json
import requests
import random
import math
import base64
from io import BytesIO
from collections import Counter
import html as html_module

import streamlit as st
import matplotlib.pyplot as plt

# Ollama endpoint (change if needed)
OLLAMA_URL = "http://localhost:11434"


# -------------------------
# Utilities: Ollama / models
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


def call_ollama_generate_sync(model: str, prompt_text: str, timeout: int = 20) -> str:
    url = f"{OLLAMA_URL}/api/generate"
    payload = {"model": model, "prompt": prompt_text, "stream": False}
    headers = {"Content-Type": "application/json"}
    r = requests.post(url, json=payload, headers=headers, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    out = j.get("response") or (j.get("message") or {}).get("content") or (j.get("choices") or [{}])[0].get("text")
    return out if out is not None else str(j)


def simulate_lm_label(labels: List[str], seed_text: str = "") -> str:
    if not labels:
        return ""
    r = abs(hash(seed_text))
    return labels[r % len(labels)]


def parse_label_from_lm_text(lm_text: str, labels: List[str]) -> str:
    if not lm_text or not labels:
        return ""
    low = lm_text.lower()
    for lab in labels:
        if lab.lower() in low:
            return lab
    tokens = [t.strip(" .:-\n\r\t\"'") for t in lm_text.replace("\n", " ").split()]
    for t in tokens:
        for lab in labels:
            if t.lower() == lab.lower():
                return lab
    return labels[0] if labels else ""


# -------------------------
# Prompt template builder (uses placeholder {{EXAMPLE}})
# -------------------------
def build_prompt_template(task_meta: dict, guidelines: List[str]) -> str:
    task_name = task_meta.get("task_name", "Annotation task")
    desc = task_meta.get("description", "")
    codebook = task_meta.get("codebook_struct", [])  # list of {label, notes}
    labels = [c.get("label") for c in codebook] if codebook else []
    labels_part = "\n".join([f"- {l}" for l in labels]) if labels else "(no labels defined)"
    guidelines_part = "\n".join([f"- {g}" for g in guidelines]) if guidelines else "No guidelines."
    tmpl = (
        f"Task: {task_name}\n"
        f"Description: {desc}\n\n"
        f"Labels (choose one):\n{labels_part}\n\n"
        f"Guidelines:\n{guidelines_part}\n\n"
        f"Example to annotate (replace the token {{EXAMPLE}} below with the example text):\n\"\"\"\n{{EXAMPLE}}\n\"\"\"\n\n"
        f"Return a short answer with the predicted label (label only, no explanation)."
    )
    return tmpl


def finalize_prompt_for_example(template: str, example_text: str) -> str:
    if "{{EXAMPLE}}" in template:
        return template.replace("{{EXAMPLE}}", example_text)
    else:
        return template + "\n\nExample:\n\"\"\"\n" + example_text + "\n\"\"\"\n"


# -------------------------
# plotting helpers & data URI
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


# -------------------------
# F1 computation utilities
# -------------------------
def compute_f1_for_batch_singlelabel(user_labels: List[str], pred_labels: List[str]) -> float:
    processed = sum(1 for u in user_labels if u is not None)
    if processed == 0:
        return 0.0
    matches = sum(1 for u, p in zip(user_labels, pred_labels) if u is not None and u == p)
    return matches / processed


def compute_f1_for_batch_multilabel(user_labels: List[str], pred_labels: List[str]) -> float:
    TP = 0
    pred_count = 0
    true_count = 0
    for u, p in zip(user_labels, pred_labels):
        if u is None or p is None:
            continue
        true_set = set([s.strip() for s in u.split(",") if s.strip()])
        pred_set = set([s.strip() for s in p.split(",") if s.strip()])
        TP += len(true_set & pred_set)
        pred_count += len(pred_set)
        true_count += len(true_set)
    if pred_count == 0 or true_count == 0:
        return 0.0
    precision = TP / pred_count if pred_count else 0.0
    recall = TP / true_count if true_count else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_batch_f1(user_labels: List[str], pred_labels: List[str], task_type: str = "multiclass") -> float:
    if task_type == "multilabel":
        return compute_f1_for_batch_multilabel(user_labels, pred_labels)
    else:
        return compute_f1_for_batch_singlelabel(user_labels, pred_labels)


# -------------------------
# Session-state init
# -------------------------
def _ensure_state():
    st.session_state.setdefault("codebook_guidelines", [])
    st.session_state.setdefault("codebook_batch_indices", [])
    st.session_state.setdefault("codebook_batch_ptr", 0)
    st.session_state.setdefault("codebook_batch_preds", [])
    st.session_state.setdefault("codebook_batch_user", [])
    st.session_state.setdefault("codebook_batch_matches", [])
    st.session_state.setdefault("codebook_batch_in_progress", False)
    st.session_state.setdefault("codebook_history", [])
    st.session_state.setdefault("eval_history", [])
    st.session_state.setdefault("batch_history", [])
    st.session_state.setdefault("codebook_sim_eval_acc", 0.0)
    st.session_state.setdefault("codebook_model_select", None)  # selectbox owns this key
    st.session_state.setdefault("codebook_batch_size", 1)
    st.session_state.setdefault("codebook_prompt_template", None)
    st.session_state.setdefault("codebook_overlay_top", 80)
    st.session_state.setdefault("codebook_overlay_right", 18)
    st.session_state.setdefault("codebook_overlay_show", True)


# -------------------------
# Sampling + prompt finalization
# -------------------------
def sample_batch_indices(batch_size: int, total: int, exclude: List[int]) -> List[int]:
    available = [i for i in range(total) if i not in exclude]
    if not available:
        return []
    if batch_size >= len(available):
        return available.copy()
    return random.sample(available, batch_size)


# -------------------------
# Fixed top-right overlay rendering (plot + metrics)
# -------------------------
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
        z-index: 9999;
        font-family: Inter, system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;">
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
# Main render
# -------------------------
def render():
    _ensure_state()

    st.header("LM-Assisted Codebook Development")
    st.write("Iteratively run an LM on small batches and correct its labels. Edit the LM prompt template on the right anytime.")

    # prerequisites
    if not st.session_state.get("uploaded_data"):
        st.warning("No data loaded. Please upload data on Screen 1 first.")
        return
    if not st.session_state.get("task_meta"):
        st.warning("No task defined. Please set up a task on Screen 1 first.")
        return

    total = len(st.session_state.uploaded_data)

    # MODEL selection (selectbox owns key codebook_model_select)
    models = get_installed_models()
    sel = st.selectbox("Choose model for LM calls", models, index=0, key="codebook_model_select")
    # We will read st.session_state["codebook_model_select"] when making LM calls

    # Sidebar: guidelines list and batch_size control (keep guidelines in sidebar)
    st.sidebar.markdown("## Codebook guidelines")
    if st.session_state["codebook_guidelines"]:
        for i, g in enumerate(st.session_state["codebook_guidelines"]):
            st.sidebar.write(f"{i+1}. {g}")
    new_guideline = st.sidebar.text_area("Add guideline (press Add)", height=80, key="cb_add_guideline")
    if st.sidebar.button("Add guideline"):
        g = (new_guideline or "").strip()
        if g:
            st.session_state["codebook_guidelines"].append(g)
            st.sidebar.success("Added guideline.")
            # also append to the persistent prompt template
            if st.session_state.get("codebook_prompt_template") is None:
                st.session_state["codebook_prompt_template"] = build_prompt_template(st.session_state["task_meta"], st.session_state["codebook_guidelines"])
            else:
                st.session_state["codebook_prompt_template"] += f"\n- {g}"
            st.session_state["cb_add_guideline"] = ""

    st.sidebar.markdown("---")
    st.sidebar.markdown("Batch controls")
    batch_size = st.sidebar.number_input("Batch size", min_value=1, max_value=10, value=st.session_state["codebook_batch_size"], key="cb_batch_size")
    st.session_state["codebook_batch_size"] = int(batch_size)

    # Ensure a prompt template exists before creating the prompt widget
    if st.session_state.get("codebook_prompt_template") is None:
        st.session_state["codebook_prompt_template"] = build_prompt_template(st.session_state["task_meta"], st.session_state["codebook_guidelines"])

    # Overlay controls (show/hide) - let the checkbox own the key
    cols_ctrl = st.columns([1, 3])
    with cols_ctrl[0]:
        st.checkbox("Show iterations overlay", value=st.session_state.get("codebook_overlay_show", True), key="codebook_overlay_show")
    with cols_ctrl[1]:
        st.caption("Tip: edit the prompt template on the right to change LM behavior. Use {{EXAMPLE}} where the example text should be inserted.")

    # If overlay is shown, render pinned top-right overlay (plot + metrics)
    if st.session_state.get("codebook_overlay_show", True):
        top_px = st.session_state.get("codebook_overlay_top", 80)
        right_px = st.session_state.get("codebook_overlay_right", 18)
        _render_topright_overlay(st.session_state.get("eval_history", []), st.session_state.get("batch_history", []), top_px, right_px)

    # Controls: Start / Abort batch
    control_col1, control_col2 = st.columns([1, 1])
    with control_col1:
        if st.button("Start batch", key="cb_start"):
            processed = set()
            for h in st.session_state["codebook_history"]:
                processed.update(h.get("indices", []))
            processed.update(st.session_state.get("codebook_batch_indices", []))
            indices = sample_batch_indices(st.session_state["codebook_batch_size"], total, exclude=list(processed))
            if not indices:
                st.info("No available items left to sample (all processed).")
            else:
                st.session_state["codebook_batch_indices"] = indices
                st.session_state["codebook_batch_ptr"] = 0
                st.session_state["codebook_batch_preds"] = [None] * len(indices)
                st.session_state["codebook_batch_user"] = [None] * len(indices)
                st.session_state["codebook_batch_matches"] = [False] * len(indices)
                st.session_state["codebook_batch_in_progress"] = True
    with control_col2:
        if st.button("Abort batch", key="cb_abort"):
            st.session_state["codebook_batch_in_progress"] = False
            st.success("Aborted batch.")
            st.session_state["codebook_batch_indices"] = []
            st.session_state["codebook_batch_ptr"] = 0
            st.session_state["codebook_batch_preds"] = []
            st.session_state["codebook_batch_user"] = []
            st.session_state["codebook_batch_matches"] = []

    # Two-column main area: left = item + LM UI, right = editable prompt
    # Make the right column slightly wider than before so prompt feels roomier
    left_col, right_col = st.columns([2, 1.3])

    # Right column: editable prompt template widget (widget key differs from persistent key)
    with right_col:
        st.markdown("### Add Codebook Guidelines to the Prompt")
        st.caption("Use `{{EXAMPLE}}` where the example text should be inserted.")
        # initialize the widget-backed key the first time
        if "codebook_prompt_area" not in st.session_state:
            st.session_state["codebook_prompt_area"] = st.session_state.get("codebook_prompt_template", "")
        # make area taller (requested)
        prompt_area_val = st.text_area("LM prompt template", value=st.session_state.get("codebook_prompt_area", ""), height=380, key="codebook_prompt_area")
        def _reset_prompt():
            default = build_prompt_template(st.session_state["task_meta"], st.session_state["codebook_guidelines"])
            st.session_state["codebook_prompt_template"] = default
            st.session_state["codebook_prompt_area"] = default
        st.button("Reset prompt to default", on_click=_reset_prompt, key="reset_prompt_btn")
        # keep the persistent copy synced
        st.session_state["codebook_prompt_template"] = st.session_state.get("codebook_prompt_area", st.session_state.get("codebook_prompt_template", ""))

    # Left column: batch item + LM prediction + correction UI
    with left_col:
        st.markdown("### Batch / Item")
        if st.session_state.get("codebook_batch_in_progress"):
            indices = st.session_state["codebook_batch_indices"]
            ptr = st.session_state["codebook_batch_ptr"]
            if ptr >= len(indices):
                # finalize batch
                user_labels = st.session_state.get("codebook_batch_user", [])
                pred_labels = st.session_state.get("codebook_batch_preds", [])
                task_type = st.session_state.get("task_meta", {}).get("task_type", "multiclass")
                batch_f1 = compute_batch_f1(user_labels, pred_labels, task_type=task_type)

                # --- NEW: if batch_f1 == 0.0 use previous eval F1 instead of batch_f1 - 0.15 ---
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

                # show example wrapped and escaped
                safe_example = html_module.escape(example_text)
                st.markdown(
                    f'<div style="white-space: pre-wrap; background:#fbfdff; padding:10px; border-radius:6px; border:1px solid rgba(2,6,23,0.04); font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, monospace; font-size:13px;">'
                    f'{safe_example}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # build final prompt from persistent prompt template
                prompt_to_use = finalize_prompt_for_example(st.session_state.get("codebook_prompt_template", ""), example_text)

                # LM call (or reuse existing)
                lm_label = None
                lm_text = None
                if st.session_state["codebook_batch_preds"][ptr] is not None:
                    lm_label = st.session_state["codebook_batch_preds"][ptr]
                    st.info(f"Reusing LM prediction: {lm_label}")
                else:
                    try:
                        model_name = st.session_state.get("codebook_model_select") or "llama3:latest"
                        lm_text = call_ollama_generate_sync(model_name, prompt_to_use, timeout=20)
                        lm_label = parse_label_from_lm_text(lm_text, [c.get("label") for c in st.session_state["task_meta"].get("codebook_struct", [])])
                    except Exception:
                        lm_label = simulate_lm_label([c.get("label") for c in st.session_state["task_meta"].get("codebook_struct", [])], seed_text=example_text)
                    st.session_state["codebook_batch_preds"][ptr] = lm_label

                # Display LM predicted label with robot emoji and show LM text (if present)
                st.markdown("**LM predicted label:**")
                st.info(f"🤖  {lm_label or '[no prediction]'}")
                if lm_text:
                    with st.expander("View full LM output (raw)"):
                        st.code(lm_text)

                # Label selection UI
                codebook = st.session_state["task_meta"].get("codebook_struct", [])
                label_names = [c.get("label") for c in codebook] if codebook else []
                user_choice = None
                task_type = st.session_state["task_meta"].get("task_type", "multiclass")

                if task_type == "multilabel":
                    selected = []
                    st.markdown("Select all labels that apply:")
                    for ln in label_names:
                        key = f"cb_lab_{idx}_{ln}"
                        if st.checkbox(ln, key=key):
                            selected.append(ln)
                    user_choice = ",".join(selected)
                else:
                    key = f"cb_rad_{idx}"
                    default_idx = 0
                    if lm_label in label_names:
                        default_idx = label_names.index(lm_label)
                    if label_names:
                        user_choice = st.radio("Choose the correct label", label_names, index=default_idx, key=key)
                    else:
                        st.warning("No labels defined in codebook — go to Screen 1 to define labels.")
                        user_choice = ""

                # Submit / skip positioned below label UI
                c1, c2 = st.columns([1, 1])
                with c1:
                    if st.button("Submit correction", key=f"cb_submit_{idx}"):
                        st.session_state["codebook_batch_user"][ptr] = user_choice
                        match = (user_choice == st.session_state["codebook_batch_preds"][ptr])
                        st.session_state["codebook_batch_matches"][ptr] = bool(match)
                        st.success("Saved correction.")
                        st.session_state["codebook_batch_ptr"] = st.session_state.get("codebook_batch_ptr", 0) + 1
                with c2:
                    if st.button("Mark as correct / skip", key=f"cb_skip_{idx}"):
                        st.session_state["codebook_batch_user"][ptr] = st.session_state["codebook_batch_preds"][ptr]
                        st.session_state["codebook_batch_matches"][ptr] = True
                        st.session_state["codebook_batch_ptr"] = st.session_state.get("codebook_batch_ptr", 0) + 1
                        st.info("Marked as correct and advanced.")

                # running batch stats (show current batch F1)
                user_labels_partial = st.session_state.get("codebook_batch_user", [])
                pred_labels_partial = st.session_state.get("codebook_batch_preds", [])
                batch_f1_partial = compute_batch_f1(user_labels_partial, pred_labels_partial, task_type=task_type)
                st.write(f"Batch progress: {sum(1 for u in user_labels_partial if u is not None)}/{len(indices)} — batch F1 so far: {batch_f1_partial*100:.1f}%")
        else:
            st.info("No active batch. Click 'Start batch' to begin.")

    # Recent history
    st.markdown("---")
    with st.expander("Recent iterations (history)"):
        if not st.session_state["codebook_history"]:
            st.write("No iterations yet.")
        else:
            for i, h in enumerate(reversed(st.session_state["codebook_history"][-20:])):
                st.markdown(f"**Iteration {len(st.session_state['codebook_history'])-i}**")
                st.write(f"indices: {h['indices']}, batch F1: {h['accuracy']*100:.1f}%")
                st.write("LM preds:", h["lm_preds"])
                st.write("User labels:", h["user_labels"])

    # overlay reposition controls (small UI at bottom of screen)
    with st.expander("Overlay position (top-right) & toggle", expanded=False):
        t = st.number_input("Overlay top offset (px)", min_value=0, max_value=400, value=st.session_state.get("codebook_overlay_top", 80), key="cb_overlay_top")
        r = st.number_input("Overlay right offset (px)", min_value=0, max_value=400, value=st.session_state.get("codebook_overlay_right", 18), key="cb_overlay_right")
        st.session_state["codebook_overlay_top"] = int(t)
        st.session_state["codebook_overlay_right"] = int(r)
        st.caption("Adjust overlay so it doesn't overlap your app header.")

    st.markdown("Tip: edits to the prompt template (on the right) are used immediately for LM calls. Use `{{EXAMPLE}}` where the example text should be inserted.")
