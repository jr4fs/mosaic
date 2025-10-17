# screens/annotate_screen.py
"""
Annotate screen with:
 - progress bar/metric for # annotated
 - simplified "Suggest a code" (samples a few unlabelled rows)
 - robust parsing of model reply
 - Accept suggestion updates codebook_struct, codebook_text, UI and plot
 - Safe Prev/Next callbacks (no direct writes after widget instantiation)
"""

from io import BytesIO
from datetime import datetime
from collections import Counter
import json
import random
import re
import base64
import html as html_module
from typing import Dict

import streamlit as st
import matplotlib.pyplot as plt
import requests

from utils.data_utils import export_annotations_csv
from services.db import SessionLocal, Annotation

# Config
OLLAMA_URL = st.session_state.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_SUGGEST_MODEL = "llama3.2:latest"


# -------------------------
# Helpers: DB upsert
# -------------------------
def _upsert_annotation_db(row_index: int, text: str, label: str, annotator: str, note: str, auto: bool = False):
    db = SessionLocal()
    try:
        existing = db.query(Annotation).filter(Annotation.row_index == int(row_index)).first()
        if existing:
            existing.text = text
            existing.label = label
            existing.annotator = annotator
            existing.note = note
            existing.auto = auto
            db.add(existing)
        else:
            new = Annotation(
                row_index=int(row_index),
                text=text,
                label=label,
                annotator=annotator,
                note=note,
                auto=auto,
            )
            db.add(new)
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# -------------------------
# Suggestion parser (robust)
# -------------------------
def parse_suggestion(resp: str) -> Dict[str, str]:
    out = {"label": "", "description": "", "raw": resp or ""}
    if not resp or not resp.strip():
        return out
    # try strict JSON
    try:
        j = json.loads(resp)
        if isinstance(j, dict):
            out["label"] = str(j.get("label", "") or "").strip()
            out["description"] = str(j.get("description", "") or "").strip()
            return out
    except Exception:
        pass
    # JSON substring
    m_json = re.search(r"\{(?:.|\s)*\}", resp)
    if m_json:
        try:
            j = json.loads(m_json.group(0))
            if isinstance(j, dict):
                out["label"] = str(j.get("label", "") or "").strip()
                out["description"] = str(j.get("description", "") or "").strip()
                return out
        except Exception:
            pass
    s = resp.strip()
    # Label: ... Description: ...
    m_label = re.search(
        r"Label\s*[:\-]?\s*(.+?)(?=(?:\r?\n\s*(?:Description|Desc|Guideline)\s*[:\-])|(?:\s+Description\s*[:\-])|$)",
        s,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m_label:
        label_candidate = m_label.group(1).strip()
        label_candidate = re.sub(r"(Description|Desc|Guideline).*$", "", label_candidate, flags=re.IGNORECASE | re.DOTALL).strip()
        out["label"] = label_candidate
    m_desc = re.search(r"(?:Description|Desc|Guideline|Guidelines)\s*[:\-]?\s*(.+)", s, flags=re.IGNORECASE | re.DOTALL)
    if m_desc:
        desc_candidate = m_desc.group(1).strip()
        desc_candidate = re.split(r"\n\s*\n", desc_candidate)[0].strip()
        out["description"] = re.sub(r"\s+", " ", desc_candidate)
    # inline fallback
    if not out["label"]:
        m_inline = re.search(r"Label\s*[-:]\s*([A-Za-z0-9_\- ]{1,80})", s, flags=re.IGNORECASE)
        if m_inline:
            out["label"] = m_inline.group(1).strip()
    if not out["label"]:
        m_any = re.search(r"Label\b[^\w]{0,3}([A-Za-z0-9_\- ]{1,40})", s, flags=re.IGNORECASE)
        if m_any:
            cand = m_any.group(1).strip()
            cand = re.split(r"\bDescription\b", cand, flags=re.IGNORECASE)[0].strip()
            out["label"] = cand[:60]
    return out


# -------------------------
# Ollama call (non-stream)
# -------------------------
def call_ollama_generate(model: str, prompt: str, timeout: int = 60, ollama_url: str = OLLAMA_URL) -> str:
    url = f"{ollama_url.rstrip('/')}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    headers = {"Content-Type": "application/json"}
    r = requests.post(url, json=payload, headers=headers, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    out = j.get("response") or (j.get("message") or {}).get("content") or (j.get("choices") or [{}])[0].get("text") or str(j)
    return out


# -------------------------
# Chart helpers
# -------------------------
def _compute_label_counts(codebook_labels, annotations):
    counts = Counter()
    seen = set(codebook_labels or [])
    for a in annotations:
        lab = (a.get("label") or "").strip()
        if not lab:
            continue
        if "," in lab:
            for p in [p.strip() for p in lab.split(",") if p.strip()]:
                counts[p] += 1
                seen.add(p)
        else:
            counts[lab] += 1
            seen.add(lab)
    labels_ordered = []
    if codebook_labels:
        labels_ordered.extend(codebook_labels)
    for extra in sorted(seen):
        if extra not in labels_ordered:
            labels_ordered.append(extra)
    counts_list = [counts.get(l, 0) for l in labels_ordered]
    return labels_ordered, counts_list


def _get_distribution_fig(codebook_labels, annotations):
    labels, counts = _compute_label_counts(codebook_labels, annotations)
    fig, ax = plt.subplots(figsize=(3.2, 1.6))
    if labels:
        ax.bar(range(len(labels)), counts)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.set_title("Annotated class distribution", fontsize=9)
        plt.tight_layout()
    else:
        ax.text(0.5, 0.5, "No labels", ha="center", va="center", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
    return fig


def _fig_to_data_uri(fig, fmt="png"):
    buf = BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches="tight")
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode("ascii")
    buf.close()
    return f"data:image/{fmt};base64,{data}"


def _render_fixed_topright_chart(codebook_labels, annotations, top_px=80, right_px=16):
    try:
        fig = _get_distribution_fig(codebook_labels, annotations)
        data_uri = _fig_to_data_uri(fig)
        plt.close(fig)
    except Exception:
        return
    html = f"""
    <div style="
        position: fixed;
        top: {top_px}px;
        right: {right_px}px;
        width: 320px;
        max-width: 36vw;
        background: rgba(255,255,255,0.98);
        border-radius: 8px;
        padding: 8px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.08);
        z-index: 9999;">
      <img src="{data_uri}" style="width:100%; height:auto; display:block; border-radius:6px;" />
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# -------------------------
# Navigation helpers (safe)
# -------------------------
def _get_annotated_indices(annotations):
    annotated_idx = set()
    for a in annotations:
        try:
            if a.get("label") and str(a.get("label")).strip():
                annotated_idx.add(int(a.get("row_index")))
        except Exception:
            continue
    return annotated_idx


def _find_next_unannotated(total, annotations):
    annotated_idx = _get_annotated_indices(annotations)
    for i in range(total):
        if i not in annotated_idx:
            return i
    return None


def _go_prev_cb():
    cur = int(st.session_state.get("annot_idx", 0))
    st.session_state["annot_idx"] = max(0, cur - 1)


def _go_next_unannotated_cb():
    total = len(st.session_state.uploaded_data or [])
    nxt = _find_next_unannotated(total, st.session_state.annotations)
    if nxt is None:
        st.session_state["_next_msg"] = "All rows annotated."
    else:
        st.session_state["annot_idx"] = int(nxt)
        st.session_state.pop("_next_msg", None)


# -------------------------
# Main render
# -------------------------
def render():
    st.header("Screen 2 — Human Annotation (autosave)")

    if not st.session_state.get("uploaded_data"):
        st.warning("No data loaded. Please upload in Screen 1.")
        return

    # session defaults
    st.session_state.setdefault("annotations", [])
    st.session_state.setdefault("last_saved_annotation_by_row", {})
    st.session_state.setdefault("last_saved_time_by_row", {})
    st.session_state.setdefault("annot_idx", 0)
    st.session_state.setdefault("show_dist_chart", True)
    st.session_state.setdefault("codebook_edit_text", None)
    st.session_state.setdefault("suggest_model_choice", DEFAULT_SUGGEST_MODEL)

    total = len(st.session_state.uploaded_data)

    # compute progress
    annotated_idx = _get_annotated_indices(st.session_state.annotations)
    annotated_count = len(annotated_idx)
    pct = int(round((annotated_count / total) * 100)) if total > 0 else 0

    # top row: metric + progress bar
    col_m1, col_m2, col_m3 = st.columns([1, 2, 1])
    with col_m1:
        st.metric("Annotated", f"{annotated_count}/{total}")
    with col_m2:
        st.progress(pct)
    with col_m3:
        st.metric("Progress", f"{pct}%")

    # show/hide chart
    codebook_struct = st.session_state.get("task_meta", {}).get("codebook_struct", []) or []
    codebook_labels = [e.get("label") for e in codebook_struct] if codebook_struct else []
    if st.checkbox("Show class distribution", value=st.session_state.get("show_dist_chart", True), key="show_dist_chart"):
        _render_fixed_topright_chart(codebook_labels, st.session_state.annotations, top_px=110, right_px=16)

    # layout: left = annotate, right = codebook
    left_col, right_col = st.columns([2, 1.25])
    with left_col:
        # info
        if st.session_state.get("_next_msg"):
            st.info(st.session_state.pop("_next_msg"))

        # annot index widget
        idx = st.number_input(
            "Row index to annotate",
            min_value=0,
            max_value=max(0, total - 1),
            value=int(st.session_state.get("annot_idx", 0)),
            step=1,
            key="annot_idx",
        )
        try:
            idx = int(st.session_state.get("annot_idx", idx))
        except Exception:
            idx = 0
            st.session_state["annot_idx"] = 0

        sample = st.session_state.uploaded_data[idx]
        st.markdown("**Row:**")
        safe = html_module.escape(sample)
        st.markdown(
            f'''
            <div style="
                white-space: pre-wrap;
                background: #fbfdff;
                padding: 10px 12px;
                border-radius: 6px;
                border: 1px solid rgba(2,6,23,0.04);
                font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, 'Courier New', monospace;
                font-size: 13px;
                line-height: 1.35;
                max-height: 340px;
                overflow: auto;
            ">{safe}</div>
            ''',
            unsafe_allow_html=True,
        )

        # labels UI
        task_type = st.session_state.get("task_meta", {}).get("task_type", "multiclass")
        label_names = [e.get("label") for e in codebook_struct] if codebook_struct else ["Label A", "Label B", "Label C"]
        st.markdown("**Label**")
        if task_type == "multilabel":
            st.markdown("Select all labels that apply:")
            selected = []
            for ln in label_names:
                chk_key = f"label_chk_{idx}_{ln}"
                if st.checkbox(ln, key=chk_key):
                    selected.append(ln)
            selected_labels = selected
        else:
            rad_key = f"label_rad_{idx}"
            if not label_names:
                label_names = ["Label A", "Label B", "Label C"]
            selected_labels = st.radio("Choose label", label_names, index=0, key=rad_key)

        # Prev/Next (safe callbacks)
        nav_left, nav_right = st.columns([1, 1])
        with nav_left:
            st.button("Prev", key=f"nav_prev_{idx}", on_click=_go_prev_cb)
        with nav_right:
            st.button("Next unannotated", key=f"nav_next_{idx}", on_click=_go_next_unannotated_cb)

        # annotator / note
        annotator_key = f"annotator_{idx}"
        note_key = f"note_{idx}"
        annotator = st.text_input("Annotator", value=st.session_state.get("annotator_default", "annotator_1"), key=annotator_key)
        note = st.text_area("Note (optional)", height=100, key=note_key)

        # last saved
        last_saved_str = st.session_state.get("last_saved_time_by_row", {}).get(str(idx))
        if last_saved_str:
            try:
                t = datetime.fromisoformat(last_saved_str)
                st.caption(f"Last saved: {t.strftime('%Y-%m-%d %H:%M:%S')}")
            except Exception:
                st.caption(f"Last saved: {last_saved_str}")

        # canonicalize label value
        if isinstance(selected_labels, list):
            ann_label_value = ",".join(selected_labels) if selected_labels else ""
        else:
            ann_label_value = selected_labels or ""

        curr_key = str(idx)
        curr_tuple = (ann_label_value, annotator or "", note or "")
        last_tuple = st.session_state["last_saved_annotation_by_row"].get(curr_key)

        if last_tuple != curr_tuple:
            # remove prior in-memory annotation for this row
            st.session_state.annotations = [a for a in st.session_state.annotations if int(a.get("row_index")) != int(idx)]
            ann = {
                "row_index": idx,
                "label": ann_label_value,
                "annotator": annotator or "",
                "note": note or "",
                "text": sample,
                "auto": False,
            }
            st.session_state.annotations.append(ann)
            try:
                _upsert_annotation_db(idx, sample, ann_label_value, annotator or "", note or "", auto=False)
                st.session_state["last_saved_annotation_by_row"][curr_key] = curr_tuple
                st.session_state["last_saved_time_by_row"][curr_key] = datetime.utcnow().isoformat()
                st.success("Autosaved.")
            except Exception as e:
                st.error(f"Autosave failed: {e}")

        # download
        if st.session_state.annotations:
            csv_bytes = export_annotations_csv(st.session_state.annotations)
            st.download_button("Download annotations CSV", data=csv_bytes, file_name="annotations.csv")

        st.markdown("#### Recent annotations")
        if st.session_state.annotations:
            for a in reversed(st.session_state.annotations[-10:]):
                st.write(f"Row {a['row_index']} — {a['label']} — {a['annotator']}")
                st.write(a.get("note", ""))
        else:
            st.write("No annotations yet.")

    # RIGHT: codebook editor + suggest
    with right_col:
        st.markdown("### Codebook & Prompt")
        task_meta = st.session_state.get("task_meta", {}) or {}
        # default assembled prompt if none
        def assemble_codebook_prompt(tm: Dict) -> str:
            title = tm.get("task_name", "Annotation task")
            desc = tm.get("description", "")
            codebook_struct_local = tm.get("codebook_struct", []) or []
            lines = [f"Task: {title}", f"Description: {desc}", "", "Codebook:"]
            for e in codebook_struct_local:
                lines.append(f"- {e.get('label')}: {e.get('notes','')}")
            lines.append("")
            lines.append("Given an example, output the best label(s) and a short rationale.")
            return "\n".join(lines)

        if st.session_state.get("codebook_edit_text") is None:
            st.session_state["codebook_edit_text"] = task_meta.get("codebook_text", assemble_codebook_prompt(task_meta))
        codebook_text = st.text_area("Editable LM prompt / codebook (edits persist)", value=st.session_state["codebook_edit_text"], height=420, key="codebook_edit_text")
        # persist edits
        tm = st.session_state.get("task_meta", {}) or {}
        tm["codebook_text"] = codebook_text
        st.session_state["task_meta"] = tm

        st.markdown("---")
        st.markdown("**Suggest a new label**")
        suggest_model = st.selectbox("LLM model (local Ollama)", [DEFAULT_SUGGEST_MODEL, "llama3:latest", "mistral:latest"], index=0, key="suggest_model_choice")

        if st.button("🪄🤖 Suggest a code (small sample of unlabelled rows)", key="suggest_code_btn"):
            # pick small sample of unlabelled rows (no label or empty label)
            all_texts = st.session_state.get("uploaded_data", [])
            ann_by_idx = {int(a["row_index"]): a for a in st.session_state.annotations if a.get("row_index") is not None}
            candidates = []
            for i, txt in enumerate(all_texts):
                ann = ann_by_idx.get(i)
                if ann is None or not (ann.get("label") or "").strip():
                    candidates.append((i, txt))
            sampled = [t for _, t in (candidates[:5] if len(candidates) > 5 else candidates)]
            cb_text = st.session_state.get("task_meta", {}).get("codebook_text") or assemble_codebook_prompt(task_meta)
            examples_block = "\n\n".join([f"Example {i+1}:\n{html_module.escape(t)}" for i, t in enumerate(sampled)]) if sampled else "No unlabelled examples available."
            prompt = (
                f"You are helping maintain a codebook for an annotation task.\n\n"
                f"Current codebook and prompt:\n{cb_text}\n\n"
                f"Unlabelled / example texts (small sample):\n{examples_block}\n\n"
                "If there is a missing label that would help annotate these examples, "
                "propose ONE new concise label name (no more than 3 words) and a single-line description/guideline. "
                "Respond ONLY with a JSON object like: {\"label\": \"NAME\", \"description\": \"one-line\"}. "
                "If no new label is needed, respond with {\"label\":\"\",\"description\":\"\"}."
            )
            try:
                with st.spinner("Contacting Ollama for suggestion..."):
                    resp = call_ollama_generate(suggest_model, prompt)
                st.markdown("**LM suggestion (raw):**")
                st.write(resp)
                suggestion = parse_suggestion(resp)
                if suggestion.get("label") or suggestion.get("description"):
                    st.markdown("**Parsed suggestion:**")
                    st.write(suggestion)
                    col_a, col_b = st.columns([1, 1])
                    with col_a:
                        if st.button("Accept suggestion and add as new label", key=f"accept_sugg_{random.randint(0,999999)}"):
                            # append to codebook_struct in session_state
                            tm = st.session_state.get("task_meta", {}) or {}
                            struct = tm.get("codebook_struct", []) or []
                            # guard duplicate labels
                            existing_labels = {s.get("label") for s in struct}
                            new_label = suggestion.get("label").strip()
                            new_desc = suggestion.get("description").strip()
                            if not new_label:
                                new_label = (suggestion.get("raw") or "SuggestedLabel")[:40]
                            if new_label in existing_labels:
                                st.warning(f"Label '{new_label}' already exists in codebook.")
                            else:
                                struct.append({"label": new_label, "notes": new_desc})
                                tm["codebook_struct"] = struct
                                # update codebook text/prompt
                                prev_prompt = tm.get("codebook_text", assemble_codebook_prompt(tm))
                                appended = prev_prompt + f"\n- {new_label}: {new_desc}"
                                tm["codebook_text"] = appended
                                # persist
                                st.session_state["task_meta"] = tm
                                st.session_state["codebook_edit_text"] = appended
                                st.success(f"Added new label: {new_label}")
                                # update chart labels immediately by updating local vars and letting rerun reflect changes
                                codebook_struct.clear()
                                codebook_struct.extend(tm.get("codebook_struct", []))
                                # update progress display (no need to explicitly call rerun; button click will rerun the script)
                    with col_b:
                        if st.button("Reject suggestion", key=f"reject_sugg_{random.randint(0,999999)}"):
                            st.info("Suggestion rejected.")
                else:
                    st.info("Model did not suggest a new label.")
            except requests.exceptions.RequestException as e:
                st.error(f"Ollama request failed: {e}")
            except Exception as e:
                st.error(f"Suggestion step failed: {e}")

        st.markdown("---")
        if st.button("Save codebook edits to task_meta", key="save_codebook_edits"):
            tm = st.session_state.get("task_meta", {}) or {}
            tm["codebook_text"] = st.session_state.get("codebook_edit_text", "")
            st.session_state["task_meta"] = tm
            st.success("Saved codebook text into task_meta (available to Screen 3).")

    # footer
    st.markdown("---")
    st.markdown("Tip: 'Suggest a code' proposes one label from a small unlabelled sample. Accepting it updates the in-memory codebook, the prompt, and the class-distribution chart.")

# end file
