# screens/annotate_screen.py
"""
Screen 2 — Human Annotation (autosave) with fixed top-right chart, show/hide toggle,
prev/next below labels, and wrapped smaller row text.

Chart placement: fixed overlay at top-right (default top offset avoids header).
"""

from io import BytesIO
from datetime import datetime
from collections import Counter
import html as html_module
import base64

import streamlit as st
import matplotlib.pyplot as plt

from utils.data_utils import export_annotations_csv
from services.db import SessionLocal, Annotation


# -------------------------
# DB helpers (upsert)
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
# Annotation helpers
# -------------------------
def _remove_existing_annotation_for_row_in_memory(row_index: int):
    st.session_state.annotations = [
        a for a in st.session_state.annotations if int(a.get("row_index")) != int(row_index)
    ]


def _get_annotated_indices():
    annotated_idx = set()
    for a in st.session_state.annotations:
        try:
            annotated_idx.add(int(a.get("row_index")))
        except Exception:
            continue
    return annotated_idx


def _find_next_unannotated(total):
    annotated_idx = _get_annotated_indices()
    for i in range(total):
        if i not in annotated_idx:
            return i
    return None


# -------------------------
# Chart helpers (fig -> data URI)
# -------------------------
def _compute_label_counts(codebook_labels, annotations):
    counts = Counter()
    seen = set(codebook_labels or [])
    for a in annotations:
        lab = (a.get("label") or "").strip()
        if not lab:
            continue
        if "," in lab:
            parts = [p.strip() for p in lab.split(",") if p.strip()]
            for p in parts:
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
    fig, ax = plt.subplots(figsize=(2.8, 1.4))  # compact
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


def _render_fixed_topright_chart(codebook_labels, annotations, top_px=80, right_px=20):
    """
    Render a small fixed-position image in the top-right of the page.
    top_px/right_px: control offsets so it won't overlap the header.
    """
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
        width: 300px;
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
# Navigation callbacks
# -------------------------
def _go_prev_cb():
    cur = int(st.session_state.get("annot_idx", 0))
    st.session_state["annot_idx"] = max(0, cur - 1)


def _go_next_unannotated_cb():
    total = len(st.session_state.uploaded_data or [])
    nxt = _find_next_unannotated(total)
    if nxt is None:
        st.session_state["_next_msg"] = "All rows annotated."
    else:
        st.session_state["annot_idx"] = int(nxt)
        st.session_state.pop("_next_msg", None)


# -------------------------
# Main render function
# -------------------------
def render():
    st.header("Let's Annotate Some Data!")

    if not st.session_state.get("uploaded_data"):
        st.warning("No data loaded. Please upload in Screen 1.")
        return

    # session defaults
    st.session_state.setdefault("annotations", [])
    st.session_state.setdefault("last_saved_annotation_by_row", {})
    st.session_state.setdefault("last_saved_time_by_row", {})
    st.session_state.setdefault("annot_idx", 0)
    st.session_state.setdefault("show_dist_chart", True)

    total = len(st.session_state.uploaded_data)
    st.write(f"Rows loaded: **{total}**")

    # Top-right fixed chart: show/hide toggle is at the top of the page
    # Put the toggle next to the header (simple placement)
    show_chart = st.checkbox("Show class distribution", value=st.session_state.get("show_dist_chart", True), key="show_dist_chart")
    # default offsets; you can change the top_px if header is taller
    top_px = 80
    right_px = 20
    if show_chart:
        codebook_struct = st.session_state.task_meta.get("codebook_struct", [])
        codebook_labels = [e.get("label") for e in codebook_struct] if codebook_struct else []
        _render_fixed_topright_chart(codebook_labels, st.session_state.annotations, top_px=top_px, right_px=right_px)

    # display any message from callbacks
    if st.session_state.get("_next_msg"):
        st.info(st.session_state.pop("_next_msg"))

    # number input for index (safe: widget and state are consistent because we only set annot_idx via callbacks)
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

    if idx < 0:
        idx = 0
        st.session_state["annot_idx"] = 0
    if idx >= total:
        idx = total - 1
        st.session_state["annot_idx"] = idx

    sample = st.session_state.uploaded_data[idx]

    # Row text: smaller, wrapped
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
        ">{safe}</div>
        ''',
        unsafe_allow_html=True,
    )

    # Label UI
    codebook = st.session_state.task_meta.get("codebook_struct", [])
    label_names = [e.get("label") for e in codebook] if codebook else ["Label A", "Label B", "Label C"]

    if st.session_state.task_meta.get("task_type", "multiclass") == "multilabel":
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
        selected_labels = st.radio("Label", label_names, index=0, key=rad_key)

    # Prev / Next: placed immediately under label UI
    nav_left, nav_right = st.columns([1, 1])
    with nav_left:
        st.button("Prev", key=f"nav_prev_{idx}", on_click=_go_prev_cb)
    with nav_right:
        st.button("Next unannotated", key=f"nav_next_{idx}", on_click=_go_next_unannotated_cb)

    # Annotator & note fields
    annotator_key = f"annotator_{idx}"
    note_key = f"note_{idx}"
    annotator = st.text_input("Annotator", value=st.session_state.get("annotator_default", "annotator_1"), key=annotator_key)
    note = st.text_area("Note", height=80, key=note_key)

    # Show last saved timestamp
    last_saved_str = st.session_state.get("last_saved_time_by_row", {}).get(str(idx))
    if last_saved_str:
        try:
            t = datetime.fromisoformat(last_saved_str)
            st.caption(f"Last saved: {t.strftime('%Y-%m-%d %H:%M:%S')}")
        except Exception:
            st.caption(f"Last saved: {last_saved_str}")

    # Canonicalize labels string
    if isinstance(selected_labels, list):
        ann_label_value = ",".join(selected_labels) if selected_labels else ""
    else:
        ann_label_value = selected_labels or ""

    # Detect change vs last saved
    curr_key = str(idx)
    curr_tuple = (ann_label_value, annotator or "", note or "")
    last_tuple = st.session_state["last_saved_annotation_by_row"].get(curr_key)

    # Autosave when changed
    if last_tuple != curr_tuple:
        _remove_existing_annotation_for_row_in_memory(idx)
        ann = {
            "row_index": idx,
            "label": ann_label_value,
            "annotator": annotator or "",
            "note": note or "",
            "text": sample,
            "auto": True,
        }
        st.session_state.annotations.append(ann)
        try:
            _upsert_annotation_db(idx, sample, ann_label_value, annotator or "", note or "", auto=True)
            st.session_state["last_saved_annotation_by_row"][curr_key] = curr_tuple
            st.session_state["last_saved_time_by_row"][curr_key] = datetime.utcnow().isoformat()
            st.success("Autosaved.")
        except Exception as e:
            st.error(f"Autosave failed: {e}")

    # Export button
    if st.session_state.annotations:
        csv_bytes = export_annotations_csv(st.session_state.annotations)
        st.download_button("Download annotations CSV", data=csv_bytes, file_name="annotations.csv")

    # Recent annotations
    st.markdown("#### Recent annotations")
    if st.session_state.annotations:
        for a in reversed(st.session_state.annotations[-10:]):
            st.write(f"Row {a['row_index']} — {a['label']} — {a['annotator']}")
            st.write(a.get("note", ""))
    else:
        st.write("No annotations yet.")
