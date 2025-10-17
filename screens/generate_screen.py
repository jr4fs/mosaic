# screens/generate_screen.py
"""
Screen 4 — Generate Annotations from D_pool using final prompt (Ollama).

Reads:
 - st.session_state['d_pool_rows'] (list of texts)
 - st.session_state['d_pool_indices'] (original dataset indices)
 - st.session_state['task_meta']['codebook_text'] (LM prompt / codebook)

Writes:
 - st.session_state['generated_annotations'] (list of annotation dicts)
 - st.session_state['generated_annotations_map'] (map row_index -> annotation)
 - st.session_state['lm_generation_log'] (list of strings)
"""

from io import BytesIO
from datetime import datetime
import json
import random
import re
import base64
from typing import Dict, Any, List

import streamlit as st
import requests

from utils.data_utils import export_annotations_csv
# services.db may not be available in all environments; wrap usage in try/except
try:
    from services.db import SessionLocal, Annotation  # optional
except Exception:
    SessionLocal = None
    Annotation = None

# Defaults (do not access st.session_state at import time)
DEFAULT_MODEL = "llama3.2:latest"
DEFAULT_TIMEOUT = 15


# -------------------------
# DB upsert (optional save)
# -------------------------
def _upsert_annotation_db(row_index: int, text: str, label: str, annotator: str, note: str, auto: bool = True):
    if SessionLocal is None or Annotation is None:
        return
    try:
        db = SessionLocal()
    except Exception:
        return
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
        try:
            db.rollback()
        except Exception:
            pass
    finally:
        try:
            db.close()
        except Exception:
            pass


# -------------------------
# LM call & parsing helpers
# -------------------------
def call_ollama_generate(model: str, prompt: str, timeout: int = DEFAULT_TIMEOUT, ollama_url: str = None) -> str:
    """
    Call Ollama endpoint; raise on HTTP errors to be caught by caller.
    """
    if ollama_url is None:
        ollama_url = st.session_state.get("OLLAMA_URL") or "http://localhost:11434"
    url = f"{ollama_url.rstrip('/')}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    headers = {"Content-Type": "application/json"}
    r = requests.post(url, json=payload, headers=headers, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    # Ollama returns different shapes; attempt to extract human-friendly content
    out = j.get("response") or (j.get("message") or {}).get("content") or (j.get("choices") or [{}])[0].get("text") or str(j)
    return out


def parse_lm_annotation(resp: str) -> Dict[str, str]:
    """
    Robust parser: try JSON first, then heuristics.
    Returns dict with keys: label, rationale, raw
    """
    out = {"label": "", "rationale": "", "raw": resp or ""}
    if not resp:
        return out
    # try strict JSON
    try:
        j = json.loads(resp)
        if isinstance(j, dict):
            out["label"] = str(j.get("label") or j.get("prediction") or "").strip()
            out["rationale"] = str(j.get("rationale") or j.get("reason") or j.get("cot") or "").strip()
            return out
    except Exception:
        pass
    # try JSON substring
    m = re.search(r"\{(?:.|\s)*\}", resp)
    if m:
        try:
            j = json.loads(m.group(0))
            if isinstance(j, dict):
                out["label"] = str(j.get("label") or j.get("prediction") or "").strip()
                out["rationale"] = str(j.get("rationale") or j.get("reason") or j.get("cot") or "").strip()
                return out
        except Exception:
            pass
    # heuristic: look for "Label: X" patterns
    m_label = re.search(r"label\s*[:\-]\s*([^\n,]+)", resp, flags=re.IGNORECASE)
    if m_label:
        out["label"] = m_label.group(1).strip().strip('"').strip("'")
    else:
        # fallback: pick first reasonable token
        tokens = re.findall(r"[A-Za-z0-9_\- ]{1,40}", resp)
        tokens = [t.strip() for t in tokens if t.strip()]
        if tokens:
            for t in tokens:
                if any(c.isalpha() for c in t) and len(t) <= 40:
                    out["label"] = t
                    break
            if not out["label"]:
                out["label"] = tokens[0][:40]
    # rationale: capture 'because' clause or remainder
    m_reason = re.search(r"(?:because|reason|rationale|why)\s*[:\-]?\s*(.+)", resp, flags=re.IGNORECASE | re.DOTALL)
    if m_reason:
        out["rationale"] = m_reason.group(1).strip().split("\n\n")[0][:800]
    else:
        txt = resp.strip()
        if out["label"]:
            txt = txt.replace(out["label"], "", 1)
        out["rationale"] = txt[:800].strip()
    return out


# -------------------------
# Utility: build prompt for generation
# -------------------------
def assemble_prompt_for_generation(task_meta: dict) -> str:
    if not task_meta:
        task_meta = {}
    cb_text = task_meta.get("codebook_text")
    if cb_text:
        return cb_text
    # fallback assemble
    title = task_meta.get("task_name", "Annotation task")
    desc = task_meta.get("description", "")
    codebook_struct = task_meta.get("codebook_struct", []) or []
    lines = [f"Task: {title}", f"Description: {desc}", "", "Codebook:"]
    for e in codebook_struct:
        lines.append(f"- {e.get('label')}: {e.get('notes','')}")
    lines.append("")
    lines.append("Given the example under Example:, return a JSON with keys 'label' and 'rationale'.")
    return "\n".join(lines)


# -------------------------
# Ensure state keys used by this screen
# -------------------------
def _ensure_screen_state():
    st.session_state.setdefault("generated_annotations", st.session_state.get("generated_annotations", []))
    st.session_state.setdefault("generated_annotations_map", st.session_state.get("generated_annotations_map", {}))
    st.session_state.setdefault("lm_generation_log", st.session_state.get("lm_generation_log", []))
    st.session_state.setdefault("generate_model_choice", st.session_state.get("generate_model_choice", DEFAULT_MODEL))
    st.session_state.setdefault("generate_timeout", st.session_state.get("generate_timeout", DEFAULT_TIMEOUT))
    st.session_state.setdefault("generate_in_progress", False)


# ensure at top of render to avoid widget-state race
_ensure_screen_state()


# -------------------------
# Generation routine
# -------------------------
def generate_annotations_from_dpool(model: str, timeout: int = DEFAULT_TIMEOUT):
    d_pool_rows = st.session_state.get("d_pool_rows") or []
    d_pool_indices = st.session_state.get("d_pool_indices") or []
    if not d_pool_rows:
        st.warning("No D_pool rows found. Prepare a subsample on Screen 1 first.")
        return

    prompt_template = assemble_prompt_for_generation(st.session_state.get("task_meta", {}))
    total = len(d_pool_rows)
    prog = st.progress(0)
    logs: List[str] = st.session_state.get("lm_generation_log", [])[:]
    generated: List[Dict[str, Any]] = []

    st.session_state["generate_in_progress"] = True
    try:
        for i, (text, orig_idx) in enumerate(zip(d_pool_rows, d_pool_indices)):
            final_prompt = prompt_template + "\n\nExample:\n\"\"\"\n" + text + "\n\"\"\"\n\nReturn a JSON with keys 'label' and 'rationale'."
            try:
                with st.spinner(f"Generating {i+1}/{total}"):
                    resp = call_ollama_generate(model, final_prompt, timeout=timeout)
            except Exception as e:
                # fallback: use simple rule or random label to avoid hanging
                labels = [c.get("label") for c in (st.session_state.get("task_meta", {}) or {}).get("codebook_struct", [])] or []
                fallback_label = random.choice(labels) if labels else f"Label_{random.randint(1,999)}"
                resp = json.dumps({"label": fallback_label, "rationale": f"simulated fallback due to error: {str(e)}"})

            parsed = parse_lm_annotation(resp)
            ann = {
                "row_index": int(orig_idx) if orig_idx is not None else i,
                "text": text,
                "label": parsed.get("label", ""),
                "rationale": parsed.get("rationale", ""),
                "raw": parsed.get("raw", ""),
                "generated_at": datetime.utcnow().isoformat(),
                "model": model,
            }
            generated.append(ann)
            st.session_state["generated_annotations_map"][str(ann["row_index"])] = ann
            logs.append(f"[{i+1}/{total}] idx={ann['row_index']} label={ann['label']}")
            prog.progress(int((i + 1) / total * 100))
    finally:
        st.session_state["generated_annotations"] = generated
        st.session_state["lm_generation_log"] = logs
        st.session_state["generate_in_progress"] = False
        prog.empty()
    st.success(f"Generated {len(generated)} annotations for D_pool.")


# -------------------------
# Render screen
# -------------------------
def render():
    st.header("Screen 4 — Generate Annotations from D_pool")

    # initialize keys (safe)
    _ensure_screen_state()

    # minimal prerequisites
    if not st.session_state.get("uploaded_data"):
        st.warning("No data loaded. Please upload in Screen 1.")
        return

    st.write(f"Total rows uploaded: {len(st.session_state.get('uploaded_data', []))}")
    st.write(f"Human annotations (manual): {len(st.session_state.get('annotations', []))}")

    d_pool_n = len(st.session_state.get("d_pool_rows", []) or [])
    st.write(f"D_pool rows available: **{d_pool_n}**")

    st.markdown("---")
    st.markdown("### Generation settings")

    # Provide selectbox defaults from session_state; **do not** overwrite the same key after widget creation.
    model_choice = st.selectbox(
        "Model (local Ollama)",
        options=[st.session_state.get("generate_model_choice", DEFAULT_MODEL), DEFAULT_MODEL, "llama3:latest", "mistral:latest"],
        index=0,
        key="generate_model_choice",
    )

    timeout = st.number_input(
        "Per-call timeout (s)",
        min_value=3,
        max_value=120,
        value=int(st.session_state.get("generate_timeout", DEFAULT_TIMEOUT)),
        step=1,
        key="generate_timeout",
    )

    st.markdown("---")

    col_run, col_clear = st.columns([1, 1])
    with col_run:
        if st.button("Generate annotations for D_pool", key="run_generate"):
            # run generation (blocking)
            generate_annotations_from_dpool(model=model_choice, timeout=int(timeout))

    with col_clear:
        if st.button("Clear generated annotations (in memory)", key="clear_generated"):
            st.session_state["generated_annotations"] = []
            st.session_state["generated_annotations_map"] = {}
            st.session_state["lm_generation_log"] = []
            st.success("Cleared generated annotations and logs from memory.")

    # show recent logs
    if st.session_state.get("lm_generation_log"):
        st.markdown("#### Generation log (recent)")
        for ln in st.session_state["lm_generation_log"][-50:]:
            st.write(ln)

    st.markdown("---")
    st.markdown("### Generated Annotations (preview & export)")
    generated = st.session_state.get("generated_annotations", [])
    if generated:
        st.write(f"Generated {len(generated)} annotations.")
        # preview first N
        for g in generated[:50]:
            st.write(f"Row {g['row_index']} → {g.get('label')}")
            st.write(g.get("text"))
            if g.get("rationale"):
                st.caption(g.get("rationale"))
            st.markdown("---")

        # CSV export using helper (if it expects certain fields, adjust accordingly)
        try:
            csv_bytes = export_annotations_csv(generated)
            st.download_button("Download generated annotations (CSV)", data=csv_bytes, file_name="generated_annotations.csv")
        except Exception:
            # fallback TSV
            lines = ["row_index\tlabel\trationale\ttext\tgenerated_at\tmodel"]
            for g in generated:
                safe_text = g.get("text", "").replace("\n", " ").replace("\t", " ")
                safe_r = (g.get("rationale") or "").replace("\n", " ").replace("\t", " ")
                lines.append(f"{g.get('row_index')}\t{g.get('label')}\t{safe_r}\t{safe_text}\t{g.get('generated_at')}\t{g.get('model')}")
            st.download_button("Download generated annotations (TSV)", data="\n".join(lines).encode("utf-8"), file_name="generated_annotations.tsv")

        st.download_button("Download generated raw JSON", data=json.dumps(generated, indent=2).encode("utf-8"), file_name="generated_annotations.json")

        # download prompt
        prompt_text = (st.session_state.get("task_meta", {}) or {}).get("codebook_text", "")
        if prompt_text:
            st.download_button("Download final prompt (txt)", data=prompt_text.encode("utf-8"), file_name="final_prompt.txt")

        st.markdown("---")
        if st.button("Save generated annotations to DB (mark auto=True)"):
            cnt = 0
            for g in generated:
                try:
                    _upsert_annotation_db(g["row_index"], g["text"], g.get("label", ""), "lm_auto", g.get("rationale", ""), auto=True)
                    cnt += 1
                except Exception:
                    continue
            st.success(f"Saved {cnt} generated annotations to DB.")
    else:
        st.info("No generated annotations yet. Use 'Generate annotations for D_pool' to run the LM.")

    st.markdown("---")
    st.markdown("Tip: you can review generated annotations and export them. You can also save them into the DB for downstream usage.")
