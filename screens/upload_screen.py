# screens/upload_screen.py
"""
Upload screen with FAISS-based subsampling by label-keywords, fair per-label allocation,
examples per label, and subsample splitting into D_val (10%) and D_pool (90%).

Stores everything into st.session_state so other screens can read:
- working_subset_df / working_subset_rows / subsample_indices
- d_val_rows / d_val_df / d_val_indices
- d_pool_rows / d_pool_df / d_pool_indices
- codebook_pool_rows
- uploaded_data is set to d_val_rows when the user Accepts the subsample (so Screen 2 annotates D_val)
"""

from typing import List, Tuple
from datetime import datetime
from io import BytesIO
import random
import html as html_module

import streamlit as st
import pandas as pd
import numpy as np

# Optional heavy deps
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import faiss
except Exception:
    faiss = None


# -------------------------
# Utility helpers
# -------------------------
def read_uploaded_file(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()
    uploaded.seek(0)
    if name.endswith(".csv"):
        return pd.read_csv(uploaded)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded)
    if name.endswith((".txt", ".md")):
        raw = uploaded.read().decode("utf-8", errors="replace")
        rows = [r for r in raw.splitlines() if r.strip()]
        return pd.DataFrame({"text": rows})
    # fallback attempts
    try:
        uploaded.seek(0)
        return pd.read_csv(uploaded)
    except Exception:
        uploaded.seek(0)
        raw = uploaded.read().decode("utf-8", errors="replace")
        rows = [r for r in raw.splitlines() if r.strip()]
        return pd.DataFrame({"text": rows})


def init_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    if SentenceTransformer is None:
        raise ImportError("sentence-transformers is not installed in this environment.")
    # cache in session_state
    if st.session_state.get("embed_model_name") != model_name or "embed_model" not in st.session_state:
        st.session_state["embed_model"] = SentenceTransformer(model_name)
        st.session_state["embed_model_name"] = model_name
    return st.session_state["embed_model"]


def compute_embeddings(texts: List[str], model, batch_size: int = 128) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    all_emb = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        emb = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        all_emb.append(emb)
    return np.vstack(all_emb) if all_emb else np.empty((0, 0), dtype=np.float32)


def build_faiss_index(emb: np.ndarray):
    # normalize to unit vectors and build IndexFlatIP (inner-product => cosine after normalization)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    emb_norm = emb / norms
    dim = emb_norm.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb_norm.astype(np.float32))
    return index, emb_norm.astype(np.float32)


def search_faiss(index, emb_norm, query_vec: np.ndarray, topk: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    q = query_vec.reshape(1, -1).astype(np.float32)
    qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    D, I = index.search(qn, topk)
    return I[0], D[0]


def safe_union_list_preserve_order(lists: List[List[int]]) -> List[int]:
    seen = set()
    out = []
    for lst in lists:
        for v in lst:
            if v not in seen:
                seen.add(v)
                out.append(v)
    return out


def round_robin_fill(per_label_cands: List[List[int]], target_n: int) -> List[int]:
    pointers = [0] * len(per_label_cands)
    out = []
    seen = set()
    while len(out) < target_n:
        added = False
        for li, cands in enumerate(per_label_cands):
            if pointers[li] < len(cands):
                idx = cands[pointers[li]]
                pointers[li] += 1
                if idx not in seen:
                    seen.add(idx)
                    out.append(idx)
                    added = True
                    if len(out) >= target_n:
                        break
        if not added:
            break
    return out


# -------------------------
# Main render
# -------------------------
def render():
    st.header("Screen 1 — Upload Data & Prepare Subsample")

    # session defaults (keys used across app)
    st.session_state.setdefault("uploaded_df", None)
    st.session_state.setdefault("uploaded_data", [])
    st.session_state.setdefault("working_subset_df", None)
    st.session_state.setdefault("working_subset_rows", [])
    st.session_state.setdefault("subsample_indices", [])
    st.session_state.setdefault("embed_model_name", "all-MiniLM-L6-v2")
    st.session_state.setdefault("embed_matrix", None)
    st.session_state.setdefault("faiss_index_present", False)
    st.session_state.setdefault("task_meta", st.session_state.get("task_meta", {}))

    # constants for accept UI (defined here so persistent rendering can reference them)
    _ACCEPTED_KEY = "subsample_accepted"
    _ACCEPTED_AT = "subsample_accepted_at"
    _BACKUP_KEY = "uploaded_data_backup"

    # Layout: upload / options columns
    col_up, col_opts = st.columns([2, 1])
    with col_up:
        uploaded = st.file_uploader("Upload dataset (.csv, .xlsx, .txt/.md) or paste below", type=["csv", "xlsx", "txt", "md"], key="upload_file")
        st.markdown("— OR —")
        pasted = st.text_area("Paste text (each line becomes a row)", height=120, key="paste_area")
    with col_opts:
        st.markdown("Sampling options")
        subsample_size = st.number_input("Subsample size (target)", min_value=2, max_value=100000, value=50, step=1, key="subsample_size")
        max_index_rows = st.number_input("Max rows to index (cap for speed)", min_value=50, max_value=200000, value=5000, step=50, key="max_index_rows")
        embed_model_name = st.text_input("Embedding model (sentence-transformers)", value=st.session_state.get("embed_model_name", "all-MiniLM-L6-v2"), key="embed_model_name_input")

    # Load dataframe
    df = None
    if uploaded is not None:
        try:
            df = read_uploaded_file(uploaded)
        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")
            return
    elif pasted and pasted.strip():
        rows = [r for r in pasted.splitlines() if r.strip()]
        df = pd.DataFrame({"text": rows})

    if df is None:
        st.info("Upload a file or paste text to start.")
        return

    # show preview and select columns
    st.session_state["uploaded_df"] = df
    st.write(f"Uploaded dataset — {len(df)} rows. Preview:")
    st.dataframe(df.head(5))

    cols = list(df.columns)
    st.markdown("**Select column(s) that contain the text to annotate.**")
    preselected = st.session_state.get("task_meta", {}).get("selected_columns", [])
    selected_cols = st.multiselect("Columns to use", cols, default=preselected, key="upload_selected_cols")

    # show visual pills for selection (red pill like earlier)
    if selected_cols:
        pill_html = "".join(
            f'<span style="display:inline-block;background:#fee2e2;color:#b91c1c;padding:6px 12px;margin-right:8px;border-radius:999px;font-weight:700">'
            f'{html_module.escape(str(c))}</span>'
            for c in selected_cols
        )
        st.markdown(f'<div style="margin-top:8px"><strong>Selected column(s)</strong><div style="margin-top:6px">{pill_html}</div></div>', unsafe_allow_html=True)

    # Confirm columns -> create rows
    if st.button("Confirm columns and load for annotation", key="confirm_columns"):
        if not selected_cols:
            st.error("Select at least one column.")
        else:
            working = df[selected_cols].astype(str).fillna("").apply(lambda row: " ".join([c for c in row.values if str(c).strip()]), axis=1)
            rows = working.tolist()
            # Try to call project clean_rows helper if present
            try:
                from utils.data_utils import clean_rows
                rows = clean_rows(rows)
            except Exception:
                rows = [r for r in rows if isinstance(r, str) and r.strip()]
            st.session_state["uploaded_data"] = rows
            tm = st.session_state.get("task_meta", {})
            tm["selected_columns"] = selected_cols
            st.session_state["task_meta"] = tm
            st.session_state["uploaded_df"] = df.copy()
            st.success(f"Loaded {len(rows)} rows into session for sampling/annotation.")

    st.markdown("---")
    st.markdown("### Task definition")
    task_name = st.text_input("Task name", value=st.session_state.get("task_meta", {}).get("task_name", ""), key="task_name")
    task_desc = st.text_area("Short description", value=st.session_state.get("task_meta", {}).get("description", ""), height=80, key="task_desc")
    task_type = st.selectbox("Task type", ["multiclass", "multilabel"], index=0, key="task_type")

    # labels input
    n_labels = st.number_input("Number of labels", min_value=1, max_value=20, value=3, key="n_labels")
    existing_codebook = st.session_state.get("task_meta", {}).get("codebook_struct", []) or []
    codebook_defaults = [dict(x) for x in existing_codebook]
    while len(codebook_defaults) < int(n_labels):
        codebook_defaults.append({"label": f"Label_{len(codebook_defaults)+1}", "notes": ""})
    if len(codebook_defaults) > int(n_labels):
        codebook_defaults = codebook_defaults[: int(n_labels)]

    st.markdown("Provide label names and keywords/guidelines (used for FAISS queries).")
    labels = []
    for i in range(int(n_labels)):
        st.markdown(f"**Label {i+1}**")
        c1, c2 = st.columns([1, 3])
        default_item = codebook_defaults[i] if i < len(codebook_defaults) else {"label": f"Label_{i+1}", "notes": ""}
        with c1:
            label_name = st.text_input(f"Label name #{i+1}", value=default_item.get("label", f"Label_{i+1}"), key=f"label_name_{i}")
        with c2:
            keywords = st.text_area(f"Keywords/guidelines for {label_name or f'Label_{i+1}'}", value=default_item.get("notes", ""), height=80, key=f"label_kw_{i}")
        labels.append({"label": label_name or f"Label_{i+1}", "keywords": keywords or ""})

    st.markdown("---")

    # Prepare subsample button: builds embeddings/faiss, queries per-label, fair sampling, preview, accept UI
    if st.button("Save task definition and prepare subsample", key="prepare_subsample"):
        # persist codebook and task meta
        codebook_struct = [{"label": l["label"], "notes": l["keywords"]} for l in labels]
        st.session_state["task_meta"] = {
            "task_name": task_name,
            "description": task_desc,
            "task_type": task_type,
            "codebook_struct": codebook_struct,
            "selected_columns": st.session_state.get("task_meta", {}).get("selected_columns", selected_cols),
        }
        st.success("Saved task metadata into session_state.")

        # prepare base text list
        df_full = st.session_state.get("uploaded_df")
        sel_cols = st.session_state.get("task_meta", {}).get("selected_columns", selected_cols)
        if df_full is None or not sel_cols:
            st.error("No uploaded dataframe or selected columns. Please confirm columns before subsampling.")
            return

        texts = df_full[sel_cols].astype(str).fillna("").apply(lambda row: " ".join([c for c in row.values if str(c).strip()]), axis=1).tolist()
        st.session_state["uploaded_data"] = texts  # store full list

        # build an index pool (cap to max_index_rows for speed)
        idx_pool = list(range(len(texts)))
        if len(idx_pool) > int(max_index_rows):
            st.warning(f"Dataset large ({len(idx_pool)} rows). Indexing random cap of {max_index_rows} rows for speed.")
            idx_pool = random.sample(idx_pool, int(max_index_rows))
        texts_to_index = [texts[i] for i in idx_pool]

        # try to load embedding model
        model = None
        try:
            model = init_embedding_model(embed_model_name)
        except Exception as e:
            st.warning(f"Could not load embedding model '{embed_model_name}': {e}")
            st.info("Falling back to keyword-based sampling / random supplementation.")

        # compute embeddings + faiss index if available
        emb = None
        index = None
        emb_norm = None
        if model is not None and faiss is not None:
            try:
                emb = compute_embeddings(texts_to_index, model, batch_size=256)
                index, emb_norm = build_faiss_index(emb)
            except Exception as e:
                st.warning(f"Embeddings/FAISS build failed: {e}")
                index = None
                emb_norm = None

        # per-label candidate collection with strict sim_threshold
        per_label_candidates: List[List[int]] = []
        sim_threshold = 0.5  # requested strict threshold
        base_topk = min(max(20, int(subsample_size * 2)), len(idx_pool))

        for lb in codebook_struct:
            kw = (lb.get("notes") or "").strip()
            if not kw:
                kw = lb.get("label", "")
            if not kw:
                per_label_candidates.append([])
                continue

            candidates = []
            # FAISS path if available
            if index is not None and emb_norm is not None and model is not None:
                try:
                    q_emb = model.encode([kw], convert_to_numpy=True)
                    q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)
                    ids, scores = search_faiss(index, emb_norm, q_emb[0], topk=base_topk)
                    for i_val, s_val in zip(ids, scores):
                        i_int = int(i_val)
                        if 0 <= i_int < len(idx_pool) and s_val >= sim_threshold:
                            candidates.append(idx_pool[i_int])
                    # dedupe preserve order
                    seen = set()
                    candidates = [c for c in candidates if not (c in seen or seen.add(c))]
                except Exception:
                    candidates = []

            # keyword substring fallback
            if not candidates:
                kw_tokens = [t.strip().lower() for t in kw.split(",") if t.strip()]
                matches = []
                for i, t in enumerate(texts):
                    score = 0
                    lt = t.lower()
                    for tok in kw_tokens:
                        if not tok:
                            continue
                        if tok in lt:
                            score += 1
                    if score > 0:
                        matches.append((i, score))
                matches.sort(key=lambda x: -x[1])
                candidates = [m[0] for m in matches[: min(len(matches), int(subsample_size * 2))]]

            per_label_candidates.append(candidates)

        # Show candidate counts per label
        st.markdown("### Candidate pool sizes per label (after thresholding)")
        for lb, cands in zip(codebook_struct, per_label_candidates):
            st.write(f"- **{lb.get('label')}**: {len(set(cands))} candidates")

        # Attempt fair sampling: round-robin across label candidate pools
        chosen_indices = round_robin_fill(per_label_candidates, int(subsample_size))

        # If insufficient candidates, try to expand from neighbours (if faiss) then supplement with randoms
        if len(chosen_indices) < int(subsample_size):
            extra = []
            if index is not None and emb_norm is not None and model is not None:
                for lb in codebook_struct:
                    kw = (lb.get("notes") or "").strip() or lb.get("label", "")
                    if not kw:
                        continue
                    try:
                        q_emb = model.encode([kw], convert_to_numpy=True)
                        q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)
                        ids2, scores2 = search_faiss(index, emb_norm, q_emb[0], topk=min(len(emb_norm), int(subsample_size * 4)))
                        for i_val in ids2:
                            i_int = int(i_val)
                            if 0 <= i_int < len(idx_pool):
                                extra.append(idx_pool[i_int])
                    except Exception:
                        continue
            all_candidates = safe_union_list_preserve_order([chosen_indices, extra])
            if len(all_candidates) < int(subsample_size):
                remaining = [i for i in range(len(texts)) if i not in all_candidates]
                random.shuffle(remaining)
                all_candidates.extend(remaining[: max(0, int(subsample_size) - len(all_candidates))])
            chosen_indices = all_candidates[: int(subsample_size)]

        # final unique truncated list
        final_indices = []
        seen = set()
        for i in chosen_indices:
            if i not in seen:
                seen.add(i)
                final_indices.append(i)
            if len(final_indices) >= int(subsample_size):
                break

        # persist raw subsample
        st.session_state["subsample_indices"] = final_indices
        st.session_state["working_subset_rows"] = [texts[i] for i in final_indices]
        st.session_state["working_subset_df"] = df_full.iloc[final_indices].copy()
        st.session_state["embed_matrix"] = emb
        st.session_state["faiss_index_present"] = index is not None
        st.session_state["faiss_index_obj"] = index
        st.session_state["faiss_idx_pool_map"] = idx_pool

        st.success(f"Prepared subsample of {len(final_indices)} rows (target {subsample_size}).")

        # Show examples per label (from the subsample)
        st.markdown("### Examples selected per label (up to 3 each)")
        for lb_idx, lb in enumerate(codebook_struct):
            st.markdown(f"**{lb.get('label')}**")
            cands_set = set(per_label_candidates[lb_idx])
            examples = [i for i in final_indices if i in cands_set][:3]
            if not examples:
                st.write("_No high-confidence examples found for this label in the subsample._")
            else:
                for ex in examples:
                    st.write(f"- row {ex}:")
                    st.code(texts[ex])

        # show preview (persisted)
        preview_df = st.session_state["working_subset_df"].copy()
        st.markdown("#### Subsample preview (first 20 rows)")
        st.dataframe(preview_df.head(20))

        # ---------- Split into D_val (10%) and D_pool (90%) ----------
        sub_n = len(final_indices)
        d_val_n = max(1, int(round(0.10 * sub_n)))  # at least 1
        # reproducible seed stored in session_state
        if "subsample_split_seed" not in st.session_state:
            st.session_state["subsample_split_seed"] = random.randint(0, 2**31 - 1)
        rnd = random.Random(st.session_state["subsample_split_seed"])
        shuffled = list(final_indices)
        rnd.shuffle(shuffled)
        d_val_indices = sorted(shuffled[:d_val_n])
        d_pool_indices = [i for i in final_indices if i not in d_val_indices]

        d_val_rows = [texts[i] for i in d_val_indices]
        d_pool_rows = [texts[i] for i in d_pool_indices]
        d_val_df = df_full.iloc[d_val_indices].copy() if d_val_indices else pd.DataFrame(columns=df_full.columns)
        d_pool_df = df_full.iloc[d_pool_indices].copy() if d_pool_indices else pd.DataFrame(columns=df_full.columns)

        st.session_state["d_val_indices"] = d_val_indices
        st.session_state["d_pool_indices"] = d_pool_indices
        st.session_state["d_val_rows"] = d_val_rows
        st.session_state["d_pool_rows"] = d_pool_rows
        st.session_state["d_val_df"] = d_val_df
        st.session_state["d_pool_df"] = d_pool_df
        st.session_state["subsample_prepared_at"] = datetime.utcnow().isoformat()

        st.success(f"Split subsample: D_val = {len(d_val_indices)} rows, D_pool = {len(d_pool_indices)} rows.")
        st.markdown("**D_val (annotation set) examples:**")
        for row_idx in d_val_indices[:5]:
            st.write(f"- row {row_idx}")
            st.code(texts[row_idx])
        st.markdown("**D_pool (pool for LM / iterated sampling) examples:**")
        for row_idx in d_pool_indices[:5]:
            st.write(f"- row {row_idx}")
            st.code(texts[row_idx])

        # ---------- Accept UI (persistent) ----------
        # Accept button (no experimental_rerun)
        if not st.session_state.get(_ACCEPTED_KEY, False):
            if st.button("Accept subsample as working dataset (use for subsequent screens)", key="accept_subsample_btn"):
                # backup previous uploaded_data
                st.session_state[_BACKUP_KEY] = list(st.session_state.get("uploaded_data", []))
                # apply D_val as the dataset used by Screen 2
                st.session_state["uploaded_data"] = list(st.session_state.get("d_val_rows", []))
                # persist pool for codebook screen
                st.session_state["codebook_pool_rows"] = list(st.session_state.get("d_pool_rows", []))
                st.session_state[_ACCEPTED_KEY] = True
                st.session_state[_ACCEPTED_AT] = datetime.utcnow().isoformat()
                st.success(f"Subsample accepted — D_val ({len(st.session_state.get('d_val_rows', []))}) set as current annotation dataset.")
                # Set screen pointer (no forced rerun — session_state change will be visible on interaction)
                st.session_state["screen"] = "Screen 2"
                st.session_state["annot_idx"] = 0
                st.session_state["annot_idx_widget"] = 0

        # If already accepted, show badge + undo
        if st.session_state.get(_ACCEPTED_KEY, False):
            ts = st.session_state.get(_ACCEPTED_AT)
            try:
                ts_readable = datetime.fromisoformat(ts).strftime("%Y-%m-%d %H:%M:%S UTC")
            except Exception:
                ts_readable = str(ts)
            st.markdown(
                f'''
                <div style="display:flex;align-items:center;gap:12px;margin-top:8px">
                  <div style="background:linear-gradient(90deg,#10b981,#059669);color:white;padding:8px 14px;border-radius:10px;font-weight:700">
                    ✅ Subsample accepted
                  </div>
                  <div style="color:#6b7280;font-size:13px">Accepted at {ts_readable}</div>
                </div>
                ''',
                unsafe_allow_html=True,
            )
            if st.button("Undo acceptance (restore previous dataset)", key="undo_accept_btn"):
                prev = st.session_state.get(_BACKUP_KEY)
                if prev is not None:
                    st.session_state["uploaded_data"] = prev
                    st.success("Restored previous uploaded dataset.")
                else:
                    st.warning("No backup — attempting to restore from uploaded_df.")
                    if st.session_state.get("uploaded_df") is not None:
                        try:
                            df_full = st.session_state["uploaded_df"]
                            cols_restore = st.session_state.get("task_meta", {}).get("selected_columns", [])
                            if not cols_restore:
                                cols_restore = [df_full.columns[0]] if len(df_full.columns) > 0 else []
                            restored = df_full[cols_restore].astype(str).fillna("").apply(lambda row: " ".join([c for c in row.values if str(c).strip()]), axis=1).tolist()
                            st.session_state["uploaded_data"] = restored
                            st.success("Restored from uploaded_df.")
                        except Exception:
                            st.error("Failed to restore from uploaded_df.")
                # clear accept flags and split keys
                st.session_state.pop(_ACCEPTED_KEY, None)
                st.session_state.pop(_ACCEPTED_AT, None)
                st.session_state.pop(_BACKUP_KEY, None)
                for k in ["d_val_indices", "d_pool_indices", "d_val_rows", "d_pool_rows", "d_val_df", "d_pool_df", "codebook_pool_rows"]:
                    st.session_state.pop(k, None)

    # -----------------------
    # Persistent preview & Accept UI (renders whenever a subsample exists)
    # -----------------------
    if st.session_state.get("working_subset_df") is not None:
        st.markdown("---")
        st.markdown("### Subsample (persisted)")
        preview_df = st.session_state["working_subset_df"]
        st.write(f"Subsample rows: **{len(preview_df)}** (prepared at {st.session_state.get('subsample_prepared_at')})")
        st.dataframe(preview_df.head(20))

        # Show a few preserved examples for D_val / D_pool if available
        if st.session_state.get("d_val_rows"):
            st.markdown("**D_val (annotation set) example(s)**")
            for r in st.session_state["d_val_rows"][:5]:
                st.code(r)
        if st.session_state.get("d_pool_rows"):
            st.markdown("**D_pool (LM pool) example(s)**")
            for r in st.session_state["d_pool_rows"][:5]:
                st.code(r)

        # Show Accepted badge if already accepted
        if st.session_state.get(_ACCEPTED_KEY, False):
            ts = st.session_state.get(_ACCEPTED_AT)
            try:
                ts_readable = datetime.fromisoformat(ts).strftime("%Y-%m-%d %H:%M:%S UTC")
            except Exception:
                ts_readable = str(ts)
            st.markdown(
                f'''
                <div style="display:flex;align-items:center;gap:12px;margin-top:8px">
                <div style="background:linear-gradient(90deg,#10b981,#059669);color:white;padding:8px 14px;border-radius:10px;font-weight:700">
                    ✅ Subsample accepted
                </div>
                <div style="color:#6b7280;font-size:13px">Accepted at {ts_readable}</div>
                </div>
                ''',
                unsafe_allow_html=True,
            )
            if st.button("Undo acceptance (restore previous dataset)", key="undo_accept_btn_2"):
                prev = st.session_state.get(_BACKUP_KEY)
                if prev is not None:
                    st.session_state["uploaded_data"] = prev
                    st.success("Restored previous uploaded dataset.")
                else:
                    st.warning("No backup — attempting to restore from uploaded_df.")
                    if st.session_state.get("uploaded_df") is not None:
                        try:
                            df_full = st.session_state["uploaded_df"]
                            cols_restore = st.session_state.get("task_meta", {}).get("selected_columns", [])
                            if not cols_restore:
                                cols_restore = [df_full.columns[0]] if len(df_full.columns) > 0 else []
                            restored = df_full[cols_restore].astype(str).fillna("").apply(lambda row: " ".join([c for c in row.values if str(c).strip()]), axis=1).tolist()
                            st.session_state["uploaded_data"] = restored
                            st.success("Restored from uploaded_df.")
                        except Exception:
                            st.error("Failed to restore from uploaded_df.")
                # clear accept flags and split keys
                st.session_state.pop(_ACCEPTED_KEY, None)
                st.session_state.pop(_ACCEPTED_AT, None)
                st.session_state.pop(_BACKUP_KEY, None)
                for k in ["d_val_indices", "d_pool_indices", "d_val_rows", "d_pool_rows", "d_val_df", "d_pool_df", "codebook_pool_rows"]:
                    st.session_state.pop(k, None)
        else:
            # If not accepted yet, show the accept button here too (persistent)
            if st.button("Accept subsample as working dataset (use for subsequent screens)", key="accept_subsample_btn_persist"):
                st.session_state[_BACKUP_KEY] = list(st.session_state.get("uploaded_data", []))
                st.session_state["uploaded_data"] = list(st.session_state.get("d_val_rows", []))
                st.session_state["codebook_pool_rows"] = list(st.session_state.get("d_pool_rows", []))
                st.session_state[_ACCEPTED_KEY] = True
                st.session_state[_ACCEPTED_AT] = datetime.utcnow().isoformat()
                st.success(f"Subsample accepted — D_val ({len(st.session_state.get('d_val_rows', []))}) set as current annotation dataset.")
                st.session_state["screen"] = "Screen 2"
                st.session_state["annot_idx"] = 0
                st.session_state["annot_idx_widget"] = 0

    # footer hint
    st.markdown("---")
    st.markdown("Tip: After accepting a subsample, Screen 2 will annotate D_val (stored in `st.session_state['d_val_rows']`) and Screen 3 should use `st.session_state['codebook_pool_rows']` (D_pool) for LM-assisted operations.")
