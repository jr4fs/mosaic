

# screens/upload_screen.py
"""
Screen 1 — Upload Data & Define Task (revised)

Features:
 - Upload CSV or Excel file (pandas)
 - Preview dataframe
 - Select column(s) that contain the text to annotate
 - Confirm columns to create st.session_state.uploaded_data (list of text rows)
 - Define task metadata: name, description, task type (multiclass/multilabel)
 - Define labels / codebook entries (label name, description, optional keywords)
 - Persist selected_columns / task metadata / codebook_struct to Task row in DB
"""

import streamlit as st
import pandas as pd
import json
from utils.data_utils import clean_rows
from services.db import update_task, get_task, create_task

# small helper
def preview_dataframe(df, n=5):
    if df is None or df.empty:
        st.info("No dataset loaded yet.")
        return
    st.markdown(f"Showing dataframe preview ({min(n, len(df))} rows)")
    st.dataframe(df.head(n))


def render():
    st.header("Upload Data & Define Your Task")
    st.markdown(
        "Upload an Excel (.xlsx) or CSV (.csv) file. Preview it, choose which column(s) to annotate, "
        "then define task metadata and labels (the codebook)."
    )

    # Ensure the task_meta dict exists
    if "task_meta" not in st.session_state:
        st.session_state.task_meta = {}

    # If a current_task_id exists, try to load its DB values into session (but do not overwrite unsaved in-memory edits).
    current_task_id = st.session_state.get("current_task_id")
    if current_task_id and not st.session_state.get("_task_loaded_from_db"):
        try:
            db_task = get_task(current_task_id)
            if db_task:
                # populate session task_meta from DB fields (JSON strings)
                try:
                    cb = json.loads(db_task.codebook_struct) if db_task.codebook_struct else []
                except Exception:
                    cb = []
                try:
                    sel_cols = json.loads(db_task.selected_columns) if db_task.selected_columns else []
                except Exception:
                    sel_cols = []
                st.session_state.task_meta = {
                    "task_name": db_task.name or st.session_state.task_meta.get("task_name", ""),
                    "description": db_task.description or st.session_state.task_meta.get("description", ""),
                    "task_type": db_task.task_type or st.session_state.task_meta.get("task_type", "multiclass"),
                    "codebook_struct": cb or st.session_state.task_meta.get("codebook_struct", []),
                    "selected_columns": sel_cols or st.session_state.task_meta.get("selected_columns", []),
                }
            # mark as loaded so we don't re-load every rerun and clobber in-progress edits
            st.session_state["_task_loaded_from_db"] = True
        except Exception:
            # fail silently but show a notice
            st.warning("Could not load task metadata from DB for this task (continuing with session values).")

    # -------------------------
    # File upload + preview
    # -------------------------
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"], key="upload_file")
    df = None
    if uploaded_file is not None:
        try:
            if uploaded_file.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.success(f"Loaded `{uploaded_file.name}` — {df.shape[0]} rows, {df.shape[1]} columns")
            # store raw DF in session (useful later)
            st.session_state.uploaded_df = df.copy()
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            df = None
    else:
        # use an already-loaded dataframe from session if present
        df = st.session_state.get("uploaded_df", None)

    preview_dataframe(df, n=5)

    # -------------------------
    # Column selection
    # -------------------------
    if df is not None:
        cols = list(df.columns)
        st.markdown("**Select column(s) that contain the text to annotate.**")
        # pre-select columns if they were previously chosen for this task
        preselected = st.session_state.task_meta.get("selected_columns", [])
        selected_cols = st.multiselect("Columns to use", cols, default=preselected, key="upload_selected_cols")

        st.write("Selected columns:", selected_cols)

        if st.button("Confirm columns and load for annotation", key="confirm_columns"):
            if not selected_cols:
                st.error("Please select at least one column.")
            else:
                # Build rows by concatenating the selected columns for each row
                working = df[selected_cols].astype(str).fillna("").apply(
                    lambda row: " ".join([c for c in row.values if str(c).strip()]), axis=1
                )
                rows = working.tolist()
                rows = clean_rows(rows)
                st.session_state.uploaded_data = rows
                st.session_state.task_meta["selected_columns"] = selected_cols
                st.session_state.uploaded_df = df.copy()
                st.success(f"Loaded {len(rows)} rows into annotation pipeline.")

                # Persist selected_columns to DB if a current_task_id exists
                if st.session_state.get("current_task_id"):
                    try:
                        update_task(st.session_state["current_task_id"], selected_columns=selected_cols)
                    except Exception as e:
                        st.error(f"Failed to persist selected columns to DB: {e}")

    st.markdown("---")
    st.markdown("### Task definition")
    # Provide defaults from session task_meta (if any) BEFORE creating widgets
    default_name = st.session_state.task_meta.get("task_name", "")
    default_desc = st.session_state.task_meta.get("description", "")
    default_type = st.session_state.task_meta.get("task_type", "multiclass")

    task_name = st.text_input("Task name", value=default_name, key="upload_task_name")
    task_desc = st.text_area("Short task description (instructions for annotators)", value=default_desc, height=120, key="upload_task_desc")
    task_type = st.selectbox("Annotation type", options=["multiclass", "multilabel"], index=0 if default_type == "multiclass" else 1, key="upload_task_type")

    if st.button("Save task metadata", key="upload_save_task"):
        st.session_state.task_meta["task_name"] = task_name
        st.session_state.task_meta["description"] = task_desc
        st.session_state.task_meta["task_type"] = task_type

        # Persist to DB if current_task_id exists; otherwise create a new Task (and set current_task_id)
        try:
            if st.session_state.get("current_task_id"):
                update_task(
                    st.session_state["current_task_id"],
                    name=task_name,
                    description=task_desc,
                    task_type=task_type,
                )
            else:
                # If there's no current_task_id, create a DB Task record for the current user (if available)
                owner = st.session_state.get("current_user") or "local"
                t = create_task(owner=owner, name=task_name or "Untitled Task", description=task_desc or "", task_type=task_type)
                st.session_state.current_task_id = t.id
                # mark task as loaded from DB so the next rerun doesn't clobber session edits
                st.session_state["_task_loaded_from_db"] = True
            st.success("Saved task metadata.")
        except Exception as e:
            st.error(f"Failed to persist task metadata to DB: {e}")

    # -------------------------
    # Codebook / labels editing
    # -------------------------
    st.markdown("---")
    st.markdown("### Define Labels")
    st.markdown("Define label names, a short description, and optional keywords/guidelines (comma-separated).")

    codebook = st.session_state.task_meta.get("codebook_struct", [])
    if codebook:
        st.markdown("#### Current labels")
        for i, e in enumerate(codebook):
            lbl = e.get("label", "<no-label>")
            desc = e.get("desc", "")
            kw = ", ".join(e.get("keywords", [])) if e.get("keywords") else ""
            st.write(f"**{i+1}. {lbl}** — {desc}  \nKeywords: {kw}")

    st.markdown("#### Add a new label")
    # Do NOT assign to st.session_state keys after widget creation (we only read values below)
    new_label_name = st.text_input("Label name", key="new_label_name")
    new_label_desc = st.text_area("Label short description", height=80, key="new_label_desc")
    new_label_keywords = st.text_input("Keywords / examples (comma-separated)", key="new_label_keywords")

    if st.button("Add label to codebook", key="add_label"):
        if not new_label_name.strip():
            st.error("Label must have a name.")
        else:
            kw_list = [k.strip() for k in new_label_keywords.split(",")] if new_label_keywords else []
            entry = {"label": new_label_name.strip(), "desc": new_label_desc.strip(), "keywords": [k for k in kw_list if k]}
            cb = st.session_state.task_meta.get("codebook_struct", [])
            cb.append(entry)
            st.session_state.task_meta["codebook_struct"] = cb
            st.success(f"Added label '{new_label_name.strip()}' to codebook.")

            # Persist codebook to DB if current_task_id exists
            if st.session_state.get("current_task_id"):
                try:
                    update_task(st.session_state["current_task_id"], codebook_struct=st.session_state.task_meta["codebook_struct"])
                except Exception as e:
                    st.error(f"Failed to save label to DB: {e}")

    st.markdown("#### Import codebook (optional)")
    cb_file = st.file_uploader("Upload JSON codebook (array of objects)", type=["json"], key="upload_cb_file")
    if cb_file:
        try:
            loaded = json.load(cb_file)
            if isinstance(loaded, list):
                normalized = []
                for obj in loaded:
                    label = obj.get("label") or obj.get("name")
                    desc = obj.get("desc", "")
                    kws = obj.get("keywords", []) or obj.get("kw", [])
                    if label:
                        normalized.append({"label": label, "desc": desc, "keywords": kws})
                if normalized:
                    st.session_state.task_meta["codebook_struct"] = normalized
                    st.success(f"Imported {len(normalized)} labels into codebook.")
                    if st.session_state.get("current_task_id"):
                        try:
                            update_task(st.session_state["current_task_id"], codebook_struct=normalized)
                        except Exception as e:
                            st.error(f"Failed to persist imported codebook: {e}")
                else:
                    st.error("No valid labels found in JSON.")
            else:
                st.error("JSON root must be a list of label objects.")
        except Exception as e:
            st.error(f"Failed to read JSON: {e}")

    st.markdown("---")
    st.info(
        "After you confirm columns and save the task metadata & labels, "
        "Screen 2 will use these labels as the annotation options (radio for multiclass, checkboxes for multilabel). "
        "Screen 3 will receive a templated prompt which includes this task text + codebook."
    )
