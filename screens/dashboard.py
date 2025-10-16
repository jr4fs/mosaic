# screens/dashboard.py
"""
Dashboard — list / create / open annotation tasks with pretty cards and New Task button at the bottom.

- Each task is shown in a styled card.
- 'Open' uses on_click callback so a single click opens the task immediately.
- 'New Task +' button is rendered after the list (bottom).
"""

import streamlit as st
from services.db import create_task, get_tasks_for_user, get_task
import json
from datetime import datetime

# -------------------------
# Styling for the cards
# -------------------------
_CARD_CSS = """
<style>
/* Card wrapper */
.task-card {
  background: linear-gradient(180deg, #ffffff, #fbfdff);
  border: 1px solid rgba(15,23,42,0.06);
  border-radius: 12px;
  padding: 18px;
  box-shadow: 0 6px 18px rgba(2,6,23,0.04);
  margin-bottom: 14px;
}
/* Title */
.task-card .task-title {
  font-size: 16px;
  font-weight: 700;
  margin-bottom: 6px;
  color: #0f172a;
}
/* Description */
.task-card .task-desc {
  font-size: 13px;
  color: #374151;
  margin-bottom: 8px;
}
/* Meta (labels/owner) */
.task-card .task-meta {
  font-size: 12px;
  color: #6b7280;
  margin-top: 8px;
}
/* Small pill for label count */
.task-pill {
  display: inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  background: linear-gradient(90deg,#6366f1,#3b82f6);
  color: white;
  font-weight:700;
  font-size:12px;
}
.task-actions {
  display:flex;
  gap:8px;
  align-items:center;
  justify-content:flex-end;
  height:100%;
}
/* Make the Open / Delete buttons take full height visually */
.stButton>button {
  height:38px;
}
</style>
"""

def _create_and_open_task(owner: str):
    t = create_task(owner=owner, name="Untitled Task")
    st.session_state.current_task_id = t.id
    st.session_state.screen = "Screen 1"
    st.session_state["_task_loaded_from_db"] = True

def _open_task_cb(task_id: int):
    t = get_task(task_id)
    if not t:
        st.warning("Could not find task (it may have been deleted).")
        return
    try:
        cb = json.loads(t.codebook_struct) if t.codebook_struct else []
    except Exception:
        cb = []
    try:
        sel_cols = json.loads(t.selected_columns) if t.selected_columns else []
    except Exception:
        sel_cols = []

    st.session_state.task_meta = {
        "task_name": t.name or st.session_state.task_meta.get("task_name", ""),
        "description": t.description or st.session_state.task_meta.get("description", ""),
        "task_type": t.task_type or st.session_state.task_meta.get("task_type", "multiclass"),
        "codebook_struct": cb or st.session_state.task_meta.get("codebook_struct", []),
        "selected_columns": sel_cols or st.session_state.task_meta.get("selected_columns", []),
    }
    st.session_state.current_task_id = t.id
    st.session_state.uploaded_df = None
    st.session_state.screen = "Screen 1"
    st.session_state["_task_loaded_from_db"] = True


def render():
    st.markdown(_CARD_CSS, unsafe_allow_html=True)

    st.header("Dashboard - My Annotation Tasks")
    st.markdown("Create, open, and manage your annotation tasks. Tasks are saved to the local DB and scoped to your user account.")

    user = st.session_state.get("current_user")
    if not user:
        st.info("Please sign in in the sidebar (enter a username) to see your tasks.")
        return

    tasks = get_tasks_for_user(user)

    if not tasks:
        st.info("No tasks yet. Use the button below to create one.")
    else:
        for t in tasks:
            # layout: left = card content, right = action buttons
            cols = st.columns([6, 2])
            with cols[0]:
                # Render the card content using HTML so it looks polished
                try:
                    cb = json.loads(t.codebook_struct) if t.codebook_struct else []
                except Exception:
                    cb = []
                labels_summary = ", ".join([e.get("label") for e in cb]) if cb else "No labels"
                created = t.created_at.strftime("%Y-%m-%d %H:%M:%S") if getattr(t, "created_at", None) else ""
                # card HTML
                st.markdown(
                    f"""
                    <div class="task-card">
                      <div class="task-title">{t.name}</div>
                      <div class="task-desc">{(t.description or '')}</div>
                      <div class="task-meta">Labels: {labels_summary}</div>
                      <div class="task-meta">Owner: {t.owner} • Created: {created}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with cols[1]:
                # Actions column
                # Use on_click so opening is immediate
                st.button("Open", key=f"open_task_{t.id}", on_click=_open_task_cb, args=(t.id,))
                if st.button("Delete", key=f"del_task_{t.id}"):
                    st.warning("Delete is not implemented in this prototype. Use DB tools if needed.")

    # New Task button at the bottom, centered
    # ---- Replace the current "New Task" bottom block with this ----
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_l:
        st.button("+ New Task", key="dashboard_new_task", on_click=_create_and_open_task, args=(user,))
    # ----------------------------------------------------------------

