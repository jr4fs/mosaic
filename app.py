
# app.py
"""
Main entry point for the LM-Assisted Annotation Pipeline.

- Dashboard is the default landing screen.
- Simple dev-only username "login" in the sidebar (not secure for production).
- init_db() is called to ensure DB tables exist.
- Sidebar lists installed Ollama models (if available).
- Screen modules live under screens/ (dashboard, upload_screen, annotate_screen, codebook_screen, generate_screen).
"""

import streamlit as st

# Services and screens
from services.ollama_api import get_installed_models
from services.db import init_db
from screens import dashboard, upload_screen, annotate_screen, codebook_screen, generate_screen

# Initialize DB (creates tables if missing)
init_db()

# Try to load optional CSS (you may have assets/style.css)
try:
    with open("assets/style.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception:
    # no CSS available — that's fine
    pass

# Page configuration
st.set_page_config(page_title="LM-Assisted Annotation Pipeline", layout="wide")

# -------------------------
# Session state defaults
# -------------------------
if "screen" not in st.session_state:
    st.session_state.screen = "Dashboard"  # landing page by default

# data & task session keys
st.session_state.setdefault("uploaded_data", [])
st.session_state.setdefault("uploaded_df", None)
st.session_state.setdefault("task_meta", {})         # dict: task_name, description, task_type, codebook_struct, selected_columns
st.session_state.setdefault("annotations", [])
st.session_state.setdefault("generated_annotations", [])
st.session_state.setdefault("history", [])
st.session_state.setdefault("stop_generation", False)
st.session_state.setdefault("screen3_prompt", "")
# currently selected task id (Task.id in DB)
st.session_state.setdefault("current_task_id", None)
# current user (dev-only username)
st.session_state.setdefault("current_user", None)



# Simple, prominent login gate (development-only)
def _signin_callback():
    """Called when user clicks Sign in — stores username from the text input."""
    username_val = st.session_state.get("login_username", "")
    if username_val and username_val.strip():
        st.session_state.current_user = username_val.strip()

# If user not signed in, show login form and stop (do not render rest of app)
if not st.session_state.get("current_user"):
    st.title("Welcome — please sign in")
    st.markdown("Enter a username to continue (development mode).")
    # note: we use session_state key 'login_username' so the callback can read it
    st.text_input("Username", key="login_username")
    st.button("Sign in", on_click=_signin_callback, key="login_button")

    # optional: brief note and early exit to prevent rendering any other screens
    st.info("You must sign in to access the dashboard and annotation tools.")
    # stop further rendering of the script this run (user will be signed in on next rerun)
    st.stop()



# -------------------------
# Sidebar: navigation & simple auth (dev)
# -------------------------
st.sidebar.markdown('<div class="sidebar-header">Navigation</div><div class="sidebar-sub">Project steps</div>', unsafe_allow_html=True)

# Navigation buttons (styled in CSS if present)
if st.sidebar.button("Dashboard", key="nav_dashboard"):
    st.session_state.screen = "Dashboard"
if st.sidebar.button("1: Upload Data", key="nav_1"):
    st.session_state.screen = "Screen 1"
if st.sidebar.button("2: Human Annotation", key="nav_2"):
    st.session_state.screen = "Screen 2"
if st.sidebar.button("3: Codebook Development", key="nav_3"):
    st.session_state.screen = "Screen 3"
if st.sidebar.button("4: Generate Annotations", key="nav_4"):
    st.session_state.screen = "Screen 4"

# Current screen badge
current = st.session_state.screen
st.sidebar.markdown(
    f'<div style="margin-top:8px"><strong style="font-size:13px">Current</strong><div style="margin-top:6px">'
    f'<span class="current-pill">{current}</span></div></div>',
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")

# Simple dev-only username "login" (NOT secure; replace for production)
st.sidebar.markdown("### Account (dev-only)")
username = st.sidebar.text_input("Username", value=st.session_state.get("current_user") or "", key="sidebar_username")
if st.sidebar.button("Sign in", key="signin_btn"):
    if username.strip():
        st.session_state.current_user = username.strip()
        st.sidebar.success(f"Signed in as {st.session_state.current_user}")
    else:
        st.sidebar.error("Please enter a username.")
if st.sidebar.button("Sign out", key="signout_btn"):
    st.session_state.current_user = None
    st.sidebar.success("Signed out.")

# Show installed Ollama models (read-only)
st.sidebar.markdown("---")
st.sidebar.markdown("Ollama models (local):")
for m in get_installed_models():
    st.sidebar.write(f"- `{m}`")

# -------------------------
# Main area: dispatch screens
# -------------------------
st.title("LM-Assisted Codebook Development and Annotation")
# st.write("Welcome — use the Dashboard to create or open annotation tasks.")

if st.session_state.screen == "Dashboard":
    dashboard.render()
elif st.session_state.screen == "Screen 1":
    upload_screen.render()
elif st.session_state.screen == "Screen 2":
    annotate_screen.render()
elif st.session_state.screen == "Screen 3":
    codebook_screen.render()
elif st.session_state.screen == "Screen 4":
    generate_screen.render()

# Footer: LM run history
st.markdown("---")
with st.expander("Session history (recent LM runs)"):
    if not st.session_state.history:
        st.write("No runs yet.")
    else:
        for i, item in enumerate(reversed(st.session_state.history[-30:])):
            st.markdown(f"**#{len(st.session_state.history)-i}** • Model: `{item['model']}`")
            st.write("**Prompt:**")
            st.code(item["prompt"][:2000] + ("..." if len(item["prompt"]) > 2000 else ""))
            st.write("**Output:**")
            st.text_area(f"history_out_{i}", value=item["output"], height=140)
