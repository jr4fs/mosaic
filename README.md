# LM-Assisted Annotation Pipeline (modular)

Run:
pip install streamlit requests
streamlit run app.py


- Keep `ollama serve` running to use the LM features on Screen 3.
- Files:
  - `app.py` main entry point
  - `screens/` per-screen UI modules
  - `services/` API wrappers (Ollama)
  - `utils/` data helpers
  - `assets/style.css` CSS

# mosaic