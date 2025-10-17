# LM-Assisted Annotation Pipeline (mosaic)

Run:
pip install streamlit requests
streamlit run app.py --server.address 127.0.0.1

- Keep `ollama serve` running 
- Files:
  - `app.py` main entry point
  - `screens/` per-screen UI modules
  - `services/` API wrappers (Ollama)
  - `utils/` data helpers
  - `assets/style.css` CSS


