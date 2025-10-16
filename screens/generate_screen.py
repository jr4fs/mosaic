# screens/generate_screen.py
"""
Screen 4 — Generate Annotations (placeholder + simple rule-based demo).
"""

import streamlit as st
import random
from utils.data_utils import export_annotations_csv

def apply_simple_rules(text, codebook):
    """
    Very small rule-based matcher: check for keywords in text and return label or None.
    codebook: list of dicts with 'label' and optional 'keywords' list.
    """
    lowered = text.lower()
    scores = {}
    for e in codebook:
        label = e.get("label")
        kw = e.get("keywords", [])
        for k in kw:
            if k.lower() in lowered:
                scores[label] = scores.get(label, 0) + 1
    if not scores:
        return None
    # return label with highest score
    return max(scores.items(), key=lambda x: x[1])[0]


def render():
    st.header("Screen 4 — Generate Annotations with Final Codebook")
    if not st.session_state.uploaded_data:
        st.warning("No data loaded. Please upload in Screen 1.")
        return

    st.write(f"Rows loaded: {len(st.session_state.uploaded_data)}")
    st.write(f"Existing human annotations: {len(st.session_state.annotations)}")

    if st.button("Run simulated generation", key="simulate"):
        labels = ["Label A", "Label B", "Label C"]
        existing_idx = {a["row_index"] for a in st.session_state.annotations}
        gen = []
        for i, row in enumerate(st.session_state.uploaded_data):
            if i in existing_idx:
                continue
            label = random.choice(labels)
            gen.append({"row_index": i, "label": label, "auto": True, "text": row})
        st.session_state.generated_annotations = gen
        st.success(f"Generated {len(gen)} annotations (simulated).")

    # Show generated items
    if st.session_state.generated_annotations:
        st.markdown("#### Generated Annotations (simulated)")
        for g in st.session_state.generated_annotations[:50]:
            st.write(f"Row {g['row_index']} → {g['label']}")
            st.write(g["text"])
        # provide a download button
        csv_bytes = export_annotations_csv(st.session_state.generated_annotations)
        st.download_button("Download generated annotations (CSV)", data=csv_bytes, file_name="generated_annotations.csv")
    else:
        st.info("No generated annotations yet. Click 'Run simulated generation' to create dummy outputs.")
