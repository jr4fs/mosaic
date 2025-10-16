# utils/data_utils.py
"""
Lightweight data utilities used by multiple screens.
Keep only safe, deterministic logic here (no network).
"""

import re
import csv
from io import StringIO


def clean_rows(raw_rows):
    """
    Clean rows: strip, collapse whitespace, remove empty lines.
    Returns a list of cleaned strings.
    """
    cleaned = []
    for r in raw_rows:
        if r is None:
            continue
        s = str(r).strip()
        s = re.sub(r"\s+", " ", s)
        if s:
            cleaned.append(s)
    return cleaned


def export_annotations_csv(annotations):
    """
    Return bytes of a CSV suitable for st.download_button.
    annotations: list of dicts with keys row_index, text, label, annotator, note, auto (opt)
    """
    # csv.writer expects a text file-like object, so use StringIO and then encode to bytes.
    out = StringIO()
    writer = csv.writer(out)
    writer.writerow(["row_index", "text", "label", "annotator", "note", "auto"])
    for a in annotations:
        writer.writerow([
            a.get("row_index"),
            a.get("text", ""),
            a.get("label", ""),
            a.get("annotator", ""),
            a.get("note", ""),
            a.get("auto", False),
        ])
    csv_text = out.getvalue()
    out.close()
    return csv_text.encode("utf-8")
