# tests/test_db.py
import tempfile
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from services.db import Base, Annotation

def test_db_write_and_read(tmp_path):
    # temporary sqlite file path
    db_file = tmp_path / "test_db.sqlite"
    url = f"sqlite:///{db_file}"
    engine = create_engine(url, connect_args={"check_same_thread": False})
    # create tables
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    try:
        ann = Annotation(row_index=1, text="test text", label="Label A", annotator="tester", note="n", auto=False)
        db.add(ann)
        db.commit()
        res = db.query(Annotation).filter_by(row_index=1).one()
        assert res.label == "Label A"
        assert res.text == "test text"
    finally:
        db.close()
