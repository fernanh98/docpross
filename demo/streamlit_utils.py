from typing import Any
import tempfile
import os
import io
from requests import Session

from docling.datamodel.base_models import DocumentStream
from streamlit.runtime.uploaded_file_manager import UploadedFile
import streamlit as st

def set_st_state(params: dict[str, Any]) -> None:
    "Initializes streamlit state params"
    for param in params.keys():
        if param not in st.session_state:
            st.session_state[param] = params[param]

def create_temp_file(file: UploadedFile) -> str:
    "Creates a temporary file and returns its location"
    suffix = os.path.splitext(file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(file.getvalue())
        return temp_file.name

# @st.cache_data(show_spinner="Creating bytes Stream", show_time=True)
def create_bytes_stream(_file: UploadedFile) -> DocumentStream:
    """
    Creates a DocumentStream object from an UploadedFile
    """
    file_bytes = _file.getvalue()
    byte_stream = io.BytesIO(file_bytes)
    return DocumentStream(name=_file.name, stream=byte_stream)

# @st.cache_data(show_spinner="Processing document", show_time=True)
def process_document(
    _source: DocumentStream, 
    _base_url: str,
    _session: Session = Session()
):
    _source.stream.seek(0)
    files = {
        "file": (_source.name, _source.stream)
    }
    return _session.post(
        url=_base_url,
        files=files
    )