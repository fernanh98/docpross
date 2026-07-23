from typing import Any
import tempfile
import os
import io
from requests import Session
import pandas as pd

from docling.datamodel.base_models import DocumentStream
from streamlit.runtime.uploaded_file_manager import UploadedFile
import streamlit as st

from src.entities_extractor.extractor import EntitiesExtractorGraph
from src.llm.implementations import LLM

def set_st_state(_params: dict[str, Any]) -> None:
    "Initializes streamlit state params"
    for param in _params.keys():
        if param not in st.session_state:
            st.session_state[param] = _params[param]

@st.cache_resource(show_spinner="Setting extractor")
def set_extractor(_config: dict) -> EntitiesExtractorGraph:
    llm_model = LLM(
        model=_config["llm"]["model"],
        base_url=_config["llm"]["base_url"],
        api_key=_config["llm"]["api_key"]
    )
    return EntitiesExtractorGraph(llm_model)

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

def convert_to_excel(
        result_dict: dict # TODO: change to be a pydantic object instead of a dict
    ) -> pd.DataFrame:
    rows = []
    for key, content in result_dict.items():
        for val_dict in content['values']:
            rows.append({
                'entity': key,
                'description': result_dict[key]['description'],
                'value': val_dict['value']
            })
    return pd.DataFrame(rows)