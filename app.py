from pathlib import Path

import streamlit as st

from demo.streamlit_utils import (
    create_bytes_stream,
    process_document
)
from src.llm.implementations import LLM
from src.entities_extractor.extractor import EntitiesExtractorGraph
from src.entities_extractor.entities import DATA
from src.utils import read_yaml_file

_config = read_yaml_file(Path(__file__).parent / "config.yaml")
llm_config = _config["llm"]
docprocess_config = _config["doc_processing"]


uploaded_files = st.file_uploader(
    "Choose a file",
    accept_multiple_files=True,
    type="pdf"
)

if "document" not in st.session_state:
    st.session_state["document"] = None


if uploaded_files is not None and st.button("Ingest document"):
    for file in uploaded_files:
        with st.spinner(f"📄 Processing document: {file.name}..."):
            file_data = create_bytes_stream(file)
            try:
                response = process_document(
                    file_data, 
                    docprocess_config["base_url"]
                )
                if response.status_code == 200:
                    result = response.text
                    st.session_state["document"] = result
                    st.markdown(st.session_state["document"], unsafe_allow_html=False)
                else:
                    st.error(f"Error al llamar a la API de Docling: {response.content}")
                
            except Exception as e:
                st.error(f"Error al procesar {file.name}: {e}")
    
if st.button("Extract entities") and st.session_state["document"] is not None:
    llm_model = LLM(
        model=llm_config["model"],
        base_url=llm_config["base_url"],
        api_key=llm_config["api_key"]
    )
    extractor = EntitiesExtractorGraph(
        llm_model=llm_model
    )
    result = extractor.invoke(
        metadata = DATA,
        text = st.session_state["document"]
    )
    st.json(result["metadata"])
