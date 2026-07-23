from pathlib import Path
import base64

import streamlit as st

from demo.streamlit_utils import (
    create_bytes_stream,
    process_document,
    set_extractor, 
    set_st_state,
    convert_to_excel
)
from src.entities_extractor.entities import DATA
from src.utils import read_yaml_file

read_yaml_file = st.cache_data(read_yaml_file)

_config = read_yaml_file(Path(__file__).parent / "config.yaml")
llm_config = _config["llm"]
entities_extractor_config = _config["entities-extractor"]
docprocess_config = _config["doc_processing"]


set_st_state(
    {
        "document": None,
        "extraction_result": {},
        "DATA": DATA.copy(),
        "entities_to_extract": {}
    }
)
extractor = set_extractor(entities_extractor_config)

with st.sidebar:
    uploaded_files = st.file_uploader(
        "Upload a file",
        accept_multiple_files=True,
        type="pdf"
    )

    ####################
    # PROCESS DOCUMENT #
    ####################
    if st.button("Ingest document"):
        if uploaded_files:
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

        else:
            st.warning("Please upload a document")

    ####################
    # EXTRACT ENTITIES #
    ####################
    st.subheader("Information to extract")
    with st.expander("Include new information"):
        st.subheader("Define New Entity")
        new_entity_name = st.text_input("New Entity Name", key="dynamic_entity_name")
        new_entity_description = st.text_area("Description", key="dynamic_entity_description")

        if st.button("Add New Entity"):
            if new_entity_name:
                st.session_state["DATA"][new_entity_name] = {
                    "name": new_entity_name,
                    "description": new_entity_description
                }
            else:
                st.warning("Please enter a name for the new entity.")

    default_entities = list(st.session_state["DATA"].keys())
    selected_entities = []
    for entity_name in default_entities:
        # Use session state to track selection
        key = f"extract_{entity_name}"
        if st.checkbox(f"{entity_name}", key=key, value=True):
            selected_entities.append(entity_name)

    if st.button("Extract information"):
        if not st.session_state.get("document"):
            st.warning("Please ingest a document first.")
        elif not selected_entities:
            st.warning("Please select at least one entity to extract.")
        else:
            for entity_name in selected_entities:
                if entity_name in st.session_state["DATA"]:
                    # We pass the specific entity key/name defined in DATA
                    st.session_state["entities_to_extract"][entity_name] = st.session_state["DATA"][entity_name]

            st.info(f"Extracting entities: {', '.join(selected_entities)}")
            result = extractor.invoke(
                metadata = st.session_state["entities_to_extract"],
                text = st.session_state["document"]
            )
            st.session_state["extraction_result"] = result["metadata"]

tab_doc, tab_excel = st.tabs(["Results", "Excel format"])

with tab_doc:
    st.header("Information extracted from the document")
    if uploaded_files:
        col1, col2 = st.columns([3, 2], gap="large")
        
        # 2. Process your PDF
        pdf_bytes = uploaded_files[0].read()
        base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
        
        pdf_display = f"""
        <iframe 
            src="data:application/pdf;base64,{base64_pdf}" 
            width="100%" 
            height="900" 
            style="border: none; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.08);"
            type="application/pdf">
        </iframe>
        """
        
        # 3. Render the PDF viewer in the first column
        with col1:
            st.subheader("📄 Document Viewer")
            st.markdown(pdf_display, unsafe_allow_html=True)
            
        # 4. Render the extraction results in the second column
        with col2:
            st.subheader("🧠 Extracted Data")
            
            # Pull your data from session state
            extraction_data = st.session_state.get("extraction_result", {})
            
            if extraction_data:
                display_res = ["## Information extracted:"]
                
                for key, entity_info in extraction_data.items():
                    display_res.append(f"* **{key}**:")
                    values_list = entity_info.get("values", [])
                    for val in values_list:
                        display_res.append(f"    - {val["value"]}")
                st.markdown("\n".join(display_res))
            else:
                st.warning("No extraction data found yet. Run your analysis pipeline first!")
    else:
        st.info("Please upload a PDF file to view it.")

with tab_excel:
    extraction_result = st.session_state["extraction_result"]
    if extraction_result:
        df_result = convert_to_excel(extraction_result)
        st.dataframe(df_result)

