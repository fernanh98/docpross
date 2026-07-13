from fastapi import FastAPI, UploadFile, HTTPException
import io
from pathlib import Path

from docling.datamodel.base_models import DocumentStream

from modules.docling.src.process import load_doc_into_markdown
from modules.docling.src.utils import read_yaml_file

_config = read_yaml_file(Path(__file__).parent / "../config.yaml")
VLM_URL = _config["external_llm_services"]["base_url"]
VLM_PRESET = _config["external_llm_services"]["vlm_preset"]

app = FastAPI()

@app.post("/process_document")
async def process_document(
    file: UploadFile
) -> str:
    if not file:
        raise HTTPException(status_code=400, detail="No se proporcionó ningún archivo")
    
    file_bytes = await file.read()
    source = DocumentStream(name=file.filename, stream=io.BytesIO(file_bytes))
    md_file = load_doc_into_markdown(
        source, 
        external_vlm_url=VLM_URL,
        vlm_preset=VLM_PRESET,
        vlm=True
    )
    return md_file
