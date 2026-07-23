from typing import Union
from pathlib import Path 

from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, VlmPipelineOptions, VlmConvertOptions
from docling.datamodel.vlm_engine_options import ApiVlmEngineOptions
from docling.models.inference_engines.vlm.base import VlmEngineType
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.document import ConversionResult


def process_document(
        source: Union[Path, str, DocumentStream],
        external_vlm_url: str,
        vlm_preset: str,
        vlm: bool = False
) -> ConversionResult:
    if not vlm:
        pipeline_options = PdfPipelineOptions(
            generate_picture_images=False, # Required to extract images
            images_scale=2.0, # Higher = better resolution,
            accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU)
        )
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
    else:
        engine_options = ApiVlmEngineOptions(
            engine_type = VlmEngineType.API,
            url = external_vlm_url,
            timeout = 180.0,
            concurrency = 2
        )
        vlm_options = VlmConvertOptions.from_preset(
            vlm_preset,
            engine_options=engine_options
        )
        pipeline_options = VlmPipelineOptions(
            vlm_options=vlm_options,
            enable_remote_services=True
        )
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    pipeline_cls=VlmPipeline
                )
            }
        )
    return converter.convert(source)


def load_doc_into_markdown(
    source: Union[Path, str, DocumentStream],
    external_vlm_url: str,
    vlm_preset: str,
    vlm: bool = False,
) -> str:
    processed_doc = process_document(
        source = source, 
        external_vlm_url = external_vlm_url, 
        vlm_preset = vlm_preset, 
        vlm = vlm
    )
    content = processed_doc.document.export_to_markdown()
    return content
