"""Extractor layer — pluggable file-type extractors."""
from semanticsd.extractors import text  # noqa: F401  (registers TextExtractor)
from semanticsd.extractors import html  # noqa: F401  (registers AFTER text so .html maps to HtmlExtractor)
from semanticsd.extractors import pdf  # noqa: F401
from semanticsd.extractors import docx  # noqa: F401
from semanticsd.extractors import xlsx  # noqa: F401
from semanticsd.extractors import pptx  # noqa: F401
from semanticsd.extractors import epub  # noqa: F401
