"""Extractor layer — pluggable file-type extractors."""
from semanticsd.extractors import text  # noqa: F401  (registers TextExtractor)
from semanticsd.extractors import html  # noqa: F401  (registers AFTER text so .html maps to HtmlExtractor)
