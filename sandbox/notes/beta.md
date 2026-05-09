# Beta Note

Notes on the beta release of the indexing pipeline. Performance improved by
batching embedding calls. The chunker now respects code boundaries via
tree-sitter, falling back to a sliding window for unsupported languages.
