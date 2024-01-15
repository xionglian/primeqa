from primeqa.components.reranker.colbert_reranker import ColBERTReranker
reranker = ColBERTReranker(model="DrDecr.dnn")
reranker.load()

