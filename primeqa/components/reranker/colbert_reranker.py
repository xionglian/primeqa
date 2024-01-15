from typing import List, Dict
import os
from dataclasses import dataclass, field
import json
import numpy as np
import warnings
import pdb

from primeqa.components.base import Reranker as BaseReranker
from primeqa.ir.dense.colbert_top.colbert.infra.config import ColBERTConfig
from primeqa.ir.dense.colbert_top.colbert.searcher import Searcher

@dataclass
class ColBERTReranker(BaseReranker):
    """_summary_

    Args:
        model (str, optional): Model to load.
        max_num_documents (int, optional): Maximum number of reranked document to return. Defaults to -1.
        include_title (bool, optional): Whether to concatenate text and title. Defaults to True

    Important:
    1. Each field has the metadata property which can carry additional information for other downstream usages.
    2. Two special keys (api_support and exclude_from_hash) are defined in "metadata" property.
        a. api_support (bool, optional): If set to True, that parameter is exposed via the service layer. Defaults to False.
        b. exclude_from_hash (bool,optional): If set to True, that parameter is not considered while building the hash representation for the object. Defaults to False.

    Returns:
        _type_: _description_

    """

    model: str = field(
        default="drdecr",
        metadata={
            "name": "Model",
            "api_support": True,
            "description": "Path to model",
        },
    )

    doc_maxlen: int = field(
        default=180,
        metadata={
            "name": "doc_maxlen",
            "api_support": True,
            "description": "maximum document length (sub-word units)",
        },
    )

    query_maxlen: int = field(
        default=32,
        metadata={
            "name": "query_maxlen",
            "api_support": True,
            "description": "maximum query length (sub-word units)",
        },
    )

    def __post_init__(self):
        self._config = ColBERTConfig(
            index_root=None,
            index_name=None,
            index_path=None,
            #model_type=self.model_type,
            doc_maxlen=self.doc_maxlen,
            query_maxlen = self.query_maxlen
        )

        # Placeholder variables
        self._searcher = None

    def __hash__(self) -> int:
        # Step 1: Identify all fields to be included in the hash
        hashable_fields = [
            k
            for k, v in self.__class__.__dataclass_fields__.items()
            if not "exclude_from_hash" in v.metadata
            or not v.metadata["exclude_from_hash"]
        ]

        # Step 2: Run
        return hash(
            f"{self.__class__.__name__}::{json.dumps({k: v for k, v in vars(self).items() if k in hashable_fields}, sort_keys=True)}"
        )

    def load(self, *args, **kwargs):
        print('xltest')
        self._loaded_model = Searcher(
            None,
            checkpoint=self.model,
            collection=None,
            config=self._config,
            rescore_only=True
        )

    def train(self, *args, **kwargs):
        pass

    def eval(self, *args, **kwargs):
        pass

    def predict(self, queries: List[str],
                    documents:  List[List[Dict]],
                    *args,
                    **kwargs):
        warnings.warn("The 'predict' method is deprecated. Please use `rerank'", FutureWarning)
        return self.rerank(queries, documents, *args, **kwargs)

    def rerank(self, query_infos: List[Dict],
                    recall_info:  List[List[Dict]],
                    *args,
                    **kwargs):
        """
        Args:
            queries (List[str]): search queries
            texts (List[List[Dict]]): For each query, a list of documents to rerank
                where each document is a dictionary with the following structure:
                {
                    "document": {
                        "text": "A man is eating food.",
                        "document_id": "0",
                        "title": "food"
                    },
                    "score": 1.4
                }

        Returns:
            List[List[Dict]] A list of reranked documents in the same structure as the input documents
             with the score replaced with the reranker score for each query.
        """
        # Step 1: Locally update object variable values, if provided
        max_num_documents = (
            kwargs["max_num_documents"]
            if "max_num_documents" in kwargs
            else self.max_num_documents
        )


        include_title = (
            kwargs["include_title"]
            if "include_title" in kwargs
            else self.include_title
        )

        ranking_results = []
        for query_info, queue_docs in zip(query_infos, recall_info):
            query = query_info.get('query', '')
            date = query_info.get('query_understand',{}).get('ReportScense:time', '')
            cate = query_info.get('query_understand',{}).get('ReportScense:category', '')
            brand = query_info.get('query_understand',{}).get('ReportScense:brand', '')
            region = query_info.get('query_understand',{}).get('ReportScense:region', '')
            metric = query_info.get('query_understand',{}).get('metric', '')
            texts = []
            dates = []
            cates = []
            brands = []
            regions = []
            metrics = []
            chunk_ids = []
            normalized_similarities = []
            print('query:', query)
            print('queue_docs:', queue_docs)
            self.normalize_scores(queue_docs)
            for queue in queue_docs:
                for chunk in queue['chunks']:
                    normalized_similarities.append(chunk.get('normalized_similarity', 0.0))
                    text = chunk.get('chunk_info',{}).get('content', '')
                    cate = chunk.get('chunk_info',{}).get('cate', '')
                    date = chunk.get('chunk_info',{}).get('date', '')
                    brand = chunk.get('chunk_info',{}).get('brand', '')
                    metric = chunk.get('chunk_info',{}).get('metric', '')
                    region = chunk.get('chunk_info',{}).get('chunk_region', '')
                    chunk_ids.append({'chunk_id':chunk.get('chunk_info',{}).get('chunk_id')})

                    page_content = chunk.get('page_info',{}).get('content', '')
                    if page_content != '':
                        text = text + '\n' + page_content

                    title = chunk.get('doc_info',{}).get('doc_title', '')
                    if title != '':
                        text = text + '\n' + title
                    texts.append(text)

                    page_metric = chunk.get('page_info',{}).get('metric', '')
                    if page_metric != '':
                        metric = metric + '\n' + page_metric
                    metrics.append(metric)

                    doc_cate = chunk.get('doc_info',{}).get('doc_lv2_category', '')
                    if doc_cate != '':
                        cate = cate + '\n' + doc_cate
                    cates.append(cate)

                    doc_brand = chunk.get('doc_info',{}).get('brand', '')
                    if doc_brand != '':
                        brand = brand + '\n' + doc_brand
                    brands.append(brand)

                    doc_region = chunk.get('doc_info',{}).get('region', '')
                    if doc_region != '':
                        region = region + '\n' + doc_region
                    regions.append(region)

                    doc_date = chunk.get('doc_info', {}).get('doc_date', '')
                    if doc_date != '':
                        date = date + '\n' + doc_date
                    dates.append(date)

            scores = self._loaded_model.rescore(query, texts).tolist()
            date_scores = self._loaded_model.rescore(date, dates).tolist()
            cate_scores = self._loaded_model.rescore(cate, cates).tolist()
            brand_scores = self._loaded_model.rescore(brand, brands).tolist()
            region_scores = self._loaded_model.rescore(region, regions).tolist()
            metric_scores = self._loaded_model.rescore(metric, metrics).tolist()
            # 定义不同特征的权重，后续有样本后，可以基于样本训练一个线性层
            weight_text = 0.5
            weight_date = 0.1
            weight_cate = 0.1
            weight_brand = 0.1
            weight_region = 0.1
            weight_metric = 0.1
            weight_recall_rank = 0.1

            # 计算加权分数
            weighted_scores = [
                (scores[i] * weight_text +
                date_scores[i] * weight_date +
                cate_scores[i] * weight_cate +
                brand_scores[i] * weight_brand +
                region_scores[i] * weight_region +
                metric_scores[i] * weight_metric)*
                normalized_similarities[i]
                for i in range(len(scores))
            ]

            ranked_passage_indexes = np.array(weighted_scores).argsort()[::-1][:max_num_documents if max_num_documents > 0 else len(scores)].tolist()


            results = []
            for idx in ranked_passage_indexes:
                chunk_ids[idx]['score'] = weighted_scores[idx]
                results.append(chunk_ids[idx])
            ranking_results.append(results)

        return ranking_results

    def normalize_scores(self, recall_info):
        for queue in recall_info:
            similarities = [chunk['similarity'] for chunk in queue['chunks']]
            max_sim = max(similarities)
            min_sim = min(similarities)

            # 线性映射，将最高分映射到1，最低分映射到0.6
            for chunk in queue['chunks']:
                normalized_score = 0.6 + 0.4 * (chunk['similarity'] - min_sim) / (max_sim - min_sim)
                chunk['normalized_similarity'] = normalized_score
        return

