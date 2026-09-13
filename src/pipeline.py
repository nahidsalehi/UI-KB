"""
RAGPipeline  (unified single-index version)
============================================
No domain routing — one retriever searches across all documents.
Domain detection is used only to frame the LLM system prompt.

Two ways to create a pipeline:

1. Manual:
   dense    = DenseRetriever("qwen3")
   sparse   = BM25Retriever()
   pipeline = RAGPipeline(HybridRetriever(dense, sparse), reranker=Reranker())

2. Factory shorthand (reads config defaults):
   pipeline = RAGPipeline.create(model_name="qwen3", use_hybrid=True, use_reranker=True)
"""

from retrieval import DenseRetriever, BM25Retriever, HybridRetriever, Reranker
from llm   import OllamaClient
from utils import normalize_query, detect_domain, build_context
import config


class RAGPipeline:
    """
    Parameters
    ----------
    retriever : DenseRetriever | HybridRetriever
        Single retriever that covers all domains.
    reranker : Reranker | None
    llm : OllamaClient | None
    top_k_retrieve : int
    top_k_final : int
    """

    def __init__(
        self,
        retriever,
        reranker: Reranker | None    = None,
        llm: OllamaClient | None     = None,
        top_k_retrieve: int          = config.TOP_K_RETRIEVE,
        top_k_final:    int          = config.TOP_K_FINAL,
    ):
        self.retriever      = retriever
        self.reranker       = reranker
        self.llm            = llm
        self.top_k_retrieve = top_k_retrieve
        self.top_k_final    = top_k_final

    # ── retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str) -> tuple[list[dict], str]:
        """
        Search the unified index.
        Returns (results, detected_domain).
        detected_domain is inferred from the query for LLM prompt framing;
        each result also carries its own 'domain' field from chunk metadata.
        """
        query  = normalize_query(query)
        domain = detect_domain(query)

        candidates = self.retriever.search(query, top_k=self.top_k_retrieve)

        if self.reranker:
            final = self.reranker.rerank(query, candidates, top_k=self.top_k_final)
        else:
            final = candidates[: self.top_k_final]

        return final, domain

    # ── full RAG answer ───────────────────────────────────────────────────────

    def answer(self, query: str) -> dict:
        if self.llm is None:
            raise NotImplementedError("Pipeline was built without an LLM client.")

        results, domain = self.retrieve(query)
        context = build_context(results)
        answer  = self.llm.ask(normalize_query(query), context, domain)

        return {
            "answer":  answer,
            "results": results,
            "domain":  domain,
            "query":   normalize_query(query),
        }

    # ── configuration info ────────────────────────────────────────────────────

    def info(self) -> dict:
        r = self.retriever
        if hasattr(r, "dense"):
            model_label = r.dense.label
            model_key   = r.dense.model_name
        else:
            model_label = r.label
            model_key   = r.model_name

        return {
            "embed_model":    model_label,
            "model_key":      model_key,
            "retriever":      type(r).__name__,
            "use_hybrid":     isinstance(r, HybridRetriever),
            "use_reranker":   self.reranker is not None,
            "llm_model":      self.llm.model if self.llm else None,
            "top_k_retrieve": self.top_k_retrieve,
            "top_k_final":    self.top_k_final,
            "preprocessing_profile": config.PROFILE_NAME,
        }

    # ── factory ───────────────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        model_name:   str  = config.APP_EMBED_MODEL,
        use_hybrid:   bool = config.APP_USE_HYBRID,
        use_reranker: bool = config.APP_USE_RERANKER,
        device:       str  = "cpu",
        include_llm:  bool = True,
    ) -> "RAGPipeline":
        dense = DenseRetriever(model_name, device=device)

        if use_hybrid:
            sparse    = BM25Retriever()
            retriever = HybridRetriever(dense, sparse)
        else:
            retriever = dense

        reranker = Reranker(device=device) if use_reranker else None
        llm      = OllamaClient() if include_llm else None

        return cls(retriever=retriever, reranker=reranker, llm=llm)
