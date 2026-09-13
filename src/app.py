"""
RAG Web Application  (unified single-index version)
=====================================================
  GET  /        → HTML UI
  POST /ask     → server-rendered HTML (AJAX-friendly)
  POST /api/ask → JSON response

Environment variables (all have defaults in config.py):
  UIKB_PROFILE      baseline | hazm       (default: baseline)
  APP_EMBED_MODEL   e5 | bge_m3 | qwen3 | matina   (default: qwen3)
  APP_USE_HYBRID    true | false           (default: true)
  APP_USE_RERANKER  true | false           (default: true)
  APP_PORT          port number            (default: 8000)
  LLM_MODEL         Ollama model name
  LLM_URL           Ollama endpoint URL

Start:
  python app.py
  uvicorn app:app --host 0.0.0.0 --port 8000
"""

import time
import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from pipeline import RAGPipeline
import config

_pipeline: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    print(f"Loading pipeline  [profile={config.PROFILE_NAME}, model={config.APP_EMBED_MODEL}, "
          f"hybrid={config.APP_USE_HYBRID}, reranker={config.APP_USE_RERANKER}] …")
    _pipeline = RAGPipeline.create(
        model_name   = config.APP_EMBED_MODEL,
        use_hybrid   = config.APP_USE_HYBRID,
        use_reranker = config.APP_USE_RERANKER,
    )
    print("Pipeline ready:", _pipeline.info())
    yield


app = FastAPI(title="University RAG", lifespan=lifespan)
templates = Jinja2Templates(directory=str(config.TEMPLATES_DIR))


def _template_ctx(request: Request, **kwargs) -> dict:
    info = _pipeline.info() if _pipeline else {}
    return {
        "request":       request,
        "pipeline_info": info,
        "question":      "",
        "answer":        None,
        "results":       [],
        "domain":        "",
        "elapsed":       None,
        **kwargs,
    }


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", _template_ctx(request))


@app.post("/ask", response_class=HTMLResponse)
async def ask_html(request: Request):
    form     = await request.form()
    question = form.get("question", "").strip()

    if not question:
        return templates.TemplateResponse("index.html", _template_ctx(request))

    t0 = time.time()
    try:
        result = _pipeline.answer(question)
    except Exception as exc:
        return templates.TemplateResponse(
            "index.html",
            _template_ctx(request, question=question, answer=f"خطا در پردازش: {exc}"),
        )
    elapsed = round(time.time() - t0, 2)

    return templates.TemplateResponse(
        "index.html",
        _template_ctx(
            request,
            question = question,
            answer   = result["answer"],
            results  = result["results"],
            domain   = result["domain"],
            elapsed  = elapsed,
        ),
    )


@app.post("/api/ask")
async def ask_json(request: Request):
    body     = await request.json()
    question = body.get("question", "").strip()
    if not question:
        return JSONResponse({"error": "question is required"}, status_code=400)

    t0 = time.time()
    try:
        result = _pipeline.answer(question)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)

    result["elapsed_sec"] = round(time.time() - t0, 2)
    result["pipeline"]    = _pipeline.info()
    return JSONResponse(result)


if __name__ == "__main__":
    uvicorn.run("app:app", host=config.APP_HOST, port=config.APP_PORT, reload=False)
