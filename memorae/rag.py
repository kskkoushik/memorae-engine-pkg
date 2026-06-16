from __future__ import annotations

import json
import os
import shutil
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .event_store import EventStore

COLLECTION_NAME = "memorae_events"
EMBED_URL = "https://openrouter.ai/api/v1/embeddings"
META_FILE = ".memorae_index_meta.json"


class EmbeddingError(RuntimeError):
    """Embedding API returned no usable vectors."""


def _pkg_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _persist_dir() -> str:
    path = os.environ.get("CHROMA_PERSIST_DIR", "./data/chroma")
    if not os.path.isabs(path):
        path = os.path.normpath(os.path.join(_pkg_root(), path))
    return path


def _request_embeddings(
    texts: list[str], api_key: str, model: str
) -> list[list[float]]:
    body = json.dumps({"model": model, "input": texts}).encode()
    req = urllib.request.Request(
        EMBED_URL,
        data=body,
        headers={
            "content-type": "application/json",
            "authorization": f"Bearer {api_key}",
            "http-referer": "https://memorae.local",
            "x-title": "Memorae Memory Engine",
        },
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read())

    if data.get("error"):
        err = data["error"]
        msg = err.get("message", err) if isinstance(err, dict) else str(err)
        raise EmbeddingError(f"embedding API error: {msg}")

    items = sorted(data.get("data", []), key=lambda d: d.get("index", 0))
    embeddings: list[list[float]] = []
    for item in items:
        emb = item.get("embedding")
        if emb:
            embeddings.append(emb)
    return embeddings


def embed_texts(
    texts: list[str],
    api_key: str | None = None,
    model: str | None = None,
    *,
    max_retries: int = 2,
) -> list[list[float]]:
    if not texts:
        return []
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise EmbeddingError("OPENROUTER_API_KEY is not set")
    model = model or os.environ.get("MEMORAE_EMBED_MODEL", "openai/text-embedding-3-small")

    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            embeddings = _request_embeddings(texts, api_key, model)
            if len(embeddings) == len(texts):
                return embeddings
            last_err = EmbeddingError(
                f"embedding API returned {len(embeddings)} vector(s) for {len(texts)} input(s)"
            )
        except (EmbeddingError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last_err = e
        except KeyError as e:
            last_err = EmbeddingError(f"embedding API response missing field: {e}")

        if attempt < max_retries:
            time.sleep(0.4 * (attempt + 1))

    raise EmbeddingError(str(last_err) if last_err else "embedding API returned no vectors")


@dataclass
class RagHit:
    idx: int
    score: float
    distance: float
    source: str = ""
    timestamp: str = ""
    content: str = ""


class EventIndex:
    def __init__(self, collection, indexed: int, embed_model: str) -> None:
        self._collection = collection
        self._indexed = indexed
        self._embed_model = embed_model

    @classmethod
    def build_from_store(cls, store: "EventStore") -> "EventIndex | None":
        if os.environ.get("MEMORAE_RAG") != "1":
            return None
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            return None

        persist_dir = _persist_dir()
        os.makedirs(persist_dir, exist_ok=True)
        embed_model = os.environ.get("MEMORAE_EMBED_MODEL", "openai/text-embedding-3-small")
        visible = list(store._visible)
        expected = len(visible)
        meta_path = os.path.join(persist_dir, META_FILE)
        meta = {"count": expected, "model": embed_model, "now": store.now.isoformat()}

        def _event_rows():
            for e in visible:
                yield e.idx, e.content, e.source, e.ts

        return cls._build_index(
            persist_dir, embed_model, expected, meta_path, meta, api_key, _event_rows()
        )

    @classmethod
    def _build_index(
        cls,
        persist_dir: str,
        embed_model: str,
        expected: int,
        meta_path: str,
        meta: dict,
        api_key: str,
        rows,
    ) -> "EventIndex | None":
        import chromadb

        def _load_existing() -> "EventIndex | None":
            try:
                client = chromadb.PersistentClient(path=persist_dir)
                col = client.get_collection(name=COLLECTION_NAME)
                if col.count() == expected:
                    return cls(col, col.count(), embed_model)
            except Exception:
                pass
            return None

        if os.path.exists(meta_path):
            try:
                with open(meta_path, encoding="utf-8") as f:
                    saved = json.load(f)
                if saved.get("count") == expected and saved.get("model") == embed_model:
                    existing = _load_existing()
                    if existing:
                        return existing
            except Exception:
                pass

        if os.path.isdir(persist_dir):
            shutil.rmtree(persist_dir, ignore_errors=True)
        os.makedirs(persist_dir, exist_ok=True)

        client = chromadb.PersistentClient(path=persist_dir)
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        if expected == 0:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f)
            return cls(collection, 0, embed_model)

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []
        for idx, content, source, ts in rows:
            ids.append(str(idx))
            documents.append(content)
            metadatas.append({
                "idx": idx,
                "source": source,
                "timestamp": ts.isoformat().replace("+00:00", "Z"),
            })

        batch_size = 16
        for i in range(0, len(ids), batch_size):
            batch_docs = documents[i : i + batch_size]
            try:
                embeddings = embed_texts(batch_docs, api_key, embed_model)
            except EmbeddingError:
                return None
            collection.add(
                ids=ids[i : i + batch_size],
                documents=batch_docs,
                embeddings=embeddings,
                metadatas=metadatas[i : i + batch_size],
            )

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)

        return cls(collection, collection.count(), embed_model)

    @property
    def indexed_count(self) -> int:
        return self._indexed

    def query(self, query_text: str, top_k: int | None = None) -> list[RagHit]:
        if top_k is None:
            top_k = int(os.environ.get("MEMORAE_RAG_TOP_K", "20"))
        query_text = (query_text or "").strip()
        if not query_text or self._indexed == 0:
            return []

        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        try:
            q_emb = embed_texts([query_text], api_key, self._embed_model)
        except EmbeddingError:
            return []
        if not q_emb:
            return []

        result = self._collection.query(
            query_embeddings=q_emb,
            n_results=min(top_k, self._indexed),
            include=["metadatas", "distances", "documents"],
        )
        hits: list[RagHit] = []
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        for meta, dist, doc in zip(metas, dists, docs):
            idx = int(meta["idx"])
            similarity = max(0.0, 1.0 - float(dist))
            hits.append(RagHit(
                idx=idx,
                score=similarity,
                distance=float(dist),
                source=str(meta.get("source", "")),
                timestamp=str(meta.get("timestamp", "")),
                content=doc or "",
            ))
        return hits

    def stats(self) -> dict:
        return {
            "enabled": True,
            "indexed": self._indexed,
            "embed_model": self._embed_model,
            "persist_dir": _persist_dir(),
        }
