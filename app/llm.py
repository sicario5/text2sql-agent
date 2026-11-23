"""LLM integration and the Vanna instance factory.

Vanna's RAG engine (``VannaBase`` + ``ChromaDB_VectorStore``) stores and
retrieves DDL, documentation and question->SQL examples. Vanna needs a "chat"
implementation for the LLM calls; the adapters here provide that for:

* ``openai``             OpenAI API
* ``anthropic``          Anthropic Messages API
* ``openai_compatible``  any OpenAI-compatible endpoint (Azure, Groq, Together,
                          OpenRouter, vLLM, LM Studio, Gemini's OpenAI endpoint ...)
* ``ollama``             local Ollama through its OpenAI-compatible endpoint

The adapters are small and explicit, so the provider is a single env var
(``LLM_PROVIDER``) and model parameters (temperature, output tokens,
timeouts) behave the same everywhere.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from pathlib import Path
from typing import Any

from .config import Settings

try:  # Vanna >= 2.0 keeps the RAG API under vanna.legacy
    from vanna.legacy.base import VannaBase
    from vanna.legacy.chromadb import ChromaDB_VectorStore
except ImportError:  # pragma: no cover - Vanna 0.x layout
    from vanna.base import VannaBase  # type: ignore
    from vanna.chromadb import ChromaDB_VectorStore  # type: ignore

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = "http://localhost:11434/v1"


class LLMError(RuntimeError):
    """Raised when the LLM provider call fails."""


# --------------------------------------------------------------------------- chat adapters
class _ChatMixin:
    """Implements Vanna's abstract chat interface with OpenAI-style message dicts."""

    settings: Settings

    def system_message(self, message: str) -> dict:
        return {"role": "system", "content": message}

    def user_message(self, message: str) -> dict:
        return {"role": "user", "content": message}

    def assistant_message(self, message: str) -> dict:
        return {"role": "assistant", "content": message}

    def log(self, message: str, title: str = "Info") -> None:  # Vanna prints by default
        logger.debug("%s: %s", title, message)


class OpenAIChat(_ChatMixin):
    def _init_llm(self, settings: Settings) -> None:
        from openai import OpenAI

        base_url = settings.llm_base_url or None
        if settings.llm_provider == "ollama":
            base_url = base_url or DEFAULT_OLLAMA_URL
        if settings.llm_provider in ("openai_compatible", "ollama") and not base_url:
            raise ValueError("LLM_BASE_URL is required for the openai_compatible provider")
        api_key = settings.llm_api_key or ("ollama" if settings.llm_provider == "ollama" else None)
        if not api_key:
            raise ValueError("LLM_API_KEY (or OPENAI_API_KEY) is not set")
        self.llm_client = OpenAI(api_key=api_key, base_url=base_url, timeout=settings.llm_timeout_seconds)
        self._send_temperature = True

    def submit_prompt(self, prompt: list[dict], **kwargs: Any) -> str:
        import openai

        params: dict[str, Any] = {"model": self.settings.llm_model, "messages": prompt}
        # OpenAI's newer models use max_completion_tokens; compatible servers still expect max_tokens.
        token_key = "max_completion_tokens" if self.settings.llm_provider == "openai" else "max_tokens"
        params[token_key] = self.settings.llm_max_output_tokens
        if self._send_temperature:
            params["temperature"] = self.settings.llm_temperature
        try:
            response = self.llm_client.chat.completions.create(**params)
        except openai.BadRequestError as exc:
            # Reasoning models reject a custom temperature: retry once without it.
            if "temperature" in str(exc).lower() and self._send_temperature:
                self._send_temperature = False
                params.pop("temperature", None)
                response = self.llm_client.chat.completions.create(**params)
            else:
                raise LLMError(f"LLM request rejected: {exc}") from exc
        except openai.OpenAIError as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc
        return (response.choices[0].message.content or "").strip()


class AnthropicChat(_ChatMixin):
    def _init_llm(self, settings: Settings) -> None:
        import anthropic

        if not settings.llm_api_key:
            raise ValueError("LLM_API_KEY (or ANTHROPIC_API_KEY) is not set")
        kwargs: dict[str, Any] = {"api_key": settings.llm_api_key, "timeout": settings.llm_timeout_seconds}
        if settings.llm_base_url:
            kwargs["base_url"] = settings.llm_base_url
        self.llm_client = anthropic.Anthropic(**kwargs)

    def submit_prompt(self, prompt: list[dict], **kwargs: Any) -> str:
        import anthropic

        system = "\n\n".join(m["content"] for m in prompt if m["role"] == "system")
        messages: list[dict] = []
        for m in prompt:
            if m["role"] == "system":
                continue
            # The Messages API requires alternating roles: merge consecutive same-role messages.
            if messages and messages[-1]["role"] == m["role"]:
                messages[-1]["content"] += "\n\n" + m["content"]
            else:
                messages.append({"role": m["role"], "content": m["content"]})
        if not messages or messages[0]["role"] != "user":
            messages.insert(0, {"role": "user", "content": "(context above)"})
        try:
            response = self.llm_client.messages.create(
                model=self.settings.llm_model,
                system=system,
                messages=messages,
                max_tokens=self.settings.llm_max_output_tokens,
                temperature=self.settings.llm_temperature,
            )
        except anthropic.AnthropicError as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc
        return "".join(block.text for block in response.content if getattr(block, "type", "") == "text").strip()


CHAT_ADAPTERS = {
    "openai": OpenAIChat,
    "openai_compatible": OpenAIChat,
    "ollama": OpenAIChat,
    "anthropic": AnthropicChat,
}


# --------------------------------------------------------------------------- embeddings
class HashingEmbeddingFunction:
    """Dependency-free, offline embedding (hashed word + bigram counts).

    Lower quality than a neural model but needs no download or API key: useful for
    air-gapped deployments and tests. Select with ``EMBEDDING_PROVIDER=hashing``.
    """

    DIM = 512
    _token_re = re.compile(r"[a-z0-9]+")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in input]

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return self(input)

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self(input)

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.DIM
        words = [w.rstrip("s") if len(w) > 3 else w for w in self._token_re.findall(text.lower())]
        features = words + [f"{a}_{b}" for a, b in zip(words, words[1:], strict=False)]
        for feat in features:
            digest = hashlib.md5(feat.encode()).digest()
            idx = int.from_bytes(digest[:4], "little") % self.DIM
            vec[idx] += 1.0 if digest[4] % 2 == 0 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    # chroma >= 1.0 embedding-function protocol
    @staticmethod
    def name() -> str:
        return "text2sql_hashing"

    def is_legacy(self) -> bool:
        return False

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> list[str]:
        return ["cosine", "l2", "ip"]

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config: dict) -> HashingEmbeddingFunction:
        return HashingEmbeddingFunction()

    def validate_config(self, config: dict) -> None:
        return None

    def validate_config_update(self, old_config: dict, new_config: dict) -> None:
        return None


def build_embedding_function(settings: Settings):
    if settings.embedding_provider == "hashing":
        return HashingEmbeddingFunction()
    if settings.embedding_provider == "openai":
        from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

        key = settings.llm_api_key if settings.llm_provider in ("openai", "openai_compatible") else None
        key = os.getenv("EMBEDDING_API_KEY") or key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("EMBEDDING_PROVIDER=openai needs EMBEDDING_API_KEY or OPENAI_API_KEY")
        kwargs: dict[str, Any] = {"api_key": key, "model_name": settings.embedding_model}
        if os.getenv("EMBEDDING_BASE_URL"):
            kwargs["api_base"] = os.getenv("EMBEDDING_BASE_URL")
        return OpenAIEmbeddingFunction(**kwargs)
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    return DefaultEmbeddingFunction()  # local all-MiniLM-L6-v2 (ONNX), downloaded once


# --------------------------------------------------------------------------- factory
def vector_store_path(settings: Settings, database: str, schema: str) -> str:
    """One Chroma store per database/schema/embedding model, so they never mix."""
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{database}__{schema}__{settings.embedding_provider}")
    path = Path(settings.vector_db_path) / safe
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def create_vanna(
    settings: Settings,
    database: str,
    schema: str,
    chroma_client: Any = None,
    adapter: type | None = None,
) -> VannaBase:
    """Build the Vanna instance (Chroma vector store + configured LLM adapter).

    ``adapter`` overrides the chat adapter (tests pass a scripted fake LLM).
    """
    if adapter is None:
        if not settings.llm_model:
            raise ValueError("LLM_MODEL is not set")
        adapter = CHAT_ADAPTERS[settings.llm_provider]
    vanna_cls = type(f"Text2SQLVanna_{adapter.__name__}", (adapter, ChromaDB_VectorStore), {})

    config: dict[str, Any] = {
        "path": vector_store_path(settings, database, schema),
        "embedding_function": build_embedding_function(settings),
        "n_results": settings.n_results,
        "dialect": "PostgreSQL",
        "max_tokens": settings.context_token_budget,  # Vanna: token budget for retrieved context
    }
    if chroma_client is not None:
        config["client"] = chroma_client

    instance = vanna_cls.__new__(vanna_cls)
    ChromaDB_VectorStore.__init__(instance, config=config)
    instance.settings = settings
    instance._init_llm(settings)
    return instance
