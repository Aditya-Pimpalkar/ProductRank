"""Shared slowapi limiter (PR-16 / ARCHITECTURE §9.1).

Per-IP rate limiting protects the OpenAI budget and the in-process reranker from a
public demo being hammered. Defined in its own module so routers and the app factory
import the same Limiter instance.
"""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address, default_limits=["120/minute"])
