"""Shared Gemini client, schema, prompt, and cost tracking."""
import os
import threading
from dataclasses import dataclass, field

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

load_dotenv()

_GEMINI_MODEL = "gemini-3.8-flash"

# Pricing for gemini-3.8-flash (USD per million tokens, introductory until 2026-12-31)
# Standard from 2027-01-01: $1.50/M input, $7.50/M output.
# Thinking tokens are billed at the output rate.
_PRICE_INPUT_PER_M  = 0.75
_PRICE_OUTPUT_PER_M = 3.75


def make_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set — add it to .env")
    return genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=1000))


class CandidateLampPost(BaseModel):
    ymin: int = Field(description="Minimum y coordinate (normalized 0-1000)")
    xmin: int = Field(description="Minimum x coordinate (normalized 0-1000)")
    ymax: int = Field(description="Maximum y coordinate (normalized 0-1000)")
    xmax: int = Field(description="Maximum x coordinate (normalized 0-1000)")
    label_text: str = Field(
        description="The exact text letters read inside this label, e.g. 'L.', 'L.P'"
    )


class LampPostResponse(BaseModel):
    lamp_posts: list[CandidateLampPost] = Field(
        description="List of candidate lamp post detections"
    )


PROMPT = (
    "Here is a tile from a 1880s town plan map. "
    "Search ONLY for potential lamp posts labeled 'L', 'L.' or 'L.P'. "
    "Do not output general map text, numbers, or non-lamp post markings. "
    "For each candidate found, return its bounding box (normalized 0-1000) "
    "and transcribe the exact text read."
)

_THINKING_BUDGET = 4096   # tokens; was uncapped ("HIGH") — reduce to cut latency/cost

GENERATE_CONFIG = types.GenerateContentConfig(
    response_mime_type="application/json",
    response_schema=LampPostResponse,
    thinking_config=types.ThinkingConfig(thinking_budget=_THINKING_BUDGET),
)


def has_l(label: str) -> bool:
    return "L" in label.upper()


@dataclass
class CostTracker:
    in_tokens:    int = 0
    out_tokens:   int = 0
    think_tokens: int = 0
    calls:        int = 0
    errors:       int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def add(self, usage) -> None:
        with self._lock:
            if usage:
                self.in_tokens    += usage.prompt_token_count or 0
                self.out_tokens   += usage.candidates_token_count or 0
                self.think_tokens += usage.thoughts_token_count or 0
            self.calls += 1

    def add_error(self) -> None:
        with self._lock:
            self.errors += 1

    @property
    def cost_usd(self) -> float:
        return (
            self.in_tokens / 1_000_000 * _PRICE_INPUT_PER_M
            + (self.out_tokens + self.think_tokens) / 1_000_000 * _PRICE_OUTPUT_PER_M
        )

    def print_summary(self) -> None:
        print(
            f"API calls: {self.calls}  errors: {self.errors}  "
            f"tokens in: {self.in_tokens:,}  out: {self.out_tokens:,}  "
            f"think: {self.think_tokens:,}  est. cost: ${self.cost_usd:.4f}"
        )
