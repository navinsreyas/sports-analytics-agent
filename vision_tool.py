"""Gemini Flash vision pre-processing layer for the sports analytics agent."""

import io
import os
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

# ---------------------------------------------------------------------------
# Defensive import — if google-genai is missing the module still loads and
# vision features simply return None (app falls back to text-only mode).
# ---------------------------------------------------------------------------
try:
    from google import genai as _genai
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False
    print(
        "[vision_tool] ERROR: 'google-genai' package not found.\n"
        "              Run:  pip install google-genai\n"
        "              Vision features will be disabled until it is installed."
    )

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
_GEMINI_MODEL  = "gemini-flash-latest"

UNSUPPORTED_MESSAGE = (
    "I could not read sports data from this image. "
    "Please upload a clear screenshot of a match result, league table, "
    "player stats card, or fixture list from IPL or Premier League."
)

_EXTRACTION_PROMPT = """\
You are a sports data extractor. Analyze this sports image and extract all
structured information you can find.

Identify:
(1) IMAGE TYPE — is this a match result, league table, player stats card,
    fixture list, or something else (unknown)?
(2) TEAM NAMES — list every team name exactly as written in the image.
(3) PLAYER NAMES — list every player name exactly as written in the image.
(4) SCORES OR STATS — all numerical data: scores, goals, runs, wickets,
    points, averages, strike rates, economy rates, etc.
(5) DATES OR SEASONS — any dates, years, or season identifiers visible.
(6) SPORT AND COMPETITION — the sport (cricket / football) and the
    competition name (IPL, Premier League, etc.) if visible.

If this is NOT a sports image, or you cannot confidently extract sports data,
respond with exactly this word and nothing else: UNSUPPORTED_IMAGE

Otherwise return a clean structured summary using the six headings above.
Do not use JSON. Do not add commentary outside the six headings.
"""

# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------

def _to_pil(image_input) -> Image.Image:
    """Normalise a file path, bytes, or PIL Image into a PIL Image."""
    if isinstance(image_input, Image.Image):
        return image_input
    if isinstance(image_input, (str, Path)):
        return Image.open(image_input)
    if isinstance(image_input, (bytes, bytearray)):
        return Image.open(io.BytesIO(image_input))
    raise TypeError(
        f"image_input must be a file path, bytes, or PIL.Image — got {type(image_input)}"
    )

# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def extract_sports_data(image_input) -> "str | None":
    """
    Send *image_input* to Gemini 1.5 Flash and return structured sports data.

    Returns
    -------
    str   — structured extraction text ready for senior_agent().
    str   — UNSUPPORTED_MESSAGE when the image is not a sports image.
    None  — when the API call fails (caller should fall back to text-only).
    """
    if not _GENAI_AVAILABLE:
        print("[vision_tool] google-genai not available — skipping vision.")
        return None

    if not GEMINI_API_KEY:
        print("[vision_tool] GEMINI_API_KEY not set in .env — skipping vision.")
        return None

    try:
        pil_image = _to_pil(image_input)
    except Exception as e:
        print(f"[vision_tool] Could not decode image: {e}")
        return None

    try:
        client   = _genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=_GEMINI_MODEL,
            contents=[_EXTRACTION_PROMPT, pil_image],
        )
        extracted = response.text.strip()
    except Exception as e:
        print(f"[vision_tool] Gemini API error: {e}")
        return None

    if "UNSUPPORTED_IMAGE" in extracted:
        return UNSUPPORTED_MESSAGE

    return extracted


def is_supported(extracted_text: "str | None") -> bool:
    """
    Return True when *extracted_text* holds real sports data.
    Returns False for None (API failure) or UNSUPPORTED_MESSAGE.
    """
    if extracted_text is None:
        return False
    return extracted_text != UNSUPPORTED_MESSAGE


def build_enriched_prompt(original_question: str, extracted_text: str) -> str:
    """
    Combine Gemini's extraction with the user's question into one prompt
    for senior_agent().
    """
    return (
        "The user has uploaded a sports image. "
        "Here is the structured data extracted from it:\n\n"
        f"{extracted_text}\n\n"
        "Based on the above image data, answer this question: "
        f"{original_question}"
    )
