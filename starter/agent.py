from __future__ import annotations

import json #read product data
import re #patterns for finding words in a sentence
import sqlite3 #a tiny database that lives in memory
from pathlib import Path
from sentence_transformers import SentenceTransformer #for embeddings
import numpy as np
import pickle
import os
from anthropic import Anthropic

#finds chunks of letters and numbers (basically words) and removes the unn
TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}

ALLOWED_ATTRIBUTES = (
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
)
MATERIALS = (
    "cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric",
    "alloy", "silver", "gold", "sterling", "stainless", "platinum", "brass", "copper", "titanium",
)
COLOR_WORDS = ("black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey", "purple", "yellow", "orange")
USE_CASE_WORDS = ("hiking", "running", "gym", "winter", "outdoor", "work")
STYLE_WORDS = ("department", "style", "fit", "sleeve", "neck")
SIZE_WORDS = ("size", "sizing", "width", "wide", "narrow")
CATEGORY_WORDS = (
    "shoes", "sneakers", "boots", "sandals", "heels", "flats",
    "jacket", "coat", "sweater", "hoodie", "shirt", "blouse", "dress", "skirt", "pants", "jeans", "shorts",
    "necklace", "bracelet", "earrings", "ring", "watch", "jewelry",
    "bag", "belt", "hat", "scarf", "gloves", "socks",
)
BROWSING_MARKERS = ("still exploring", "just looking", "just browsing", "not sure yet", "exploring")
GENERIC_FILLER_PHRASES = (
    "not quite right yet",
    "i don't have an additional preference",
    "i don't have a preference",
    "please use your judgment",
)

MATERIAL_RE = re.compile(r"\b(" + "|".join(MATERIALS) + r")\b", re.IGNORECASE)
COLOR_RE = re.compile(r"\b(" + "|".join(COLOR_WORDS) + r")\b", re.IGNORECASE)
BUDGET_RE = re.compile(r"(?:\$|<=|under)\s*(\d+)", re.IGNORECASE)
SIZE_NUM_RE = re.compile(r"\bsize\s*(\d+(?:\.\d+)?)\b", re.IGNORECASE)
CATEGORY_RE = re.compile(r"\b(" + "|".join(CATEGORY_WORDS) + r")\b", re.IGNORECASE)
OVERRIDE_PHRASES = ("actually", "never mind", "nevermind", "instead", "forget that", "forget it", "ignore my earlier", "scratch that", "forget everything")


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]: #"I want black shoes" -> ["black", "shoes"]
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


#given a message, actually pulls out the specific values mentioned and returns them as dict
def _extract_slots(text: str) -> dict:
    lower = text.lower()
    found: dict[str, str] = {}
    category_match = CATEGORY_RE.search(lower)
    if category_match:
        found["category"] = category_match.group(1)
    material_match = MATERIAL_RE.search(lower)
    if material_match:
        found["material"] = material_match.group(1)
    color_match = COLOR_RE.search(lower)
    if color_match:
        found["color"] = color_match.group(1)
    size_match = SIZE_NUM_RE.search(lower)
    if size_match:
        found["size"] = size_match.group(1)
    budget_match = BUDGET_RE.search(lower)
    if budget_match:
        found["budget"] = budget_match.group(1)
    for word in USE_CASE_WORDS:
        if word in lower:
            found["use_case"] = word
            break
    for word in ("style", "fit", "sleeve", "neck"):
        if word in lower and word != "style":
            found["style"] = word
            break
    return found

#handles the case whereby the user changes their mind, erase all prev slots
def _is_override(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in OVERRIDE_PHRASES)

def _detect_intent(text: str, slots: dict) -> str:
    lower = text.lower()

    if any(marker in lower for marker in BROWSING_MARKERS):
        return "browsing"

    if any(phrase in lower for phrase in GENERIC_FILLER_PHRASES):
        return "browsing"  # no real new info here — don't treat as a strong buying signal

    # Real extractable slot value found — clear buying signal
    if _extract_slots(text):
        return "buying"

    # Fallback: message has enough substantive content beyond generic filler
    if len(_terms(text)) >= 3:
        return "buying"

    return "browsing"



class Agent:
    """Editable weak baseline: stateless BM25 retrieval with no LLM dependency."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path) #get the data on this agent so that other methods can use later
        self.connection = sqlite3.connect(":memory:") #a databse js for the session
        self._sessions: set[str] = set() #each session is "a user having one convo with the agent"
        self._slots: dict [str,dict] ={} #per-session slot storage
        self._clarify_count: dict[str,int] = {} #tracks clarifying qns asked
        self._asked_attributes: dict[str, set] = {}
        self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self._build_index() #immediately loads every product into the database
        self._build_embeddings()
        self._product_titles = self._load_product_titles()
        self._preference_summary: dict[str, str] = {}
        self._llm_client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def _build_index(self) -> None: #loading the catlog into a search table
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                batch.append(
                    (
                        str(product["parent_asin"]),
                        _text(product.get("title")),
                        _text(product.get("categories")),
                        _text(product.get("features")),
                        _text(product.get("details")),
                        _text(product.get("store")),
                        _text(product.get("description")),
                    )
                )
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()
    
    def _build_embeddings(self) -> None:
        """Compute and store an embedding vector for every product, once (cached to disk)."""
        cache_path = self.catalog_path.parent / "embeddings_cache.pkl"

        if cache_path.exists():
            with cache_path.open("rb") as handle:
                cached = pickle.load(handle)
            self._product_ids = cached["product_ids"]
            self._product_embeddings = cached["embeddings"]
            return

        self._product_ids: list[str] = []
        texts: list[str] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                self._product_ids.append(str(product["parent_asin"]))
                combined = " ".join([
                    _text(product.get("title")),
                    _text(product.get("categories")),
                    _text(product.get("features")),
                ])
                texts.append(combined)

        self._product_embeddings = self._embedding_model.encode(
            texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True
        )

        with cache_path.open("wb") as handle:
            pickle.dump({"product_ids": self._product_ids, "embeddings": self._product_embeddings}, handle)
        
    def _load_product_titles(self) -> dict[str, str]:
        """Quick lookup: parent_asin -> title, for LLM prompting."""
        titles: dict[str, str] = {}
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                titles[str(product["parent_asin"])] = _text(product.get("title"))
        return titles 
    

    def reset(self, session_id: str, user_profile: dict) -> None: #start new session
        # The profile is anonymized and may be used for personalization.
        self._sessions.add(session_id)
        self._slots[session_id] = {
            attr: None for attr in ALLOWED_ATTRIBUTES if attr != "other"
        }
        self._clarify_count[session_id] = 0
        self._asked_attributes[session_id] = set()
        self._preference_summary[session_id] = ""

    #problem to solve: the agent is only looking at the current user_message and doesnt look at what was said in the turns before
    def respond( #ans one turn of the conversation
        self,
        session_id: str,
        user_message: str, #what the shopper says
        turn: int, #1 through 10, imp: unused in this starter (a smarter agent might ask questions early and recommend later)
        #Early turns (like turn 1-2): the agent probably doesn't have enough information yet — maybe it should ask a clarifying question instead of guessing blindly ("What's your budget?" "What color are you looking for?")
        #Later turns (like turn 5+): by now, hopefully enough information has been gathered through the conversation, so the agent should stop asking and start recommending actual products — because turns are limited (remember, max 10, and running out = zero score), so at some point it needs to commit to an answer rather than keep asking more questions forever
        top_k: int, #how many products to return
    ) -> dict:

        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before respond")
        
        current_slots = self._slots[session_id]

        if _is_override(user_message):
            for key in current_slots:
                current_slots[key] = None
        
        new_info = _extract_slots(user_message)
        for key, value in new_info.items():
            current_slots[key] = value
        intent = _detect_intent(user_message, current_slots)
        
        combined_terms = list(dict.fromkeys(_terms(user_message)))
        for value in current_slots.values():
            if value:
                combined_terms.extend(_terms(str(value)))
        unique_terms = list(dict.fromkeys(combined_terms))[:40]
        expression = " OR ".join(f'"{term}"' for term in unique_terms)

        # Get BM25 with actual scores (not just IDs) for proper blending
        if not expression:
            bm25_scores: dict[str, float] = {}
            candidate_count = 0
        else:
            rows = self.connection.execute(
                "SELECT parent_asin, bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) as score "
                "FROM products WHERE products MATCH ? ORDER BY score LIMIT ?",
                (expression, top_k * 5),
            ).fetchall()
            raw_scores = [row[1] for row in rows]
            if raw_scores:
                worst, best = max(raw_scores), min(raw_scores)
                spread = worst - best if worst != best else 1.0
                bm25_scores = {str(row[0]): (worst - row[1]) / spread for row in rows}
            else:
                bm25_scores = {}
            candidate_count = self.connection.execute(
                "SELECT COUNT(*) FROM products WHERE products MATCH ?", (expression,)
            ).fetchone()[0]

        # Embedding scores: cosine similarity is already 0-1-ish after normalization
        embedding_query = user_message
        query_vector = self._embedding_model.encode(
            [embedding_query], convert_to_numpy=True, normalize_embeddings=True
        )[0]
        all_similarities = self._product_embeddings @ query_vector
        top_embed_indices = np.argsort(-all_similarities)[:top_k * 5]
        embedding_scores = {self._product_ids[i]: float(all_similarities[i]) for i in top_embed_indices}

        # Blend: weight BM25 vs embeddings based on detected intent
        if intent == "buying":
            bm25_weight, embed_weight = 1.0, 0.0
        else:  # browsing
            bm25_weight, embed_weight = 0.4, 0.6

        all_ids = set(bm25_scores) | set(embedding_scores)
        blended = {
            pid: bm25_weight * bm25_scores.get(pid, 0.0) + embed_weight * embedding_scores.get(pid, 0.0)
            for pid in all_ids
        }

        candidate_ids = sorted(blended, key=blended.get, reverse=True)[:top_k]

        empty_slots = [k for k, v in current_slots.items() if v is None]
        clarifications_used = self._clarify_count[session_id]

        already_asked = self._asked_attributes[session_id]
        askable_slots = [s for s in empty_slots if s not in already_asked]

        should_clarify = bool(
            clarifications_used < 5
            and (len(empty_slots) >= 6 or candidate_count > 500)
            and askable_slots
        )

        if should_clarify:
            self._clarify_count[session_id] += 1
            ask = askable_slots[0]
            already_asked.add(ask)
            return {
                "message": f"Could you tell me your preference for {ask}?",
                "ask_attribute": ask,
                "recommendations": [{"parent_asin": pid} for pid in candidate_ids],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            }

        llm_ranked_ids, summary, usage = self._llm_rank(user_message, candidate_ids, session_id)
        if summary:
            self._preference_summary[session_id] = summary

        return {
            "message": "Here are the closest matches I found.",
            "ask_attribute": None,
            "recommendations": [{"parent_asin": pid} for pid in llm_ranked_ids],
            "usage": usage,
        }
    
    def _llm_rank(self, user_message: str, candidate_ids: list[str], session_id: str) -> tuple[list[str], str, dict]:
        """Ask Claude to rank candidates and summarize preferences. Returns (ranked_ids, summary, usage)."""
        candidate_lines = "\n".join(
            f"{i+1}. [{pid}] {self._product_titles.get(pid, 'Unknown product')}"
            for i, pid in enumerate(candidate_ids)
        )

        prior_summary = self._preference_summary.get(session_id, "")
        context_line = f'\nWhat we know about this shopper so far: "{prior_summary}"\n' if prior_summary else ""

        prompt = f"""You are ranking products for a shopper based on their message.

Shopper's message: "{user_message}"
{context_line}
Candidate products:
{candidate_lines}

Return ONLY a JSON object with this exact format, no other text:
{{"ranked_ids": ["id1", "id2", ...], "summary": "one short sentence about the shopper's preferences"}}

The ranked_ids list should contain the parent_asin values (the text in brackets) ordered from most to least relevant, using all {len(candidate_ids)} candidates."""

        ranked: list[str] = list(candidate_ids)
        summary: str = ""
        usage = {"prompt_tokens": 0, "completion_tokens": 0}

        try:
            response = self._llm_client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=500,
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "{"},
                ],
            )

            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
            }

            raw_text = "{" + response.content[0].text

            start = raw_text.find("{")
            if start == -1:
                raise ValueError("No JSON object found in LLM response")

            depth = 0
            end = -1
            for i in range(start, len(raw_text)):
                if raw_text[i] == "{":
                    depth += 1
                elif raw_text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            if end == -1:
                raise ValueError("No matching closing brace found")

            json_text = raw_text[start:end + 1]
            result = json.loads(json_text)

            candidate_ranked = [pid for pid in result.get("ranked_ids", []) if pid in candidate_ids]

            summary = result.get("summary", "")
            for pid in candidate_ids:
                if pid not in candidate_ranked:
                    candidate_ranked.append(pid)
            ranked = candidate_ranked
        except Exception as e:
            print(f"LLM ranking failed, falling back to blended order: {e}")

        return ranked, summary, usage