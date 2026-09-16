import json
import re
from difflib import SequenceMatcher
from pathlib import Path


class TerminologyResolver:
    def __init__(self, vocabulary_path: Path, auto_threshold: float = 0.88, review_threshold: float = 0.65):
        data = json.loads(vocabulary_path.read_text(encoding="utf-8"))
        self.version = data["version"]
        self.terms = data["terms"]
        self.auto_threshold = auto_threshold
        self.review_threshold = review_threshold

    @staticmethod
    def _normalise(value: str) -> str:
        return re.sub(r"[^a-z0-9 ]", "", value.lower()).strip()

    def resolve(self, text: str) -> list[dict]:
        words = text.split()
        found = []
        for term in self.terms:
            candidates = [term["canonical_name"], *term.get("aliases", []), *term.get("phonetic_variants", [])]
            best = (0.0, "")
            for candidate in candidates:
                score = SequenceMatcher(None, self._normalise(candidate), self._normalise(text)).ratio()
                if score > best[0]:
                    best = (score, candidate)
            for start in range(len(words)):
                for end in range(start + 1, min(len(words), start + 5) + 1):
                    phrase = " ".join(words[start:end])
                    for candidate in candidates:
                        score = SequenceMatcher(None, self._normalise(candidate), self._normalise(phrase)).ratio()
                        if score > best[0]:
                            best = (score, phrase)
            score, matched = best
            if score >= self.review_threshold:
                found.append({
                    "text": matched,
                    "resolved_term": term["canonical_name"] if score >= self.auto_threshold else None,
                    "category": term["category"],
                    "confidence": round(score, 3),
                    "source": "medical_vocabulary",
                    "requires_review": score < self.auto_threshold,
                })
        return found

    def apply(self, text: str) -> tuple[str, list[dict]]:
        terms = self.resolve(text)
        resolved = text
        for term in sorted(terms, key=lambda item: len(item["text"]), reverse=True):
            if term["resolved_term"] and not term["requires_review"]:
                resolved = re.sub(re.escape(term["text"]), term["resolved_term"], resolved, flags=re.IGNORECASE)
        return resolved, terms
