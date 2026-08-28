"""
Word-level BIO tag construction and decoding for aspect/opinion term extraction.
Pure Python, no tokenizer dependency -- this is tested locally. The subword
alignment step (word-level BIO -> subword-level BIO using a HF fast tokenizer's
word_ids()) lives in align.py and needs to run where transformers is installed.
"""
from typing import List, Tuple


def build_word_bio(n_tokens: int, spans: List[Tuple[int, int]]) -> List[str]:
    """spans: list of (start, end) word-index spans, end EXCLUSIVE, -1 spans skipped
    (implicit terms have no span to tag). Overlapping spans from different quads
    are simply unioned onto the same tag sequence -- pairing is resolved in stage 2."""
    tags = ["O"] * n_tokens
    for start, end in spans:
        if start == -1 or end == -1:
            continue
        if not (0 <= start < end <= n_tokens):
            raise ValueError(f"Span ({start},{end}) out of range for {n_tokens} tokens")
        tags[start] = "B"
        for i in range(start + 1, end):
            tags[i] = "I"
    return tags


def decode_bio_spans(tags: List[str]) -> List[Tuple[int, int]]:
    """Inverse of build_word_bio: BIO tag sequence -> list of (start, end) spans."""
    spans = []
    start = None
    for i, tag in enumerate(tags + ["O"]):  # sentinel to close trailing span
        if tag == "B":
            if start is not None:
                spans.append((start, i))
            start = i
        elif tag == "O":
            if start is not None:
                spans.append((start, i))
                start = None
        # tag == "I": continue current span (if start is None, treat as noise -> ignore)
    return spans


if __name__ == "__main__":
    # Sanity check against real rows from the dataset.
    import json
    with open("/home/claude/absa/prepared/train.jsonl", encoding="utf-8") as f:
        lines = [json.loads(next(f)) for _ in range(5)]

    for rec in lines:
        n = len(rec["tokens"])
        a_spans = [(q["a_start"], q["a_end"]) for q in rec["quads"]]
        o_spans = [(q["o_start"], q["o_end"]) for q in rec["quads"]]

        a_tags = build_word_bio(n, a_spans)
        o_tags = build_word_bio(n, o_spans)

        a_decoded = decode_bio_spans(a_tags)
        o_decoded = decode_bio_spans(o_tags)

        expected_a = sorted(s for s in a_spans if s[0] != -1)
        expected_o = sorted(s for s in o_spans if s[0] != -1)

        assert sorted(a_decoded) == expected_a, (a_decoded, expected_a)
        assert sorted(o_decoded) == expected_o, (o_decoded, expected_o)

        print("TEXT:", rec["text"][:60])
        print("  tokens:", rec["tokens"])
        print("  aspect tags:", a_tags)
        print("  opinion tags:", o_tags)
        print("  aspect spans decoded:", [" ".join(rec["tokens"][s:e]) for s, e in a_decoded])
        print("  opinion spans decoded:", [" ".join(rec["tokens"][s:e]) for s, e in o_decoded])
        print()

    print("All round-trip checks passed.")
