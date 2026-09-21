"""
Word-level BIO tag construction and decoding for aspect/opinion term extraction.
Pure Python, no tokenizer dependency -- this is tested locally. The subword
alignment step (word-level BIO -> subword-level BIO using a HF fast tokenizer's
word_ids()) lives in align.py and needs to run where transformers is installed.
"""
def build_word_bio(n_tokens: int, spans: list[tuple[int, int]]) -> list[str]:
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


def decode_bio_spans(tags: list[str]) -> list[tuple[int, int]]:
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


def build_word_bio_5way(
    n_tokens: int,
    aspect_spans: list[tuple[int, int]],
    opinion_spans: list[tuple[int, int]],
) -> list[str]:
    """
    Builds a unified 5-way BIO tag sequence:
      'O': Outside
      'B-ASP': Begin Aspect
      'I-ASP': Inside Aspect
      'B-OPN': Begin Opinion
      'I-OPN': Inside Opinion
    Aspect spans are assigned first; opinion spans fill unassigned positions.
    """
    tags = ["O"] * n_tokens
    for start, end in aspect_spans:
        if start == -1 or end == -1:
            continue
        if not (0 <= start < end <= n_tokens):
            raise ValueError(f"Span ({start},{end}) out of range for {n_tokens} tokens")
        tags[start] = "B-ASP"
        for i in range(start + 1, end):
            tags[i] = "I-ASP"

    for start, end in opinion_spans:
        if start == -1 or end == -1:
            continue
        if not (0 <= start < end <= n_tokens):
            raise ValueError(f"Span ({start},{end}) out of range for {n_tokens} tokens")
        if tags[start] == "O":
            tags[start] = "B-OPN"
            for i in range(start + 1, end):
                if tags[i] == "O":
                    tags[i] = "I-OPN"
    return tags


def decode_5way_bio_spans(tags: list[str]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """
    Decodes a unified 5-way BIO tag sequence into (aspect_spans, opinion_spans).
    tags: list of 'O', 'B-ASP', 'I-ASP', 'B-OPN', 'I-OPN'.
    Returns:
        (aspect_spans, opinion_spans)
    """
    a_tags = ["B" if t == "B-ASP" else "I" if t == "I-ASP" else "O" for t in tags]
    o_tags = ["B" if t == "B-OPN" else "I" if t == "I-OPN" else "O" for t in tags]
    return decode_bio_spans(a_tags), decode_bio_spans(o_tags)



if __name__ == "__main__":
    # Sanity check against real rows from the dataset. Run data_prep.py first
    # to generate data/prepared/train.jsonl (it's gitignored, so this won't
    # exist right after a fresh clone). See tests/test_bio_labels.py for the
    # dependency-free unit tests that run in CI.
    import json
    import os

    data_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "prepared", "train.jsonl")
    if not os.path.exists(data_path):
        raise SystemExit(
            f"{data_path} not found. Run data_prep.py first to generate it from your raw TSVs."
        )
    with open(data_path, encoding="utf-8") as f:
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
