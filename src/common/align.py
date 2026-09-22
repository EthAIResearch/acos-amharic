"""
Word-level BIO tags -> subword-level BIO tags, using a HF fast tokenizer's word_ids().
Requires: transformers (fast tokenizer). Run in your GPU environment, not this sandbox.
"""
LABEL2ID = {"O": 0, "B": 1, "I": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
IGNORE_INDEX = -100  # HF loss ignores this automatically

LABEL2ID_5WAY = {"O": 0, "B-ASP": 1, "I-ASP": 2, "B-OPN": 3, "I-OPN": 4}
ID2LABEL_5WAY = {v: k for k, v in LABEL2ID_5WAY.items()}


def align_labels_to_subwords(word_ids: list[int], word_tags: list[str]) -> list[int]:
    """
    word_ids: output of tokenizer(...).word_ids(batch_index=i) -- one entry per
        subword token, giving the source word index (or None for special tokens).
    word_tags: BIO tags at word level (from bio_labels.build_word_bio).

    Rule: first subword of a word gets the word's tag; continuation subwords of
    a 'B' word get 'I' (so a multi-subword aspect term reads B,I,I,... not B,B,B);
    continuation subwords of an 'I' word stay 'I'; special tokens get IGNORE_INDEX.
    """
    label_ids = []
    prev_word_id = None
    for wid in word_ids:
        if wid is None:
            label_ids.append(IGNORE_INDEX)
        elif wid != prev_word_id:
            label_ids.append(LABEL2ID[word_tags[wid]])
        else:
            # continuation subword of the same word
            tag = word_tags[wid]
            label_ids.append(LABEL2ID["I"] if tag in ("B", "I") else LABEL2ID["O"])
        prev_word_id = wid
    return label_ids


def align_labels_to_subwords_5way(word_ids: list[int], word_tags: list[str]) -> list[int]:
    """
    Subword alignment for unified 5-way BIO tags (O, B-ASP, I-ASP, B-OPN, I-OPN).
    First subword receives the word tag; continuation subwords of B-ASP/I-ASP
    become I-ASP; continuation subwords of B-OPN/I-OPN become I-OPN.
    Special tokens (wid is None) receive IGNORE_INDEX (-100).
    """
    label_ids = []
    prev_word_id = None
    for wid in word_ids:
        if wid is None:
            label_ids.append(IGNORE_INDEX)
        elif wid != prev_word_id:
            label_ids.append(LABEL2ID_5WAY[word_tags[wid]])
        else:
            tag = word_tags[wid]
            if tag in ("B-ASP", "I-ASP"):
                sub_tag = "I-ASP"
            elif tag in ("B-OPN", "I-OPN"):
                sub_tag = "I-OPN"
            else:
                sub_tag = "O"
            label_ids.append(LABEL2ID_5WAY[sub_tag])
        prev_word_id = wid
    return label_ids


def decode_subword_predictions(
    word_ids: list[int],
    pred_ids: list[int],
    strategy: str = "entity_first",
) -> list[str]:
    """Inverse: take the model's per-subword predictions and reduce back to one
    tag per word.

    Strategies:
      - "first": uses the first subword's prediction.
      - "entity_first": entity-aware aggregation. If the first subword predicted non-O,
        uses it; if the first subword was 'O' (e.g. grammatical prefix) but a later
        subword predicted an entity, rescues the entity tag for the word.
    """
    if strategy == "first":
        word_tags: dict[int, str] = {}
        for wid, pid in zip(word_ids, pred_ids):
            if wid is None:
                continue
            if wid not in word_tags:
                word_tags[wid] = ID2LABEL[pid]
        n_words = max(word_tags) + 1 if word_tags else 0
        return [word_tags.get(i, "O") for i in range(n_words)]

    word_subwords: dict[int, list[int]] = {}
    for wid, pid in zip(word_ids, pred_ids):
        if wid is None:
            continue
        if wid not in word_subwords:
            word_subwords[wid] = []
        word_subwords[wid].append(pid)

    n_words = max(word_subwords) + 1 if word_subwords else 0
    final_tags: list[str] = ["O"] * n_words

    for wid in range(n_words):
        pids = word_subwords.get(wid, [])
        if not pids:
            final_tags[wid] = "O"
            continue
        p0 = pids[0]
        if p0 != 0:
            final_tags[wid] = ID2LABEL[p0]
        else:
            non_o = [p for p in pids if p != 0]
            if not non_o:
                final_tags[wid] = "O"
            else:
                tag = ID2LABEL[non_o[0]]
                prev_tag = final_tags[wid - 1] if wid > 0 else "O"
                if tag == "I" and prev_tag not in ("B", "I"):
                    tag = "B"
                final_tags[wid] = tag

    return final_tags


def decode_subword_predictions_5way(
    word_ids: list[int],
    pred_ids: list[int],
    strategy: str = "entity_first",
) -> list[str]:
    """Inverse for 5-way BIO: takes per-subword predictions and reduces to word-level tags.

    Strategies:
      - "first": uses the first subword's prediction.
      - "entity_first": entity-aware aggregation. If the first subword predicted non-O,
        uses it; if the first subword was 'O' (common in Amharic with morphological prefixes
        like 'የ-', 'በ-', 'አል-') but a later subword predicted an entity, rescues the entity
        tag for the word, promoting orphaned I tags to B when starting a new span.
    """
    if strategy == "first":
        word_tags: dict[int, str] = {}
        for wid, pid in zip(word_ids, pred_ids):
            if wid is None:
                continue
            if wid not in word_tags:
                word_tags[wid] = ID2LABEL_5WAY[pid]
        n_words = max(word_tags) + 1 if word_tags else 0
        return [word_tags.get(i, "O") for i in range(n_words)]

    word_subwords: dict[int, list[int]] = {}
    for wid, pid in zip(word_ids, pred_ids):
        if wid is None:
            continue
        if wid not in word_subwords:
            word_subwords[wid] = []
        word_subwords[wid].append(pid)

    n_words = max(word_subwords) + 1 if word_subwords else 0
    final_tags: list[str] = ["O"] * n_words

    for wid in range(n_words):
        pids = word_subwords.get(wid, [])
        if not pids:
            final_tags[wid] = "O"
            continue
        p0 = pids[0]
        if p0 != 0:
            final_tags[wid] = ID2LABEL_5WAY[p0]
        else:
            non_o = [p for p in pids if p != 0]
            if not non_o:
                final_tags[wid] = "O"
            else:
                tag = ID2LABEL_5WAY[non_o[0]]
                prev_tag = final_tags[wid - 1] if wid > 0 else "O"
                if tag == "I-ASP" and prev_tag not in ("B-ASP", "I-ASP"):
                    tag = "B-ASP"
                elif tag == "I-OPN" and prev_tag not in ("B-OPN", "I-OPN"):
                    tag = "B-OPN"
                final_tags[wid] = tag

    return final_tags

