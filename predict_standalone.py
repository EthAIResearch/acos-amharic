"""
Standalone inference and evaluation script for Amharic Aspect-Opinion Pair Extraction (SDRN).

This script is 100% self-contained: it requires NO repository folder structure or source code.
It only requires:
    pip install torch transformers sentencepiece

Usage:
    # 1. Run all 20 example sentences from example_test_sentences.jsonl:
    python3 predict_standalone.py --checkpoint best_model.pt

    # 2. Test a single sentence:
    python3 predict_standalone.py --checkpoint best_model.pt --text "የኢንተርኔት ፍጥነቱ በጣም ፈጣን ነው ግን ዋጋው ውድ ነው"

    # 3. Interactive prompt mode:
    python3 predict_standalone.py --checkpoint best_model.pt --interactive

    # 4. Custom test file:
    python3 predict_standalone.py --checkpoint best_model.pt --file my_sentences.txt
"""

import argparse
import json
import os
import sys

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    F = None
    AutoConfig = None
    AutoModel = None
    AutoTokenizer = None
    TORCH_AVAILABLE = False
    class DummyModule:
        pass
    class DummyNN:
        Module = DummyModule
        Parameter = lambda *args, **kwargs: None
    nn = DummyNN()


NUM_BIO_LABELS = 5
ID2LABEL_5WAY = {0: "O", 1: "B-ASP", 2: "I-ASP", 3: "B-OPN", 4: "I-OPN"}
LABEL2ID_5WAY = {v: k for k, v in ID2LABEL_5WAY.items()}


# =====================================================================
# 1. Self-Contained CRF & Neural Modules
# =====================================================================

class LinearChainCRF(nn.Module):
    """Linear-Chain Conditional Random Field with syntactic BIO constraints."""

    def __init__(self, num_tags: int = 5, bio_weights=None, enforce_bio_constraints: bool = True):
        super().__init__()
        self.num_tags = num_tags
        self.enforce_bio_constraints = enforce_bio_constraints

        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))

        if bio_weights is not None:
            self.register_buffer("bio_weights", bio_weights)
        else:
            self.bio_weights = None

        if self.enforce_bio_constraints and self.num_tags == 5:
            # 0: O, 1: B-ASP, 2: I-ASP, 3: B-OPN, 4: I-OPN
            trans_mask = torch.zeros(num_tags, num_tags, dtype=torch.bool)
            trans_mask[0, 2] = True  # O -> I-ASP
            trans_mask[0, 4] = True  # O -> I-OPN
            trans_mask[1, 4] = True  # B-ASP -> I-OPN
            trans_mask[2, 4] = True  # I-ASP -> I-OPN
            trans_mask[3, 2] = True  # B-OPN -> I-ASP
            trans_mask[4, 2] = True  # I-OPN -> I-ASP
            start_mask = torch.zeros(num_tags, dtype=torch.bool)
            start_mask[2] = True     # START -> I-ASP
            start_mask[4] = True     # START -> I-OPN
            self.register_buffer("forbidden_trans_mask", trans_mask)
            self.register_buffer("forbidden_start_mask", start_mask)
        else:
            self.forbidden_trans_mask = None
            self.forbidden_start_mask = None

        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    def _get_constrained_transitions(self, mask_value: float = -1e4):
        trans = self.transitions.clone()
        start = self.start_transitions.clone()
        if self.forbidden_trans_mask is not None:
            trans = trans.masked_fill(self.forbidden_trans_mask, mask_value)
        if self.forbidden_start_mask is not None:
            start = start.masked_fill(self.forbidden_start_mask, mask_value)
        return trans, start

    def _pack_valid_sequence(self, emissions, tags=None, mask=None):
        B, T, K = emissions.shape
        if mask is None:
            mask = torch.ones((B, T), dtype=torch.bool, device=emissions.device)
        else:
            mask = mask.bool()

        lengths = mask.sum(dim=1)
        T_max = int(lengths.max().item()) if B > 0 else 0
        if T_max == 0:
            return (
                torch.empty((B, 0, K), dtype=emissions.dtype, device=emissions.device),
                torch.empty((B, 0), dtype=torch.long, device=emissions.device),
                torch.empty((B, 0), dtype=torch.bool, device=emissions.device),
                lengths,
            )

        packed_emissions = torch.zeros((B, T_max, K), dtype=emissions.dtype, device=emissions.device)
        packed_tags = torch.zeros((B, T_max), dtype=torch.long, device=emissions.device)
        packed_mask = torch.zeros((B, T_max), dtype=torch.bool, device=emissions.device)

        for b in range(B):
            l_b = int(lengths[b].item())
            if l_b > 0:
                valid_idx = torch.nonzero(mask[b], as_tuple=True)[0]
                packed_emissions[b, :l_b] = emissions[b, valid_idx]
                if tags is not None:
                    packed_tags[b, :l_b] = tags[b, valid_idx]
                packed_mask[b, :l_b] = True

        return packed_emissions, packed_tags, packed_mask, lengths

    def decode(self, emissions: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Viterbi decoding for maximum-a-posteriori sequence inference."""
        if self.bio_weights is not None:
            emissions = emissions * self.bio_weights.view(1, 1, -1)

        B, T = emissions.shape[:2]
        if mask is None:
            mask = torch.ones((B, T), dtype=torch.bool, device=emissions.device)
        else:
            mask = mask.bool()

        packed_emissions, _, packed_mask, lengths = self._pack_valid_sequence(emissions, None, mask)
        feats = packed_emissions.transpose(0, 1)  # (T_packed, B, K)
        mask_t = packed_mask.transpose(0, 1)      # (T_packed, B)
        T_packed = feats.shape[0]

        transitions, start_transitions = self._get_constrained_transitions()

        viterbi_var = start_transitions.unsqueeze(0) + feats[0]  # (B, K)
        backpointers = []

        for t in range(1, T_packed):
            emit_score = feats[t].unsqueeze(1)
            trans_score = transitions.unsqueeze(0)
            next_var = viterbi_var.unsqueeze(2) + trans_score + emit_score
            max_var, bptr = torch.max(next_var, dim=1)
            backpointers.append(bptr)
            viterbi_var = torch.where(mask_t[t].unsqueeze(1), max_var, viterbi_var)

        pred_tensor = torch.zeros((B, T), dtype=torch.long, device=emissions.device)

        for b in range(B):
            l_b = int(lengths[b].item())
            if l_b <= 0:
                continue

            score = viterbi_var[b] + self.end_transitions
            best_last_tag = int(torch.argmax(score).item())
            path = [best_last_tag]

            cur_tag = best_last_tag
            for t in range(l_b - 2, -1, -1):
                cur_tag = int(backpointers[t][b, cur_tag].item())
                path.append(cur_tag)
            path.reverse()

            indices = torch.nonzero(mask[b], as_tuple=True)[0]
            pred_tensor[b, indices] = torch.tensor(path, dtype=torch.long, device=emissions.device)

        return pred_tensor


class RelationAttention(nn.Module):
    """Biaffine Relation Attention Unit (Chen et al. 2020, Eq. 4-5)."""

    def __init__(self, hidden_dim: int, attention_dim: int | None = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim or hidden_dim

        self.w_ta = nn.Parameter(torch.empty(self.attention_dim, self.hidden_dim))
        self.w_ja = nn.Parameter(torch.empty(self.attention_dim, self.hidden_dim))
        self.b = nn.Parameter(torch.zeros(1, 1, 1, self.attention_dim))
        self.v = nn.Parameter(torch.empty(1, self.attention_dim))

        nn.init.xavier_uniform_(self.w_ta)
        nn.init.xavier_uniform_(self.w_ja)
        nn.init.xavier_uniform_(self.v)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, relation_hidden: torch.Tensor, mask2d: torch.Tensor = None) -> torch.Tensor:
        ta = F.linear(relation_hidden, self.w_ta).unsqueeze(2)  # (B, N, 1, A)
        ja = F.linear(relation_hidden, self.w_ja).unsqueeze(1)  # (B, 1, N, A)
        alpha = torch.tanh(ta + ja + self.b)                   # (B, N, N, A)
        scores = F.linear(alpha, self.v).squeeze(-1)            # (B, N, N)

        if mask2d is not None:
            scores = scores.masked_fill(~mask2d, -1e4)

        return self.softmax(scores)


class JointAOPESDRN(nn.Module):
    """Synchronous Double-channel Recurrent Network for Amharic AOPE."""

    def __init__(
        self,
        model_name: str,
        num_recurrent_steps: int = 2,
        relation_threshold: float = 0.1,
        dropout: float = 0.1,
        use_crf: bool = True,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        h = self.config.hidden_size
        self.T = num_recurrent_steps
        self.beta = relation_threshold
        self.use_crf = use_crf

        bio_w = torch.tensor([1.0, 4.0, 4.0, 8.0, 8.0], dtype=torch.float32)
        self.register_buffer("bio_weights", bio_w)
        self.register_buffer("relation_loss_weights", torch.tensor([0.01, 1.0], dtype=torch.float32))

        self.dropout = nn.Dropout(dropout)
        self.target_head = nn.Linear(h, NUM_BIO_LABELS)

        if self.use_crf:
            self.crf = LinearChainCRF(
                num_tags=NUM_BIO_LABELS,
                bio_weights=self.bio_weights,
                enforce_bio_constraints=True,
            )

        # Target Synchronization
        self.targetSyn_r = nn.Linear(h, h, bias=False)
        self.targetSyn_s = nn.Linear(h, h, bias=False)

        # Relation Synchronization
        self.relationSyn_u = nn.Linear(h, h, bias=False)
        self.relationSyn_s = nn.Linear(h, h, bias=False)

        # Relation attention
        self.relation_attention = RelationAttention(hidden_dim=h, attention_dim=h)

    def decode_tags(self, logits: torch.Tensor, mask: torch.Tensor = None) -> list[list[int]]:
        if self.use_crf:
            pred_tensor = self.crf.decode(logits, mask=mask)
            return pred_tensor.cpu().tolist()
        else:
            return logits.argmax(dim=-1).cpu().tolist()

    def _make_entity_tensor(self, tag_preds, batch_size, seq_len, device, content_mask):
        t_tensor = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.float32, device=device)
        for b in range(batch_size):
            valid_positions = torch.nonzero(content_mask[b], as_tuple=True)[0].tolist()
            if not valid_positions:
                continue
            seq = tag_preds[b]
            a_begin, o_begin = -1, -1
            for pos in valid_positions:
                tag = seq[pos]
                if a_begin != -1 and tag != 2:
                    t_tensor[b, a_begin:pos, a_begin:pos] = 1.0
                    a_begin = -1
                if o_begin != -1 and tag != 4:
                    t_tensor[b, o_begin:pos, o_begin:pos] = 1.0
                    o_begin = -1
                if tag == 1:
                    a_begin = pos
                elif tag == 3:
                    o_begin = pos
            if a_begin != -1:
                t_tensor[b, a_begin:valid_positions[-1] + 1, a_begin:valid_positions[-1] + 1] = 1.0
            if o_begin != -1:
                t_tensor[b, o_begin:valid_positions[-1] + 1, o_begin:valid_positions[-1] + 1] = 1.0
        return t_tensor

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        content_mask: torch.Tensor = None,
        subword_to_word: torch.Tensor = None,
        word_mask: torch.Tensor = None,
    ):
        enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h_sub = enc_out.last_hidden_state
        h_sub = self.dropout(h_sub)

        if subword_to_word is not None:
            # Word-level pooling
            h_s = torch.bmm(subword_to_word.to(h_sub.dtype), h_sub)
            B, N = h_s.shape[:2]
            active_mask = word_mask.bool() if word_mask is not None else (h_s.abs().sum(dim=-1) > 0)
        else:
            h_s = h_sub
            B, N = input_ids.shape
            active_mask = content_mask

        mask2d = active_mask.unsqueeze(1) & active_mask.unsqueeze(2)
        t_tensor = torch.zeros((B, N, N), dtype=torch.float32, device=h_s.device)
        r_tensor = torch.zeros((B, N, N), dtype=torch.float32, device=h_s.device)

        target_emissions = None
        relation_score = None

        for step in range(self.T):
            r_filtered = torch.where(r_tensor >= self.beta, r_tensor, torch.zeros_like(r_tensor)) * mask2d.float()
            target_weighted = torch.bmm(r_filtered, h_s)
            target_div = r_filtered.sum(dim=-1, keepdim=True)
            target_div = target_div + (target_div == 0).float()
            target_r = target_weighted / target_div
            target_hidden = torch.tanh(self.targetSyn_s(h_s) + self.targetSyn_r(target_r))

            t_masked = t_tensor * mask2d.float()
            relation_weighted = torch.bmm(t_masked, h_s)
            relation_div = t_masked.sum(dim=-1, keepdim=True)
            relation_div = relation_div + (relation_div == 0).float()
            relation_u = relation_weighted / relation_div
            relation_hidden = torch.tanh(self.relationSyn_s(h_s) + self.relationSyn_u(relation_u))

            target_emissions = self.target_head(target_hidden)
            relation_score = self.relation_attention(relation_hidden, mask2d=mask2d)

            if step < self.T - 1:
                r_tensor = relation_score
                tag_preds = self.decode_tags(target_emissions, mask=active_mask)
                t_tensor = self._make_entity_tensor(tag_preds, B, N, h_s.device, active_mask)

        return {
            "logits": target_emissions,
            "relation_logits": relation_score,
            "word_mask": active_mask,
        }


# =====================================================================
# 2. Decoding & Relation Helpers
# =====================================================================

def decode_bio_spans(tags: list[str]) -> list[tuple[int, int]]:
    spans = []
    start = None
    for i, tag in enumerate(tags + ["O"]):
        if tag == "B":
            if start is not None:
                spans.append((start, i))
            start = i
        elif tag == "O":
            if start is not None:
                spans.append((start, i))
                start = None
    return spans


def decode_5way_bio_spans(tags: list[str]):
    a_tags = ["B" if t == "B-ASP" else "I" if t == "I-ASP" else "O" for t in tags]
    o_tags = ["B" if t == "B-OPN" else "I" if t == "I-OPN" else "O" for t in tags]
    return decode_bio_spans(a_tags), decode_bio_spans(o_tags)


def correlation_degree(rel_matrix, a_span: tuple, o_span: tuple) -> float:
    a1, a2 = a_span
    o1, o2 = o_span
    a_len = a2 - a1
    o_len = o2 - o1
    if a_len <= 0 or o_len <= 0:
        return 0.0

    a_to_o = sum(rel_matrix[i][j] for i in range(a1, a2) for j in range(o1, o2)) / a_len
    o_to_a = sum(rel_matrix[j][i] for i in range(a1, a2) for j in range(o1, o2)) / o_len
    return 0.5 * (a_to_o + o_to_a)


# =====================================================================
# 3. Prediction Runner
# =====================================================================

def predict_sentence(
    model,
    tokenizer,
    text: str,
    device,
    threshold: float = 0.5,
    max_length: int = 256,
    max_words: int = 128,
):
    tokens = text.strip().split()
    n = len(tokens)
    if n == 0:
        return {"text": text, "tokens": [], "aspects": [], "opinions": [], "pairs": []}

    enc = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    word_ids_raw = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
    ).word_ids()

    L = max_length
    M = max_words
    word_to_subwords = [[] for _ in range(M)]
    for si, wi in enumerate(word_ids_raw):
        if wi is not None and wi < M:
            word_to_subwords[wi].append(si)

    subword_to_word = [[0.0] * L for _ in range(M)]
    word_mask = [False] * M
    for wi in range(min(n, M)):
        sis = word_to_subwords[wi]
        if sis:
            word_mask[wi] = True
            inv_len = 1.0 / len(sis)
            for si in sis:
                subword_to_word[wi][si] = inv_len

    subword_to_word_tensor = torch.tensor([subword_to_word], dtype=torch.float32, device=device)
    word_mask_tensor = torch.tensor([word_mask], dtype=torch.bool, device=device)

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            subword_to_word=subword_to_word_tensor,
            word_mask=word_mask_tensor,
        )

        pred_ids = model.decode_tags(out["logits"], mask=word_mask_tensor)[0]
        rel_scores = out["relation_logits"][0].cpu().tolist()

    word_tags = [ID2LABEL_5WAY.get(pid, "O") for pid in pred_ids[:n]]
    a_spans, o_spans = decode_5way_bio_spans(word_tags)
    word_rel = [row[:n] for row in rel_scores[:n]]

    aspects = [{"span": (s, e), "text": " ".join(tokens[s:e])} for s, e in a_spans]
    opinions = [{"span": (s, e), "text": " ".join(tokens[s:e])} for s, e in o_spans]

    pairs = []
    for a_span in a_spans:
        for o_span in o_spans:
            score = correlation_degree(word_rel, a_span, o_span)
            if score >= threshold:
                pairs.append(
                    {
                        "aspect": " ".join(tokens[a_span[0] : a_span[1]]),
                        "opinion": " ".join(tokens[o_span[0] : o_span[1]]),
                        "score": round(score, 4),
                        "a_span": a_span,
                        "o_span": o_span,
                    }
                )

    return {
        "text": text,
        "tokens": tokens,
        "aspects": aspects,
        "opinions": opinions,
        "pairs": pairs,
    }


def print_prediction_card(pred: dict, expected: dict | None = None, index: int | None = None):
    header = f"=== Example {index} ===" if index is not None else "=== Prediction ==="
    print("\n" + "=" * 65)
    print(header)
    print("=" * 65)
    print(f"Sentence: \"{pred['text']}\"")
    if expected and "domain" in expected:
        print(f"Domain:   [{expected['domain']}]")

    print("\nExtracted Entities:")
    if pred["aspects"]:
        for a in pred["aspects"]:
            print(f"  • Aspect:  \"{a['text']}\" (span: {a['span']})")
    else:
        print("  • Aspect:  (None detected)")

    if pred["opinions"]:
        for o in pred["opinions"]:
            print(f"  • Opinion: \"{o['text']}\" (span: {o['span']})")
    else:
        print("  • Opinion: (None detected)")

    print("\nPredicted Aspect-Opinion Pairs (AOPE):")
    if pred["pairs"]:
        for p in pred["pairs"]:
            print(f"  🔗 [Score: {p['score']:.3f}] \"{p['aspect']}\"  <--->  \"{p['opinion']}\"")
    else:
        print("  (No pairs exceeded the acceptance threshold)")

    if expected:
        print("\nGround Truth Reference:")
        if "expected_aspects" in expected:
            print(f"  ✓ Expected Aspects:  {expected['expected_aspects']}")
        if "expected_opinions" in expected:
            print(f"  ✓ Expected Opinions: {expected['expected_opinions']}")


def main():
    parser = argparse.ArgumentParser(description="Standalone test runner for Amharic SDRN AOPE.")
    parser.add_argument("--checkpoint", default="best_model.pt", help="Path to best_model.pt checkpoint")
    parser.add_argument("--model_name", default="Davlan/afro-xlmr-base", help="Pretrained encoder backbone")
    parser.add_argument("--text", default=None, help="Single Amharic sentence to test")
    parser.add_argument("--file", default=None, help="Input test file (JSONL or TXT)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Pair acceptance correlation threshold (delta-hat)")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive prompt mode")
    parser.add_argument("--device", default=None, help="Device (cuda or cpu)")
    args = parser.parse_args()

    if not TORCH_AVAILABLE:
        print("\n[Error] PyTorch and Transformers must be installed to run inference.")
        print("Run: pip install torch transformers sentencepiece\n")
        sys.exit(1)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading tokenizer: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print("Initializing model architecture...")
    model = JointAOPESDRN(
        model_name=args.model_name,
        num_recurrent_steps=2,
        relation_threshold=0.1,
        use_crf=True,
    )

    # Locate checkpoint
    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        # Fallback search locations
        fallbacks = [
            "best_model.pt",
            "results/stage_aope_sdrn/afroxlmr_gemini_run1/best_model.pt",
            "stage_aope_sdrn_ckpt/best_model.pt",
        ]
        found = False
        for fb in fallbacks:
            if os.path.exists(fb):
                ckpt_path = fb
                found = True
                break
        if not found:
            print(f"\n[Warning] Checkpoint '{args.checkpoint}' not found.")
            print("Running model with untrained base weights for demonstration purposes.\n")

    if os.path.exists(ckpt_path):
        print(f"Loading weights from: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()

    if args.interactive:
        print("\n" + "=" * 70)
        print(" SDRN Amharic Aspect-Opinion Extraction (Interactive)")
        print(" Type 'q' or 'exit' to quit.")
        print("=" * 70)
        while True:
            try:
                line = input("\nአማርኛ ጽሑፍ ያስገቡ (Text) > ").strip()
                if not line or line.lower() in ("q", "exit", "quit"):
                    break
                pred = predict_sentence(model, tokenizer, line, device, threshold=args.threshold)
                print_prediction_card(pred)
            except (KeyboardInterrupt, EOFError):
                break
    elif args.text:
        pred = predict_sentence(model, tokenizer, args.text, device, threshold=args.threshold)
        print_prediction_card(pred)
    else:
        # Default to example file if present
        target_file = args.file
        if target_file is None:
            for candidate in ["example_test_sentences.jsonl", "data/example_test_sentences.jsonl"]:
                if os.path.exists(candidate):
                    target_file = candidate
                    break

        if not target_file or not os.path.exists(target_file):
            print("\nNo input text or file provided, and default example file was not found.")
            print("Try running:")
            print("  python3 predict_standalone.py --checkpoint best_model.pt --text \"የኢንተርኔት ፍጥነቱ በጣም ፈጣን ነው\"")
            return

        print(f"\nRunning predictions on: {target_file}")
        with open(target_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                expected = None
                if line.startswith("{"):
                    try:
                        record = json.loads(line)
                        sentence_text = record.get("text", "")
                        expected = record
                    except json.JSONDecodeError:
                        sentence_text = line
                else:
                    sentence_text = line

                pred = predict_sentence(model, tokenizer, sentence_text, device, threshold=args.threshold)
                print_prediction_card(pred, expected=expected, index=idx)


if __name__ == "__main__":
    main()
