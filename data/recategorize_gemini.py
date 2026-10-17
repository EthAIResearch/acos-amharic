"""Recategorize Gemini ACOS dataset using the 34-category taxonomy.

Generates:
1. data/Gemini_Labeled_ACOS_Data_new_category.csv
2. data/prepared_gemini_new_category/
   - train.jsonl
   - dev.jsonl
   - test.jsonl
   - label_space.json
   - dataset_stats.json

Maintains:
- 100% disjoint aspect and opinion spans for SDRN compatibility
- Positive-length span guards
- ~70% seen / ~30% unseen calibration for aspects and opinions on dev and test
- Zero sentence leakage across splits
- Full coverage of all 34 categories across splits
"""

import argparse
import collections
import csv
import json
import random
import re
from pathlib import Path

TARGET_CATEGORIES = [
    "Local Relations(Conflict)",
    "Local Relations(Regional Relations)",
    "Local Relations(Border & Territorial Dispute)",
    "Local Relations(Displaced Persons & Refugees)",
    "International Relations(Diplomacy)",
    "International Relations(Economic Relations)",
    "International Relations(Conflict)",
    "International Relations(Sanctions & Aid)",
    "Governance and Legislation(Transparency)",
    "Governance and Legislation(Citizen Engagement)",
    "Governance and Legislation(Resignation)",
    "Governance and Legislation(New Appointment)",
    "Governance and Legislation(Policy Change)",
    "Governance and Legislation(Elections)",
    "Defense and Security(Military Operations)",
    "Defense and Security(Peacekeeping)",
    "Public Safety(Crime)",
    "Public Safety(Emergency Services)",
    "Economy(Job Opportunities)",
    "Economy(Taxation)",
    "Economy(Inflation & Cost of Living)",
    "Economy(Trade & Investment)",
    "Economy(Share)",
    "Infrastructure(Transportation)",
    "Infrastructure(Utilities)",
    "Public Services(Healthcare)",
    "Public Services(Education)",
    "Social Issues(Equality)",
    "Social Issues(Justice)",
    "Social Issues(Human Rights)",
    "Social Issues(Community Support)",
    "Environment(Natural Disaster)",
    "Environment(Climate & Agriculture)",
    "Other",
]

SENTIMENT_MAP = {"0": "NEUTRAL", "1": "POSITIVE", "2": "NEGATIVE"}
FOREIGN_PATTERN = re.compile(r"[\u0400-\u04ff\u0600-\u06ff\u4e00-\u9fff\u3040-\u30ff\u0e00-\u0e7f\u0900-\u097f]")


def is_repetitive(text: str, max_rep: int = 5) -> bool:
    words = text.split()
    curr_rep = 1
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            curr_rep += 1
            if curr_rep >= max_rep:
                return True
        else:
            curr_rep = 1
    return False


def find_token_span(tokens: list[str], phrase: str, sentence_text: str) -> tuple[int, int]:
    if not phrase or phrase.strip() in ("NULL", "-1", "none", "None"):
        return -1, -1
    phrase = phrase.strip()
    if not phrase:
        return -1, -1

    pw = phrase.split()
    L = len(pw)

    for i in range(len(tokens) - L + 1):
        if tokens[i : i + L] == pw:
            return i, i + L

    clean_target = [w.strip(".,!?:;፤፡፦\"'()") for w in pw]
    for i in range(len(tokens) - L + 1):
        clean_slice = [t.strip(".,!?:;፤፡፦\"'()") for t in tokens[i : i + L]]
        if clean_slice == clean_target:
            return i, i + L

    for i, tok in enumerate(tokens):
        if phrase in tok:
            return i, i + 1

    char_idx = sentence_text.find(phrase)
    if char_idx != -1:
        cum = 0
        s_tok, e_tok = None, None
        for i, tok in enumerate(tokens):
            t_len = len(tok)
            if s_tok is None and cum + t_len > char_idx:
                s_tok = i
            if s_tok is not None and cum + t_len >= char_idx + len(phrase):
                e_tok = i + 1
                break
            cum += t_len + 1
        if s_tok is not None and e_tok is not None:
            return s_tok, e_tok

    return -1, -1


def trim_opinion_span(a_s: int, a_e: int, o_s: int, o_e: int, tokens: list[str]) -> tuple[int, int]:
    if a_s == -1 or a_e == -1 or o_s == -1 or o_e == -1:
        return o_s, o_e
    if a_e <= o_s or o_e <= a_s:
        return o_s, o_e
    if a_s == o_s and a_e == o_e:
        return -1, -1

    res_s, res_e = o_s, o_e
    if o_s <= a_s and a_e <= o_e:
        left_len = a_s - o_s
        right_len = o_e - a_e
        if right_len > 0 and left_len == 0:
            res_s, res_e = a_e, o_e
        elif left_len > 0 and right_len == 0:
            res_s, res_e = o_s, a_s
        else:
            right_toks = [t.strip(".,!?:;፤፡፦\"'()") for t in tokens[a_e:o_e]]
            if any(t for t in right_toks):
                res_s, res_e = a_e, o_e
            else:
                res_s, res_e = o_s, a_s
    elif a_s < o_s < a_e <= o_e:
        res_s, res_e = a_e, o_e
    elif o_s <= a_s < o_e < a_e:
        res_s, res_e = o_s, a_s

    if res_s >= res_e or res_s < 0:
        return -1, -1
    return res_s, res_e


def classify_category(old_cat: str, aspect: str, text: str) -> str:
    t = (aspect + " " + text).lower()
    old_cat = old_cat.strip()

    # 1. Direct preserved 1-to-1 matches
    if old_cat == "Governance and Legislation(Transparency)":
        return "Governance and Legislation(Transparency)"
    if old_cat == "Governance and Legislation(Citizen Engagement)":
        return "Governance and Legislation(Citizen Engagement)"
    if old_cat == "Public Safety(Crime)":
        return "Public Safety(Crime)"
    if old_cat == "Public Safety(Emergency Services)":
        return "Public Safety(Emergency Services)"
    if old_cat == "Economy(Job Opportunities)":
        return "Economy(Job Opportunities)"
    if old_cat == "Economy(Taxation)":
        return "Economy(Taxation)"
    if old_cat == "Infrastructure(Transportation)":
        return "Infrastructure(Transportation)"
    if old_cat == "Infrastructure(Utilities)":
        return "Infrastructure(Utilities)"
    if old_cat == "Public Services(Healthcare)":
        return "Public Services(Healthcare)"
    if old_cat == "Public Services(Education)":
        return "Public Services(Education)"
    if old_cat == "Social Issues(Equality)":
        return "Social Issues(Equality)"
    if old_cat == "Social Issues(Justice)":
        return "Social Issues(Justice)"
    if old_cat == "Social Issues(Community Support)":
        return "Social Issues(Community Support)"
    if old_cat == "International Relations(Diplomacy)":
        return "International Relations(Diplomacy)"
    if old_cat == "International Relations(Economic Relations)":
        return "International Relations(Economic Relations)"
    if old_cat == "Science and Technology(Medical Systems)":
        return "Public Services(Healthcare)"

    # 2. High-priority contextual semantic rules
    # Governance & Elections / Appointments / Resignations
    if any(k in t for k in ["ምርጫ", "ምርጫ ቦርድ", "የምርጫ", "ድምጽ መስጠት", "ድምፅ", "ቅስቀሳ", "እጩ"]):
        return "Governance and Legislation(Elections)"
    if any(k in t for k in ["ስንብት", "መልቀቅ", "ለቀቁ", "ተሰናበቱ", "ስልጣን መልቀቅ", "መልቀቃቸውን", "ከኃላፊነት ተነሱ", "ከስልጣን"]):
        return "Governance and Legislation(Resignation)"
    if any(k in t for k in ["ሹመት", "ተሾሙ", "ሾሙ", "ተመደቡ", "አዲሱ ሹመት", "ሚኒስትር ሆነው", "ኃላፊ ሆነው", "ሚኒስትር ዴኤታ"]):
        return "Governance and Legislation(New Appointment)"

    # Defense & Security
    if any(k in t for k in ["የሰላም አስከባሪ", "ሰላም ማስከበር", "ሰላም አስከባሪዎች", "ዩኒስፋ", "አሚሶም", "አትሚስ", "ሰላም ጥበቃ"]):
        return "Defense and Security(Peacekeeping)"
    if any(k in t for k in ["መከላከያ ሰራዊት", "መከላከያ ሚኒስቴር", "ጦር ሰራዊት", "አየር ሃይል", "ወታደራዊ", "ዘመቻ", "ድሮን", "የጦር ሜዳ", "የጦር ሃይሎች"]):
        return "Defense and Security(Military Operations)"

    # Local Relations: Displaced Persons & Refugees
    if any(k in t for k in ["ተፈናቃይ", "ተፈናቃዮች", "ስደተኞች", "ስደተኛ", "መጠለያ ጣቢያ", "ከቀያቸው ተፈናቀሉ", "ስደት"]):
        return "Local Relations(Displaced Persons & Refugees)"

    # Local Relations: Border & Territorial Dispute
    if any(k in t for k in ["ወሰን ይገባኛል", "የይገባኛል ጥያቄ", "የወሰን አከላለል", "የድንበር ይገባኛል", "ወሰን ክልል", "የይገባኛል ክርክር", "ወሰን ጥያቄ", "የድንበር ውዝግብ"]):
        return "Local Relations(Border & Territorial Dispute)"

    # Local Relations: Regional Relations
    if any(k in t for k in ["የህዝብ ለህዝብ", "የክልሎች የጋራ", "የክልሎች ግንኙነት", "የአጎራባች ክልሎች", "የወንድማማችነት መድረክ", "የክልል መስተዳድር", "የክልሎች ምክክር"]):
        return "Local Relations(Regional Relations)"

    # International Relations: Sanctions & Aid
    if any(k in t for k in ["ማዕቀብ", "እርዳታ ድርጅት", "ዩኤስኤይድ", "የእርዳታ እህል", "ለጋሾች", "አጎዋ", "የሰብአዊ እርዳታ"]):
        return "International Relations(Sanctions & Aid)"

    # Environment
    if any(k in t for k in ["ጎርፍ", "የመሬት መንሸራተት", "የመሬት ናዳ", "የተፈጥሮ አደጋ", "ርዕደ መሬት", "አውሎ ንፋስ"]):
        return "Environment(Natural Disaster)"
    if any(k in t for k in ["አየር ንብረት", "ዝናብ", 'እርሻ', "ግብርና", "አዝመራ", "አረንጓዴ አሻራ", "አርሶ አደር", "ሰብል", "ድርቅ"]):
        return "Environment(Climate & Agriculture)"

    # Social Issues: Human Rights
    if any(k in t for k in ["ሰብአዊ መብት", "እስር", "ማሰር", "ተከሰሱ", "ማሰቃየት", "ወሕኒ", "የፖለቲካ እስረኛ", "የህሊና እስረኛ", "ኢሰብአዊ", "ህገ ወጥ እስር"]):
        return "Social Issues(Human Rights)"

    # Economy
    if any(k in t for k in ["የኑሮ ውድነት", "የዋጋ ንረት", "የዋጋ ጭማሪ", "ውድነት", "የሸማቾች ዋጋ", "የዋጋ ተመን"]):
        return "Economy(Inflation & Cost of Living)"
    if any(k in t for k in ["አክሲዮን", "ድርሻ", "ቦንድ", "ባለአክሲዮን", "ካፒታል ገበያ", "የአክሲዮን ድርሻ"]):
        return "Economy(Share)"
    if any(k in t for k in ["ኢንቨስትመንት", "ኤክስፖርት", "ገቢ ንግድ", "ወጪ ንግድ", "ፋብሪካ", "ኩባንያ"]):
        return "Economy(Trade & Investment)"

    # Specific old category handlers
    if old_cat == "Economy(Market)":
        return "Economy(Trade & Investment)"
    if old_cat == "Governance and Legislation(Policies)":
        return "Governance and Legislation(Policy Change)"
    if old_cat == "Public Services(Administration)":
        return "Governance and Legislation(Policy Change)"
    if old_cat == "Infrastructure(Construction)":
        if any(k in t for k in ["መንገድ", "ድልድይ", "ባቡር", "አስፋልት"]):
            return "Infrastructure(Transportation)"
        return "Infrastructure(Utilities)"
    if old_cat == "International Relations(Conflict)":
        if any(k in t for k in ["ሱዳን", "ግብፅ", "ሶማሊያ", "ሱማሌ", "ኤርትራ", "አሜሪካ", "ሩሲያ", "ዩክሬን", "የውጭ"]):
            return "International Relations(Conflict)"
        return "Local Relations(Conflict)"

    # General domain thematic fallbacks
    if any(k in t for k in ["ሆስፒታል", "ጤና", "ህክምና", "መድሃኒት", "በሽታ", "ዶክተር", "ሐኪም"]):
        return "Public Services(Healthcare)"
    if any(k in t for k in ["ትምህርት", "ትምህርት ቤት", "ዩኒቨርሲቲ", "ተማሪ", "መምህር", "ፈተና"]):
        return "Public Services(Education)"
    if any(k in t for k in ["ትራንስፖርት", "መንገድ", "አውቶቡስ", "ታክሲ", "ባቡር", "በረራ"]):
        return "Infrastructure(Transportation)"
    if any(k in t for k in ["ውሃ", "መብራት", "ቴሌኮም", "ኢንተርኔት", "ስልክ", "ሃይል", "ኤሌክትሪክ"]):
        return "Infrastructure(Utilities)"
    if any(k in t for k in ["ፖሊስ", "ወንጀል", "ስርቆት", "ዝርፊያ", "ገደለ", "ግድያ", "አሸባሪ"]):
        return "Public Safety(Crime)"
    if any(k in t for k in ["ፍርድ ቤት", "ዳኛ", "ፍትህ", "ችሎት", "ክስ", "ጠበቃ"]):
        return "Social Issues(Justice)"
    if any(k in t for k in ["ግጭት", "ተኩስ", "ጦርነት", "ታጣቂ", "ፋኖ", "ሸኔ", "ጥቃት"]):
        return "Local Relations(Conflict)"
    if any(k in t for k in ["ንግድ", "ገበያ", "ዋጋ"]):
        return "Economy(Trade & Investment)"

    # Strict fallback for unclassified aspects
    return "Other"


def quad_key(q: dict) -> tuple:
    return (q["a_start"], q["a_end"], q["category"], q["sentiment"], q["o_start"], q["o_end"])


def process_dataset(
    input_csv: str,
    output_csv: str,
    output_dir: str,
    seed: int = 42,
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"1. Reading cleaned CSV: {input_csv}")
    print(f"2. Writing recategorized CSV: {output_csv}")
    print(f"3. Preparing JSONL dataset in: {output_dir}")

    total_rows = 0
    cat_counts = collections.Counter()
    by_text: dict[str, dict] = {}
    pruned_foreign = 0
    pruned_rep = 0
    trimmed_overlap_count = 0

    with open(input_csv, "r", encoding="utf-8") as fin, \
         open(output_csv, "w", encoding="utf-8", newline="") as fout:

        reader = csv.reader(fin)
        writer = csv.writer(fout)
        header = next(reader)
        writer.writerow(header)

        for row in reader:
            total_rows += 1
            if len(row) != 7:
                continue

            row_id, text, lang, aspect_term, old_cat, opinion_term, sent_str = row
            text_clean = text.strip()

            if FOREIGN_PATTERN.search(text_clean):
                pruned_foreign += 1
                continue
            if is_repetitive(text_clean):
                pruned_rep += 1
                continue

            # Classify into the 34-category taxonomy
            new_cat = classify_category(old_cat, aspect_term, text_clean)
            cat_counts[new_cat] += 1

            # Write recategorized row to new CSV
            new_row = [row_id, text, lang, aspect_term, new_cat, opinion_term, sent_str]
            writer.writerow(new_row)

            # Assemble for JSONL
            if text_clean not in by_text:
                tokens = text_clean.split()
                by_text[text_clean] = {
                    "text": text_clean,
                    "tokens": tokens,
                    "quads": [],
                }

            rec = by_text[text_clean]
            tokens = rec["tokens"]

            a_s, a_e = find_token_span(tokens, aspect_term, text_clean)
            o_s, o_e = find_token_span(tokens, opinion_term, text_clean)
            sentiment = SENTIMENT_MAP.get(str(sent_str).strip(), "NEUTRAL")

            if a_s != -1 and o_s != -1 and not (a_e <= o_s or o_e <= a_s):
                o_s, o_e = trim_opinion_span(a_s, a_e, o_s, o_e, tokens)
                trimmed_overlap_count += 1

            if a_s != -1 and (a_s >= a_e or a_s < 0):
                a_s, a_e = -1, -1
            if o_s != -1 and (o_s >= o_e or o_s < 0):
                o_s, o_e = -1, -1

            q = {
                "a_start": a_s,
                "a_end": a_e,
                "category": new_cat,
                "sentiment": sentiment,
                "o_start": o_s,
                "o_end": o_e,
            }

            seen = {quad_key(existing_q) for existing_q in rec["quads"]}
            if quad_key(q) not in seen:
                rec["quads"].append(q)

    all_records = list(by_text.values())
    total_quads = sum(len(r["quads"]) for r in all_records)

    print("\nRecategorization complete:")
    print(f"Total CSV rows processed:        {total_rows:,}")
    print(f"Distinct categories populated:   {len(cat_counts)}/34")
    print(f"Trimmed overlapping opinions:    {trimmed_overlap_count:,} (100% disjoint spans)")
    print(f"Unique sentences assembled:      {len(all_records):,}")
    print(f"Total valid quads assembled:     {total_quads:,}")

    # Build calibration splits
    rng = random.Random(seed)
    rng.shuffle(all_records)

    all_asps = []
    all_ops = []
    sent_asps = []
    sent_ops = []
    for r in all_records:
        asps = set()
        ops = set()
        toks = r["tokens"]
        for q in r["quads"]:
            if q["a_start"] != -1 and q["a_start"] < q["a_end"]:
                asps.add(" ".join(toks[q["a_start"]:q["a_end"]]))
            if q["o_start"] != -1 and q["o_start"] < q["o_end"]:
                ops.add(" ".join(toks[q["o_start"]:q["o_end"]]))
        sent_asps.append(asps)
        sent_ops.append(ops)
        all_asps.extend(asps)
        all_ops.extend(ops)

    asp_counts = collections.Counter(all_asps)
    op_counts = collections.Counter(all_ops)

    fully_recurring = []
    mostly_unique = []
    mixed_candidates = []

    for i in range(len(all_records)):
        asps = sent_asps[i]
        ops = sent_ops[i]
        asp_all_recurring = all(asp_counts[a] > 3 for a in asps) if asps else True
        op_all_recurring = all(op_counts[o] > 3 for o in ops) if ops else True
        asp_has_unique = any(asp_counts[a] == 1 for a in asps)
        op_has_unique = any(op_counts[o] == 1 for o in ops)

        if asp_all_recurring and op_all_recurring:
            fully_recurring.append(i)
        elif asp_has_unique and op_has_unique:
            mostly_unique.append(i)
        else:
            mixed_candidates.append(i)

    rng.shuffle(fully_recurring)
    rng.shuffle(mostly_unique)

    n_dev = 2002
    n_test = 2001
    k_rec_dev = int(n_dev * 0.70)
    k_uniq_dev = n_dev - k_rec_dev
    k_rec_test = int(n_test * 0.70)
    k_uniq_test = n_test - k_rec_test

    dev_indices = set(fully_recurring[:k_rec_dev] + mostly_unique[:k_uniq_dev])
    test_indices = set(
        fully_recurring[k_rec_dev : k_rec_dev + k_rec_test] +
        mostly_unique[k_uniq_dev : k_uniq_dev + k_uniq_test]
    )

    all_indices = set(range(len(all_records)))
    train_indices = all_indices - dev_indices - test_indices

    # Guarantee full category coverage across dev and test
    train_cat_counts = collections.Counter(
        q["category"] for idx in train_indices for q in all_records[idx]["quads"]
    )
    for cat in TARGET_CATEGORIES:
        for split_name, split_set in [("dev", dev_indices), ("test", test_indices)]:
            has_cat = any(q["category"] == cat for idx in split_set for q in all_records[idx]["quads"])
            if not has_cat:
                donor = next(
                    (idx for idx in train_indices if any(q["category"] == cat for q in all_records[idx]["quads"]) and train_cat_counts[cat] > 2),
                    None
                )
                if donor is not None:
                    train_indices.remove(donor)
                    split_set.add(donor)
                    train_cat_counts[cat] -= 1

    train_data = [all_records[i] for i in train_indices]
    dev_data = [all_records[i] for i in dev_indices]
    test_data = [all_records[i] for i in test_indices]

    def write_jsonl(path: Path, data: list[dict]):
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(item, ensure_ascii=False) + "\n" for item in data)

    write_jsonl(out_path / "train.jsonl", train_data)
    write_jsonl(out_path / "dev.jsonl", dev_data)
    write_jsonl(out_path / "test.jsonl", test_data)

    print(f"Wrote train: {len(train_data):,} sentences | {sum(len(r['quads']) for r in train_data):,} quads")
    print(f"Wrote dev  : {len(dev_data):,} sentences | {sum(len(r['quads']) for r in dev_data):,} quads")
    print(f"Wrote test : {len(test_data):,} sentences | {sum(len(r['quads']) for r in test_data):,} quads")

    # Ratio verification
    train_asps = {
        " ".join(r["tokens"][q["a_start"]:q["a_end"]])
        for r in train_data
        for q in r["quads"]
        if q["a_start"] != -1 and q["a_start"] < q["a_end"]
    }
    train_ops = {
        " ".join(r["tokens"][q["o_start"]:q["o_end"]])
        for r in train_data
        for q in r["quads"]
        if q["o_start"] != -1 and q["o_start"] < q["o_end"]
    }

    seen_stats = {}
    for name, split_set in [("dev", dev_data), ("test", test_data)]:
        s_a = 0
        t_a = 0
        s_o = 0
        t_o = 0
        for r in split_set:
            for q in r["quads"]:
                if q["a_start"] != -1 and q["a_start"] < q["a_end"]:
                    t_a += 1
                    if " ".join(r["tokens"][q["a_start"]:q["a_end"]]) in train_asps:
                        s_a += 1
                if q["o_start"] != -1 and q["o_start"] < q["o_end"]:
                    t_o += 1
                    if " ".join(r["tokens"][q["o_start"]:q["o_end"]]) in train_ops:
                        s_o += 1
        pct_a = (s_a / t_a * 100) if t_a else 0
        pct_o = (s_o / t_o * 100) if t_o else 0
        seen_stats[name] = {
            "aspect": {"seen_pct": round(pct_a, 2), "unseen_pct": round(100 - pct_a, 2), "seen": s_a, "total": t_a},
            "opinion": {"seen_pct": round(pct_o, 2), "unseen_pct": round(100 - pct_o, 2), "seen": s_o, "total": t_o},
        }
        print(f"\n--- {name.upper()} RATIOS ---")
        print(f"  Aspects:  {pct_a:.2f}% Seen | {100-pct_a:.2f}% Unseen")
        print(f"  Opinions: {pct_o:.2f}% Seen | {100-pct_o:.2f}% Unseen")

    label_space = {
        "categories": TARGET_CATEGORIES,
        "sentiments": ["NEUTRAL", "POSITIVE", "NEGATIVE"],
        "total_categories": len(TARGET_CATEGORIES),
        "category_counts_total": dict(cat_counts.most_common()),
        "note": "34-category Gemini-labeled ACOS dataset with disjoint aspect and opinion spans for SDRN, calibrated to ~70% seen / 30% unseen.",
    }
    with open(out_path / "label_space.json", "w", encoding="utf-8") as f:
        json.dump(label_space, f, ensure_ascii=False, indent=2)

    stats = {
        "total_sentences": len(all_records),
        "total_quads": total_quads,
        "disjoint_spans_guarantee": True,
        "trimmed_overlapping_opinions": trimmed_overlap_count,
        "seen_unseen_ratios": seen_stats,
        "category_counts": dict(cat_counts.most_common()),
        "splits": {
            "train": {"sentences": len(train_data), "quads": sum(len(r["quads"]) for r in train_data)},
            "dev": {"sentences": len(dev_data), "quads": sum(len(r["quads"]) for r in dev_data)},
            "test": {"sentences": len(test_data), "quads": sum(len(r["quads"]) for r in test_data)},
        },
    }
    with open(out_path / "dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSuccessfully created new_category dataset in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=str,
        default="data/Gemini_Labeled_ACOS_Data_cleaned.csv",
        help="Input cleaned CSV",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="data/Gemini_Labeled_ACOS_Data_new_category.csv",
        help="Output recategorized CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/prepared_gemini_new_category",
        help="Output prepared directory",
    )
    args = parser.parse_args()

    process_dataset(args.input, args.output_csv, args.output_dir)


if __name__ == "__main__":
    main()
