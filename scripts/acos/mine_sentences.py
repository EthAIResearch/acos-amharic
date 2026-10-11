"""Mine candidate multi-aspect evaluative/opinion sentences from the Amharic book corpus
specifically prioritizing multi-quadruple sentences across all 23 ACOS categories.
"""

import gzip
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

# All 23 categories and their flexible keyword anchors in Amharic
CATEGORY_LEXICONS: Dict[str, List[str]] = {
    # 1. Economy
    "ECONOMY#COMMUNITY_SUPPORT": [
        "እርዳታ", "ድጋፍ", "ህብረት", "መረዳዳት", "እቁብ", "ልገሳ", "ፈንድ", "አጋርነት",
        "የገንዘብ", "መዋጮ", "ቁጠባ", "ብድር", "የጋራ"
    ],
    "ECONOMY#EMPLOYMENT": [
        "ስራ", "ሰራተኛ", "ስራ እድል", "ስራ አጥነት", "ደመወዝ", "ቅጥር", "ስራ ገበያ",
        "ጉልበት", "ሙያ", "አሰሪ", "ተቀጣሪ", "ድርጅት", "ደሞዝ", "ገቢ", "አበል"
    ],
    "ECONOMY#TAXATION": [
        "ግብር", "ታክስ", "ቀረጥ", "ገቢ ግብር", "አስገባሪ", "ግብር ከፋይ", "ጉምሩክ",
        "እሴት ታክስ", "ቀረጥ ክፍያ", "አስራት", "እጅ መንሻ"
    ],
    "ECONOMY#UTILITIES": [
        "ታሪፍ", "የውሃ ክፍያ", "የኤሌክትሪክ ክፍያ", "የመብራት ክፍያ", "የስልክ ክፍያ", "የውሃ ቢል",
        "የመብራት ቢል", "የሃይል ክፍያ", "የመብራት ዋጋ", "የውሃ ዋጋ", "የታሪፍ ጭማሪ", "የኢነርጂ ወጪ",
        "የቴሌ ክፍያ", "የቆጣሪ ክፍያ", "የአገልግሎት ክፍያ", "የቴሌኮም ክፍያ"
    ],

    # 2. Governance
    "GOVERNANCE#CITIZEN_ENGAGEMENT": [
        "ተሳትፎ", "ህዝባዊ ተሳትፎ", "ድምፅ", "ምርጫ", "አስተያየት", "ውይይት", "መድረክ",
        "ምክክር", "አቤቱታ", "የህዝብ ድምጽ", "ስብሰባ", "የህዝብ ፈቃድ"
    ],
    "GOVERNANCE#COMMUNITY_SUPPORT": [
        "የመንግስት ድጋፍ", "ድጎማ", "ማህበራዊ ዋስትና", "ሴፍቲኔት", "የችግረኞች ድጋፍ",
        "የመንግስት እርዳታ", "የመንግስት እንክብካቤ", "የድጋፍ ፖሊሲ", "የመንግስት በጀት",
        "የድጋፍ መርሃ-ግብር", "የመንግስት ፕሮጀክት"
    ],
    "GOVERNANCE#LEGISLATION": [
        "ህግ", "አዋጅ", "ደንብ", "ምክር ቤት", "ህገ መንግስት", "የህግ ማሻሻያ", "መመሪያ",
        "ህጋዊ", "ፍርድ", "አንቀጽ", "ስርዓት", "ትእዛዝ", "ደንቦች"
    ],
    "GOVERNANCE#TRANSPARENCY": [
        "ሙስና", "ግልፅነት", "ተጠያቂነት", "ስርቆት", "ምርመራ", "አሰራር", "ብክነት",
        "ሚስጥር", "ኦዲት", "ማጭበርበር", "ታማኝነት", "ሀብት ማሳወቅ", "ፍትሃዊ አሰራር"
    ],

    # 3. Healthcare
    "HEALTHCARE#GENERAL": [
        "ህክምና", "ሆስፒታል", "ጤና ጣቢያ", "ሐኪም", "መድኃኒት", "ህመም", "በሽታ",
        "ፈውስ", "ክሊኒክ", "የጤና ጥበቃ", "ህመምተኛ", "መታከም", "የጤና አገልግሎት"
    ],

    # 4. Infrastructure
    "INFRASTRUCTURE#TRANSPORTATION": [
        "መንገድ", "አስፋልት", "አውቶቡስ", "ባቡር", "ትራንስፖርት", "መኪና", "ድልድይ",
        "ታክሲ", "ትራፊክ", "የባቡር መስመር", "የአየር መንገድ", "የመንገድ ግንባታ"
    ],
    "INFRASTRUCTURE#UTILITIES": [
        "የኤሌክትሪክ መስመር", "የውሃ ቧንቧ", "ቴሌ", "ኔትወርክ", "የመብራት ኃይል",
        "የስልክ መስመር", "የውሃ ማስተላለፊያ", "ግድብ", "ማከፋፈያ", "የሃይል ማመንጫ",
        "የቴሌኮም", "የመብራት ቆጣሪ", "የውሃ መስመር", "የውሃ ቱቦ"
    ],

    # 5. Public Safety
    "PUBLIC_SAFETY#CRIME": [
        "ወንጀል", "ሌብነት", "ሌባ", "ሽፍታ", "ዝርፊያ", "ግድያ", "ሁከት", "ጥቃት",
        "ሽብር", "አደጋ", "ዘረፋ", "ህገ ወጥ", "አስገድዶ መድፈር", "ጠብ"
    ],
    "PUBLIC_SAFETY#CRIME_SERVICES": [
        "ፖሊስ", "ፍርድ ቤት", "ክስ", "እስር ቤት", "ዳኝነት", "ፖሊስ ጣቢያ", "መርማሪ",
        "ጠባቂ", "ፍርደኛ", "የፀጥታ ኃይል", "ዳኛ", "ጠበቃ", "እስረኛ", "ቅጣት"
    ],
    "PUBLIC_SAFETY#EMERGENCY_SERVICES": [
        "እሳት አደጋ", "አደጋ መከላከል", "አድን ቡድን", "የአደጋ ጊዜ", "የነፍስ አድን",
        "አደጋ መከላከያ", "የድንገተኛ እርዳታ", "አደጋ ጊዜ", "ማዳን"
    ],

    # 6. Public Services
    "PUBLIC_SERVICES#COMMUNITY_SUPPORT": [
        "የማህበረሰብ ማዕከል", "የወጣቶች ማዕከል", "የህዝብ መጠለያ", "የአረጋውያን መርጃ",
        "የህዝብ አገልግሎት", "የማህበረሰብ አገልግሎት", "የህዝብ መዝናኛ", "የህዝብ አዳራሽ",
        "የህዝብ ቤተ-መጽሐፍት", "የማህበረሰብ ልማት", "የጋራ አገልግሎት"
    ],
    "PUBLIC_SERVICES#EDUCATION": [
        "ትምህርት", "ትምህርት ቤት", "ዩኒቨርሲቲ", "መምህራን", "ተማሪ", "ስርዓተ ትምህርት",
        "ክፍል", "ማስተማር", "ትምህርት ጥራት", "ኮሌጅ", "መምህር", "ፈተና"
    ],
    "PUBLIC_SERVICES#EMERGENCY_SERVICES": [
        "አምቡላንስ", "ድንገተኛ አደጋ", "ፈጣን ህክምና", "የድንገተኛ ጥሪ",
        "የአምቡላንስ አገልግሎት", "የድንገተኛ ህክምና", "ድንገተኛ ክፍል"
    ],
    "PUBLIC_SERVICES#HEALTHCARE": [
        "ክትባት", "የእናቶች ጤና", "ነፃ ህክምና", "የህዝብ ጤና", "ጤና ኬላ",
        "የህዝብ ክሊኒክ", "የህጻናት ጤና", "የወረርሽኝ መከላከያ", "የመንግስት ሆስፒታል",
        "የጤና ጣቢያ", "የጤና አገልግሎት", "ጤና ጥበቃ"
    ],
    "PUBLIC_SERVICES#INFRASTRUCTURE": [
        "የጎርፍ መፋሰሻ", "ድልድይ ጥገና", "የመንገድ መብራት", "የከተማ ጽዳት",
        "የህዝብ መጸዳጃ", "የመንገድ ዳር ልማት", "የፍሳሽ ማስወገጃ", "የህዝብ መናፈሻ",
        "የመንገድ ዳር", "የፍሳሽ", "የቆሻሻ አወጋገድ", "የመንገድ ጥገና", "መሰረተ ልማት"
    ],
    "PUBLIC_SERVICES#UTILITIES": [
        "ንጹህ የመጠጥ ውሃ", "መብራት አቅርቦት", "የውሃ አቅርቦት", "የኤሌክትሪክ አቅርቦት",
        "የህዝብ ውሃ", "የመብራት መቆራረጥ", "የውሃ እጥረት", "የመብራት አገልግሎት", "ውሃ ማጣት",
        "የመጠጥ ውሃ", "የኤሌክትሪክ ኃይል"
    ],

    # 7. Social
    "SOCIAL#COMMUNITY_SUPPORT": [
        "ማህበራዊ መረዳዳት", "እድር", "ጎረቤት", "አብሮነት", "ወገንተኝነት",
        "ማህበራዊ ትስስር", "የጎረቤት ድጋፍ", "ዘመድ", "ማህበረሰብ", "ማህበር"
    ],
    "SOCIAL#EDUCATION": [
        "ማንበብ", "ንቃተ ህሊና", "እውቀት", "ማህበራዊ ግንዛቤ", "መማር",
        "የንባብ ባህል", "ጥበብ", "የህሊና እድገት", "ማወቅ", "ግንዛቤ"
    ],
    "SOCIAL#EQUALITY_JUSTICE": [
        "እኩልነት", "ፍትህ", "አድሎ", "መብት", "የሰብአዊ መብት", "ነጻነት",
        "ፍትሃዊነት", "እኩል እድል", "ጭቆና", "ፍትህ ማጣት", "እኩል መብት"
    ],
}

# Connectors signaling multi-aspect, contrastive, or compound evaluations
CONNECTORS = [
    "ነገር ግን", "ግን", "ሆኖም", "ቢሆንም", "በተቃራኒው", "እንዲሁም", "በተጨማሪም",
    "በአንድ በኩል", "በሌላ በኩል", "ከመሆኑም በላይ", "ሳይሆን", "ከ... ይልቅ", "በተለይም",
    "እንጂ", "ቢሆን", "እያለ", "ከዚያም"
]

SENTIMENT_CUES = [
    "ጥሩ", "መጥፎ", "አስደናቂ", "አስከፊ", "ደካማ", "ጠንካራ", "ውብ", "አስቀያሚ",
    "ብቁ", "የማይረባ", "አስፈላጊ", "ጠቃሚ", "ጎጂ", "ፍትሃዊ", "አድሎአዊ",
    "የተሻለ", "ከፍተኛ", "ዝቅተኛ", "ውድቀት", "ስኬት", "አሳዛኝ", "አስደሳች",
    "ብልሹ", "ንጹህ", "አመርቂ", "አጥጋቢ", "የሚያሳዝን", "የሚያስደስት",
    "ችግር", "እጥረት", "ስህተት", "ጥፋት", "ጥቅም", "ጉዳት", "ምስጋና", "ነቀፋ"
]


def is_ethiopic_clean(text: str, min_ratio: float = 0.80) -> bool:
    if not text:
        return False
    letters = [c for c in text if not c.isspace() and c not in '.,!?:;፤፡፦()[]"\'0123456789፩፪፫፬፭፮፯፰፱፲']
    if not letters:
        return False
    eth_letters = sum(1 for c in letters if 0x1200 <= ord(c) <= 0x137F)
    return (eth_letters / len(letters)) >= min_ratio


def extract_sentences_from_text(text: str) -> List[str]:
    raw_sents = re.split(r'[።\?\!\n\r]+', text)
    valid = []
    for s in raw_sents:
        cleaned = re.sub(r'\s+', ' ', s).strip()
        words = cleaned.split()
        if 8 <= len(words) <= 40 and is_ethiopic_clean(cleaned):
            valid.append(cleaned)
    return valid


def match_categories(sentence: str) -> List[str]:
    matched = set()
    for cat, keywords in CATEGORY_LEXICONS.items():
        for kw in keywords:
            if kw in sentence:
                matched.add(cat)
                break

    # Compound matching for rare categories
    util_words = ["ውሃ", "መብራት", "ስልክ", "ኤሌክትሪክ", "ኢንተርኔት", "ቴሌ"]
    cost_words = ["ክፍያ", "ዋጋ", "ወጪ", "ብር", "ታሪፍ", "ሂሳብ", "ቢል"]
    pub_util_words = ["አቅርቦት", "እጥረት", "አገልግሎት", "መቆራረጥ", "ማጣት", "ቧንቧ", "መጠጥ ውሃ", "ንጹሕ ውሃ", "ንጹህ ውሃ"]
    gov_words = ["መንግስት", "አስተዳደር", "ባለስልጣን", "ቀበሌ", "ወረዳ", "የህዝብ"]
    aid_words = ["ድጋፍ", "እርዳታ", "እንክብካቤ", "ጥበቃ", "መርዳት", "ማገዝ", "ድጎማ", "ዋስትና", "መርጃ"]

    if any(u in sentence for u in util_words) and any(cw in sentence for cw in cost_words):
        matched.add("ECONOMY#UTILITIES")
    if any(u in sentence for u in ["ውሃ", "መብራት", "ኤሌክትሪክ"]) and any(pu in sentence for pu in pub_util_words):
        matched.add("PUBLIC_SERVICES#UTILITIES")
    if any(g in sentence for g in gov_words) and any(a in sentence for a in aid_words):
        matched.add("GOVERNANCE#COMMUNITY_SUPPORT")
    if any(hz in sentence for hz in ["ሕዝብ", "ህዝብ", "ማህበረሰብ"]) and any(k in sentence for k in ["አገልግሎት", "ድጋፍ", "መርዳት", "መርጃ", "ማዕከል", "መጠለያ", "ልማት"]):
        matched.add("PUBLIC_SERVICES#COMMUNITY_SUPPORT")

    return sorted(matched)


def count_evaluative_cues(sentence: str) -> int:
    return sum(1 for cue in SENTIMENT_CUES if cue in sentence)


def has_connectors(sentence: str) -> bool:
    return any(c in sentence for c in CONNECTORS)


def mine_candidate_sentences(
    corpus_path: Path,
    output_path: Path,
    target_multi_quad: int = 2500,
    target_per_category: int = 150,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    multi_category_candidates: List[Dict[str, any]] = []
    single_category_candidates: Dict[str, List[Dict[str, any]]] = defaultdict(list)
    seen_texts: Set[str] = set()

    print(f"[Miner] Opening corpus: {corpus_path}")
    doc_count = 0

    with gzip.open(corpus_path, "rt", encoding="utf-8") as f:
        for line in f:
            doc_count += 1
            doc = json.loads(line)
            doc_url = doc.get("url", f"doc_{doc_count}")
            text = doc.get("text", "")

            sentences = extract_sentences_from_text(text)
            for s in sentences:
                if s in seen_texts:
                    continue

                matched_cats = match_categories(s)
                if not matched_cats:
                    continue

                cues_count = count_evaluative_cues(s)
                has_conn = has_connectors(s)

                record = {
                    "text": s,
                    "matched_categories": matched_cats,
                    "num_categories": len(matched_cats),
                    "cues_count": cues_count,
                    "has_connectors": has_conn,
                    "source_doc": doc_url,
                }

                # PRIORITY 1: Multi-category sentences (naturally yield 2+ quads)
                if len(matched_cats) >= 2 or (len(matched_cats) == 1 and has_conn and cues_count >= 1):
                    if len(multi_category_candidates) < target_multi_quad * 2:
                        multi_category_candidates.append(record)
                        seen_texts.add(s)
                else:
                    # Single category candidates to ensure full coverage of rare categories
                    cat = matched_cats[0]
                    if len(single_category_candidates[cat]) < target_per_category:
                        single_category_candidates[cat].append(record)
                        seen_texts.add(s)

    print(f"\n[Miner] Finished scanning all {doc_count} books!")
    print(f"[Miner] Mined {len(multi_category_candidates)} multi-aspect/compound candidate sentences!")

    # Combine candidates prioritizing multi-category sentences (targeting ~80%+ multi-quads)
    final_candidates = []
    seen_final = set()

    # Add multi-aspect candidates first
    for item in multi_category_candidates[:target_multi_quad]:
        final_candidates.append(item)
        seen_final.add(item["text"])

    # Supplement rare categories with single-aspect sentences
    for cat in sorted(CATEGORY_LEXICONS):
        items = single_category_candidates[cat]
        added = 0
        for item in items:
            if item["text"] not in seen_final:
                final_candidates.append(item)
                seen_final.add(item["text"])
                added += 1
                if added >= 40:
                    break

    print(f"[Miner] Total candidates selected for annotation: {len(final_candidates)}")
    multi_count = sum(1 for c in final_candidates if c.get("num_categories", 1) >= 2 or c.get("has_connectors"))
    print(f"  Multi-aspect / Compound sentences: {multi_count} ({multi_count/len(final_candidates):.1%})")

    with open(output_path, "w", encoding="utf-8") as f:
        for item in final_candidates:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[Miner] Saved candidate dataset to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Mine candidate multi-aspect sentences")
    parser.add_argument("--corpus", type=str, default="data/corpus/amharic_corpus_mc4.jsonl.gz")
    parser.add_argument("--out", type=str, default="data/book_acos/raw_candidates.jsonl")
    parser.add_argument("--target-multi", type=int, default=2500)
    args = parser.parse_args()

    mine_candidate_sentences(
        corpus_path=Path(args.corpus),
        output_path=Path(args.out),
        target_multi_quad=args.target_multi,
    )
