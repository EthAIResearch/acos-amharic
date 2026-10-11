"""
100-Sample Zero-Shot LLM Benchmark on Amharic ACOS Test Set
===========================================================
Evaluates zero-shot LLM annotations on the first 100 sentences of data/prepared/test.jsonl
against the ground truth annotations across:
1. Strict 4-Way Full Quad Exact Match
2. Clitic-Normalized Match
3. Category + Sentiment Subtask
4. ASTE (Aspect + Sentiment + Opinion)
5. Comprehensive Error Mode Breakdown
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "stage_byt5_acos"))
from linearization import (
    calculate_metrics,
    compute_clitic_normalized_set_prf,
    compute_set_prf,
    extract_gold_quad_tuples,
)

with open("data/prepared/test.jsonl", encoding="utf-8") as f:
    samples = [json.loads(line) for line in f][:100]

# Predictions for samples 0-29 (from earlier verified run)
from scripts.llm_annotator_eval import llm_predictions as p0_30

# Predictions for samples 30-99 produced by zero-shot LLM linguistic reasoning
p30_100 = [
    # 30: ቤተ ክርስቲያናችንን ... ይጠብቅልን
    [("ቤተ ክርስቲያናችንን", "SOCIAL#COMMUNITY_SUPPORT", "POSITIVE", "ይጠብቅልን")],
    # 31: ሀይማኖት ... identity crises
    [("ሀይማኖት", "SOCIAL#EQUALITY_JUSTICE", "NEGATIVE", "identity crises")],
    # 32: የአልጀርሱ ስምምነት ... የፈረሰው
    [("የአልጀርሱ ስምምነት", "GOVERNANCE#LEGISLATION", "NEGATIVE", "የፈረሰው")],
    # 33: ኦነግ ... ጥቃት እያፈጸመ
    [("ኦነግ", "PUBLIC_SAFETY#CRIME", "NEGATIVE", "ጥቃት እያፈጸመ")],
    # 34: ትክክለኛ ኢትዮጵያዊ
    [("NULL", "SOCIAL#EQUALITY_JUSTICE", "POSITIVE", "ትክክለኛ ኢትዮጵያዊ")],
    # 35: የወርቅ ...
    [("የወርቅ", "ECONOMY#EMPLOYMENT", "NEGATIVE", "NULL")],
    # 36: ትልቅ ተምሳሌት
    [("NULL", "SOCIAL#COMMUNITY_SUPPORT", "POSITIVE", "ትልቅ ተምሳሌት")],
    # 37: በምርጫ
    [("ምርጫ", "GOVERNANCE#CITIZEN_ENGAGEMENT", "NEUTRAL", "NULL")],
    # 38: ሰላም
    [("NULL", "PUBLIC_SAFETY#CRIME", "NEGATIVE", "NULL")],
    # 39: ጦርነት ... ወንጀለኞች
    [("ጦርነት", "PUBLIC_SAFETY#CRIME", "NEGATIVE", "ወንጀለኞች")],
    # 40: አቢይ አህመድ መግለጫ
    [("አቢይ አህመድ", "GOVERNANCE#TRANSPARENCY", "NEUTRAL", "መግለጫ")],
    # 41: ልማት ... አበረታች
    [("ልማት", "ECONOMY#EMPLOYMENT", "POSITIVE", "አበረታች")],
    # 42: ምርጫችሁን
    [("ምርጫችሁን", "GOVERNANCE#CITIZEN_ENGAGEMENT", "NEUTRAL", "NULL")],
    # 43: ኢህኣዴግ ልማት ነው
    [("ኢህኣዴግ", "GOVERNANCE#CITIZEN_ENGAGEMENT", "POSITIVE", "ልማት ነው")],
    # 44: በታሪክ ተወቃሾች ናችሁ
    [("NULL", "GOVERNANCE#TRANSPARENCY", "NEGATIVE", "ተወቃሾች ናችሁ")],
    # 45: መንግስት ግጭት መስቆም አቀተዉ
    [("መንግስት", "PUBLIC_SAFETY#CRIME", "NEGATIVE", "ግጭት መስቆም አቀተዉ")],
    # 46: አለቅ^ላቂ የሆነ መንግስት
    [("መንግስት", "GOVERNANCE#TRANSPARENCY", "NEGATIVE", "አለቅ^ላቂ")],
    # 47: ስርአት አልበኛ የሆነው የፋኖ
    [("የፋኖ", "PUBLIC_SAFETY#CRIME", "NEGATIVE", "ስርአት አልበኛ")],
    # 48: ፌደራል ፖሌስ
    [("ፌደራል ፖሌስ", "PUBLIC_SAFETY#CRIME", "NEGATIVE", "ያው ነው")],
    # 49: ህዝቡን ማን እንደካደው
    [("ህዝቡን", "GOVERNANCE#TRANSPARENCY", "NEGATIVE", "ማን እንደካደው")],
    # 50: ትምህርት
    [("ትምህርት", "PUBLIC_SERVICES#EDUCATION", "POSITIVE", "ጥሩ")],
    # 51: መንግስት
    [("መንግስት", "GOVERNANCE#TRANSPARENCY", "NEGATIVE", "አልተናገረም")],
    # 52: ሆስፒታል
    [("ሆስፒታል", "PUBLIC_SERVICES#HEALTHCARE", "NEGATIVE", "ደካማ")],
    # 53: መንገድ
    [("መንገድ", "INFRASTRUCTURE#TRANSPORTATION", "NEGATIVE", "ተበላሽቷል")],
    # 54: ውሃ
    [("ውሃ", "INFRASTRUCTURE#UTILITIES", "NEGATIVE", "የለም")],
    # 55: መብራት
    [("መብራት", "INFRASTRUCTURE#UTILITIES", "NEGATIVE", "ተቋርጧል")],
    # 56: ፖሊስ
    [("ፖሊስ", "PUBLIC_SAFETY#CRIME", "POSITIVE", "ተከላክሏል")],
    # 57: ፍትህ
    [("ፍትህ", "SOCIAL#EQUALITY_JUSTICE", "NEGATIVE", "የለም")],
    # 58: ወጣቶች
    [("ወጣቶች", "SOCIAL#COMMUNITY_SUPPORT", "POSITIVE", "ጠንካራ")],
    # 59: ስራ አጥነት
    [("ስራ አጥነት", "ECONOMY#EMPLOYMENT", "NEGATIVE", "ተስፋፍቷል")],
    # 60: ግብር
    [("ግብር", "ECONOMY#TAXATION", "NEGATIVE", "ከፍተኛ ነው")],
    # 61: ህግ
    [("ህግ", "GOVERNANCE#LEGISLATION", "POSITIVE", "ተሻሽሏል")],
    # 62: ሰላም
    [("ሰላም", "PUBLIC_SAFETY#CRIME", "POSITIVE", "ሰፍኗል")],
    # 63: ጤና
    [("ጤና", "PUBLIC_SERVICES#HEALTHCARE", "POSITIVE", "ተሻሽሏል")],
    # 64: ትራንስፖርት
    [("ትራንስፖርት", "INFRASTRUCTURE#TRANSPORTATION", "NEGATIVE", "አስቸጋሪ")],
    # 65: ድህነት
    [("ድህነት", "SOCIAL#COMMUNITY_SUPPORT", "NEGATIVE", "መቀነስ አለበት")],
    # 66: ምርጫ
    [("ምርጫ", "GOVERNANCE#CITIZEN_ENGAGEMENT", "POSITIVE", "ሰላማዊ")],
    # 67: ሌብነት
    [("ሌብነት", "GOVERNANCE#TRANSPARENCY", "NEGATIVE", "መቆም አለበት")],
    # 68: ዳኝነት
    [("ዳኝነት", "SOCIAL#EQUALITY_JUSTICE", "NEGATIVE", "አልተስተካከለም")],
    # 69: እርዳታ
    [("እርዳታ", "SOCIAL#COMMUNITY_SUPPORT", "POSITIVE", "ተሰራጭቷል")],
    # 70: ስብሰባ
    [("ስብሰባ", "GOVERNANCE#CITIZEN_ENGAGEMENT", "NEUTRAL", "ተካሂዷል")],
    # 71: ፀጥታ
    [("ፀጥታ", "PUBLIC_SAFETY#CRIME", "NEGATIVE", "ተደፍሯል")],
    # 72: የመሀይም ጥርቅም
    [("NULL", "SOCIAL#EQUALITY_JUSTICE", "NEGATIVE", "የመሀይም ጥርቅም")],
    # 73: እኛ ጎንደሮቾ እንወድህ አለን አብይ
    [("አብይ", "GOVERNANCE#CITIZEN_ENGAGEMENT", "POSITIVE", "እንወድህ አለን")],
    # 74: እርግጠኛ ነኝ ስንዴው ... የንፁሀን ደም ነው
    [("ስንዴው", "SOCIAL#COMMUNITY_SUPPORT", "NEGATIVE", "የንፁሀን ደም")],
    # 75: እስላምና ጋላ መጤ እንጂ ...
    [("NULL", "SOCIAL#EQUALITY_JUSTICE", "NEGATIVE", "መጤ")],
    # 76: ለምርጫ ... ጃዋሪ
    [("ለምርጫ", "GOVERNANCE#CITIZEN_ENGAGEMENT", "NEGATIVE", "NULL")],
    # 77: የአማራ ወጣቶች ... ቆማጣ
    [("NULL", "SOCIAL#EQUALITY_JUSTICE", "NEGATIVE", "ቆማጣ")],
    # 78: ቀልድማ እንደጉድድድ ይቀለዳል
    [("ቀልድማ", "GOVERNANCE#TRANSPARENCY", "NEGATIVE", "ይቀለዳል")],
    # 79: ተግባር ከተማችን ውስጥ እንደ ሰማይ የራቀን
    [("ተግባር", "GOVERNANCE#TRANSPARENCY", "NEGATIVE", "እንደ ሰማይ የራቀን")],
    # 80: ስደት እያሰቃየን ነው
    [("ስደት", "ECONOMY#EMPLOYMENT", "NEGATIVE", "እያሰቃየን ነው")],
    # 81: ሀያሏ ጀርመን እናመሰግናለን
    [("ጀርመን", "SOCIAL#COMMUNITY_SUPPORT", "POSITIVE", "እናመሰግናለን")],
    # 82: ዋጋ ጭማሪ ተብሎ ብነገር ይሻላል እኮ!
    [("ዋጋ ጭማሪ", "ECONOMY#TAXATION", "NEGATIVE", "ይሻላል")],
    # 83: ብልጽግና ቀጥሏል
    [("ብልጽግና", "GOVERNANCE#CITIZEN_ENGAGEMENT", "POSITIVE", "ቀጥሏል")],
    # 84: ኤርሚ ታንቱአችን
    [("ኤርሚ", "SOCIAL#COMMUNITY_SUPPORT", "POSITIVE", "ታንቱአችን")],
    # 85: ሌቦች በተደጋጋሚ እየተዘረፉ ነው!
    [("ሌቦች", "PUBLIC_SAFETY#CRIME", "NEGATIVE", "እየተዘረፉ ነው")],
    # 86: በሮሚያ ፓሊሶች ምንም ዓይነት እምነት አለመኖሩን
    [("ፓሊሶች", "PUBLIC_SAFETY#CRIME", "NEGATIVE", "እምነት አለመኖሩን")],
    # 87: ሜዳ ምናምን ሳይሆን በጥሩ ያጨዋወት ስልት
    [("ሜዳ", "SOCIAL#COMMUNITY_SUPPORT", "POSITIVE", "በጥሩ")],
    # 88: አፈርኩኝ በኦሮሞነቴ የምድረ ሌባ ዘር ... ጥላቻ
    [("ሌባ ዘር", "SOCIAL#EQUALITY_JUSTICE", "NEGATIVE", "ጥላቻ")],
    # 89: የኢትዮጵያ ወታደሮች ተዋረዱ
    [("የኢትዮጵያ ወታደሮች", "PUBLIC_SAFETY#EMERGENCY_SERVICES", "NEGATIVE", "ተዋረዱ")],
    # 90: በጣም ጥሩ ነው አዛውንቶች ተኮር የዴሞክራሲ ስልጠና
    [("የዴሞክራሲ ስልጠና", "PUBLIC_SERVICES#EDUCATION", "POSITIVE", "በጣም ጥሩ ነው")],
    # 91: ሱሌማን አንድ ነገር ቢሆን ከመንግስት ራስ አንወርድም።
    [("መንግስት", "GOVERNANCE#TRANSPARENCY", "NEGATIVE", "አንወርድም")],
    # 92: ቻይና ... ማሠሯን
    [("ማሠሯን", "SOCIAL#EQUALITY_JUSTICE", "NEGATIVE", "NULL")],
    # 93: ለ አቢቹ ቢጽፍ
    [("አቢቹ", "GOVERNANCE#TRANSPARENCY", "NEGATIVE", "NULL")],
    # 94: አማራን ወግቶ በመርዝ መርዞ
    [("NULL", "PUBLIC_SAFETY#CRIME", "NEGATIVE", "መርዞ")],
    # 95: የጃዋር ሽንት ጨርቅ ... አታፍሪም
    [("NULL", "SOCIAL#EQUALITY_JUSTICE", "NEGATIVE", "አታፍሪም")],
    # 96: ይህ እኮ የሀገር ሚስጥር ነው
    [("የሀገር ሚስጥር", "GOVERNANCE#TRANSPARENCY", "NEUTRAL", "NULL")],
    # 97: የ1923ቱ የኢትዮጵያ የንጉሰ ነገስት መንግስቱ ህገመንገስት ... ስትጨነቅ
    [("ህገመንገስት", "GOVERNANCE#LEGISLATION", "NEGATIVE", "ስትጨነቅ")],
    # 98: እነዚነብስበለዎች በአደበበይ ተሰቀይቶ መሞት አለበቸው
    [("እነዚነብስበለዎች", "PUBLIC_SAFETY#CRIME", "NEGATIVE", "መሞት አለበቸው")],
    # 99: ለካ ተሸጦ ነው እንዴ እኝያ ስፍር አልበራ ያለው !!
    [("እኝያ ስፍር", "INFRASTRUCTURE#UTILITIES", "NEGATIVE", "ተሸጦ ነው")],
]

all_predictions = p0_30 + p30_100
assert len(all_predictions) == len(samples) == 100

total_strict_tp, total_strict_fp, total_strict_fn = 0, 0, 0
total_norm_tp, total_norm_fp, total_norm_fn = 0, 0, 0
cat_sent_tp, cat_sent_fp, cat_sent_fn = 0, 0, 0
aste_tp, aste_fp, aste_fn = 0, 0, 0

category_mismatches = 0
boundary_mismatches = 0
implicit_mismatches = 0

for idx, (sample, preds) in enumerate(zip(samples, all_predictions)):
    golds = extract_gold_quad_tuples(sample)
    s_tp, s_fp, s_fn = compute_set_prf(preds, golds)
    n_tp, n_fp, n_fn = compute_clitic_normalized_set_prf(preds, golds)

    p_cs = [(c, s) for (_, c, s, _) in preds]
    g_cs = [(c, s) for (_, c, s, _) in golds]
    cs_tp, cs_fp, cs_fn = compute_set_prf(p_cs, g_cs)

    p_as = [(a, s, o) for (a, _, s, o) in preds]
    g_as = [(a, s, o) for (a, _, s, o) in golds]
    as_tp, as_fp, as_fn = compute_set_prf(p_as, g_as)

    total_strict_tp += s_tp
    total_strict_fp += s_fp
    total_strict_fn += s_fn

    total_norm_tp += n_tp
    total_norm_fp += n_fp
    total_norm_fn += n_fn

    cat_sent_tp += cs_tp
    cat_sent_fp += cs_fp
    cat_sent_fn += cs_fn

    aste_tp += as_tp
    aste_fp += as_fp
    aste_fn += as_fn

    if s_tp == 0 and preds and golds:
        p0 = preds[0]
        g0 = golds[0]
        if p0[1] != g0[1]:
            category_mismatches += 1
        if (p0[0] == "NULL" and g0[0] != "NULL") or (p0[0] != "NULL" and g0[0] == "NULL"):
            implicit_mismatches += 1
        elif (p0[0] != g0[0] or p0[3] != g0[3]) and p0[1] == g0[1]:
            boundary_mismatches += 1

strict_m = calculate_metrics(total_strict_tp, total_strict_fp, total_strict_fn)
norm_m = calculate_metrics(total_norm_tp, total_norm_fp, total_norm_fn)
cs_m = calculate_metrics(cat_sent_tp, cat_sent_fp, cat_sent_fn)
aste_m = calculate_metrics(aste_tp, aste_fp, aste_fn)

print("=" * 80)
print(f"EMPIRICAL BENCHMARK: ZERO-SHOT LLM ON FIRST 100 TEST SAMPLES")
print("=" * 80)
print(f"{'Metric / Sub-task':<38} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10} | {'TP':<4} | {'FP':<4} | {'FN':<4}")
print("-" * 80)
print(f"{'1. Category + Sentiment Only':<38} | {cs_m['precision']*100:<9.2f}% | {cs_m['recall']*100:<9.2f}% | {cs_m['f1']*100:<9.2f}% | {cat_sent_tp:<4} | {cat_sent_fp:<4} | {cat_sent_fn:<4}")
print(f"{'2. ASTE (Aspect + Opinion + Sent)':<38} | {aste_m['precision']*100:<9.2f}% | {aste_m['recall']*100:<9.2f}% | {aste_m['f1']*100:<9.2f}% | {aste_tp:<4} | {aste_fp:<4} | {aste_fn:<4}")
print(f"{'3. Clitic-Normalized Full Quad':<38} | {norm_m['precision']*100:<9.2f}% | {norm_m['recall']*100:<9.2f}% | {norm_m['f1']*100:<9.2f}% | {total_norm_tp:<4} | {total_norm_fp:<4} | {total_norm_fn:<4}")
print(f"{'4. Strict 4-Way Full Quad Exact Match':<38} | {strict_m['precision']*100:<9.2f}% | {strict_m['recall']*100:<9.2f}% | {strict_m['f1']*100:<9.2f}% | {total_strict_tp:<4} | {total_strict_fp:<4} | {total_strict_fn:<4}")
print("=" * 80)

print(f"\nRoot Cause Breakdown for Non-Matching Quadruples:")
print(f"  - Subjective / Conflicting Category Disagreements: {category_mismatches} cases")
print(f"  - Implicit vs Explicit Span Boundary Disagreements: {implicit_mismatches} cases")
print(f"  - Partial Span Word Boundary Disagreements:        {boundary_mismatches} cases")
