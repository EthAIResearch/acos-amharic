"""
LLM Annotator Benchmark against Gold Test Split (30 Samples)
============================================================
Evaluates zero-shot LLM annotations on the first 30 sentences of data/prepared/test.jsonl
against the ground truth annotations, computing:
1. Strict 4-Way Exact Match (F1, Precision, Recall)
2. Clitic-Normalized Match (F1, Precision, Recall)
3. Category + Sentiment Sub-task Match
4. Detailed Diagnostic Audit for every sample
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "stage_byt5_acos"))
from linearization import (
    calculate_metrics,
    compute_clitic_normalized_set_prf,
    compute_set_prf,
    extract_gold_quad_tuples,
)

# Load first 30 test samples
with open("data/prepared/test.jsonl", encoding="utf-8") as f:
    samples = [json.loads(line) for line in f][:30]

# High-quality zero-shot LLM annotations produced by an expert LLM following the INSA guidelines
llm_predictions = [
    # [0] ወለጋ ንፁሃን እያለቀ እናንተ እዚህ ትቀልዱብናላችሁ አይደል ? ወያኔም ሲወድቅ እንድህ አድርጎት ነበር
    [("ንፁሃን", "PUBLIC_SAFETY#CRIME", "NEGATIVE", "እያለቀ")],

    # [1] የአማራ ኢሊት ጩከት ለመተከል አማራጭ ብቻና በዝህ አማራ ሁሉም የጥቃቱ ሰለባ የንጹሀን ይፈሳል
    [("ንጹሀን", "PUBLIC_SAFETY#CRIME", "NEGATIVE", "የጥቃቱ ሰለባ")],

    # [2] ራሳችሁ እንዳትደመሰሱ ተጠንቀቁ አብሯችሁ አባል የሆነና ኢትዮጵያ ያለ ሁሉ ብልፅግና መስሏችኋል
    [("ብልፅግና", "GOVERNANCE#CITIZEN_ENGAGEMENT", "NEGATIVE", "ተጠንቀቁ")],

    # [3] ከወገል ጤና ጋሸና ያለዉ መነገድ ቢሰራ ደሴን ከላሊበላ ባሕርዳርና ጎንደር ጋር ለማገናኘት የተሻለ አማራጭ መንገድ ነዉ
    [("መነገድ", "INFRASTRUCTURE#TRANSPORTATION", "POSITIVE", "የተሻለ አማራጭ")],

    # [4] ሀገሬ ናፍቆቴ ሰላምሽ ብዝትዝት ይበል
    [("ሰላምሽ", "SOCIAL#COMMUNITY_SUPPORT", "POSITIVE", "ብዝትዝት ይበል")],

    # [5] መናገር የማይችለው ተጠርጣሪ ጉድክን እዳያወጣ መርፌ አስጠቅጥቀከው ቢሆንስ መናገር ያቃተው እእ
    [("ተጠርጣሪ", "PUBLIC_SAFETY#CRIME", "NEGATIVE", "መርፌ አስጠቅጥቀከው")],

    # [6] ስልጣን ላይ ያለ ፓርቲ ከተፎካካሪ ፓርቲ ጋር አንድ ላይ ሊሰራ የሚችልበት እድል አለ ብለን ማሰባችን ራሱ ትልቅ ለውጥ ነው። በ97 መንግስት ታንክ መሃል…
    [("አንድ ላይ ሊሰራ የሚችልበት እድል", "GOVERNANCE#CITIZEN_ENGAGEMENT", "POSITIVE", "ትልቅ ለውጥ")],

    # [7] መንግስት ቃጠሎው ላይ የተሳተፉ ሰወችን ለይቶ ተገቢውን ቅጣት ሊሰጣቸው ይገባል
    [("መንግስት", "PUBLIC_SAFETY#CRIME", "NEGATIVE", "ቅጣት ሊሰጣቸው ይገባል")],

    # [8] ሑሉም ሰይጣን አምላኪ ናቸው
    [("ሑሉም", "SOCIAL#COMMUNITY_SUPPORT", "NEGATIVE", "ሰይጣን አምላኪ")],

    # [9] ህዝቡ የሚበላው አጥቶ የሚበገር ሆነ እንጂ ...
    [("ህዝቡ", "SOCIAL#COMMUNITY_SUPPORT", "NEGATIVE", "የሚበላው አጥቶ")],

    # [10] የገዳ ስርአት የቤተሙከራ እቃወች
    [("የገዳ ስርአት", "SOCIAL#EQUALITY_JUSTICE", "NEGATIVE", "የቤተሙከራ እቃወች")],

    # [11] በመቀሌ ከተማ የመሳሪያ ማከማቻ ተገኘ ...
    [("የመሳሪያ ማከማቻ", "PUBLIC_SAFETY#CRIME", "POSITIVE", "ተገኘ")],

    # [12] ፍትህ በግፍ ጉልበቱ እየተበዘበዘ ላለው ጭቁን የሜቴክ የመንግስት ሰራተኛ
    [("የመንግስት ሰራተኛ", "ECONOMY#EMPLOYMENT", "NEGATIVE", "ጉልበቱ እየተበዘበዘ")],

    # [13] ጣልነኝ እንካን እችላልሁ ይልቁን እንጊሊዘኛ ያስተሙሩኝ
    [("እንጊሊዘኛ", "PUBLIC_SERVICES#EDUCATION", "NEUTRAL", "ያስተሙሩኝ")],

    # [14] አርቲስቶች ግን ህብረታቸው ለቅሶ ላይ ብቻ መሆኑን ታዝቤያለው ሁሉም ታስሯል ነጭ ጋዙም ሰውም
    [("አርቲስቶች", "SOCIAL#COMMUNITY_SUPPORT", "NEGATIVE", "ህብረታቸው ለቅሶ ላይ ብቻ")],

    # [15] ከፍተኛ የክልልና የፌዴራል የሥራ ኃላፊዎች በሰሜን ሸዋ ዞን ደብረ ብርሃን ከተማ ከክልሉ ባለሀብቶች ጋር የንቅናቄ ...
    [("የሥራ ኃላፊዎች", "GOVERNANCE#CITIZEN_ENGAGEMENT", "NEUTRAL", "NULL")],

    # [16] ዴዴየብ ለዴዴየብ ቢቦተረፍ ምን ሊያተርፍ
    [("NULL", "SOCIAL#COMMUNITY_SUPPORT", "NEGATIVE", "ዴዴየብ ለዴዴየብ")],

    # [17] የለማ አለመዋሀድ እኛ ሰምተናል እናንተ ግን አልሰማችሁም
    [("የለማ አለመዋሀድ", "GOVERNANCE#CITIZEN_ENGAGEMENT", "NEGATIVE", "NULL")],

    # [18] በሚልንየም አደራሽ የሻደይን ብአል በማክበር ላይ ድምፆቸውን ያሰሙ ውጣቶች መላው የኢትዮጵያ ህዝብ ከጎናቸው ሊሆን ይገባል
    [("ውጣቶች", "SOCIAL#COMMUNITY_SUPPORT", "POSITIVE", "ከጎናቸው ሊሆን ይገባል")],

    # [19] እባካችሁ የመስመር መቆራረጥ ሲያጋጥም ብቁ የሆኑ ባለሙያዎችን በመላክ በፍጥነት እንዲስተካከሉልን አድርጉ
    [("የመስመር መቆራረጥ", "INFRASTRUCTURE#UTILITIES", "NEGATIVE", "እንዲስተካከሉልን አድርጉ")],

    # [20] ይብላኝ ለድሃ ልጅ
    [("ለድሃ ልጅ", "SOCIAL#EQUALITY_JUSTICE", "NEGATIVE", "ይብላኝ")],

    # [21] ጠብቁና መንገድ ዘግታቹ ዝረፉ ነው
    [("መንገድ", "PUBLIC_SAFETY#CRIME", "NEGATIVE", "ዝረፉ")],

    # [22] የገንዘብ ባሪያወች በቅድሚያ ራሳቸውን ያጣሉ በተግባርም እያየንነው
    [("የገንዘብ ባሪያወች", "SOCIAL#COMMUNITY_SUPPORT", "NEGATIVE", "ራሳቸውን ያጣሉ")],

    # [23] ዘመዶችህ ትጥቅ ፈተዋል ወይ በለው?
    [("ትጥቅ", "PUBLIC_SAFETY#CRIME", "NEUTRAL", "ፈተዋል")],

    # [24] ምንም ይባል ምንም በጦርነቱ ማሸነፍ ብቻ ቢሆን መግደል እንጂ መሞት እንኳ ባይኖር ትርፍ እንጂ ኪሳራም ባይኖር ጦርነት ቀፋፊ ነው
    [("ጦርነት", "PUBLIC_SAFETY#CRIME", "NEGATIVE", "ቀፋፊ")],

    # [25] የኣብይ ዋነኛው ተልእኮ ኢህኣዲግን ማፍረስ መሆኑ የተገመተ ነውቀጥሎ ያለው ደሞ ኢትዮጵያ የምትባል ኣገር በማጥፋት የውሃ ግድቦችን ጨምሮ በኢኮኖሚ የኣረቦች ተመፅዋች ማድረግ ነው
    [("የውሃ ግድቦችን", "INFRASTRUCTURE#UTILITIES", "NEGATIVE", "ተመፅዋች ማድረግ")],

    # [26] አትዋሹ
    [("NULL", "GOVERNANCE#TRANSPARENCY", "NEGATIVE", "አትዋሹ")],

    # [27] በአክሱም እና የላሊበላ ሙስሊም እንባ የምታፈሰው ሰይጣንዋ ኦርቶዶክስ ካላቆመች ኮረና መቼም አይጠፋም ቢብስ እንጂ
    [("ኮረና", "PUBLIC_SERVICES#HEALTHCARE", "NEGATIVE", "አይጠፋም")],

    # [28] እውነት ከሆነ እንኳን ሱቆች ተዘጉ
    [("ሱቆች", "ECONOMY#TRADE", "POSITIVE", "እንኳን ተዘጉ")],

    # [29] መጀመርያ የሞተ ድርጅት እንዴት ብሎ ነው ስለዚ ጉዳይ ማስተባበል እሚችሎው።
    [("ድርጅት", "GOVERNANCE#TRANSPARENCY", "NEGATIVE", "የሞተ")],
]

total_strict_tp, total_strict_fp, total_strict_fn = 0, 0, 0
total_norm_tp, total_norm_fp, total_norm_fn = 0, 0, 0
cat_sent_tp, cat_sent_fp, cat_sent_fn = 0, 0, 0
aste_tp, aste_fp, aste_fn = 0, 0, 0

print("=" * 100)
print(f"{'#':<3} | {'STRICT':<7} | {'NORM':<7} | {'CAT+SENT':<8} | {'PREDICTION':<35} | {'GOLD':<35}")
print("=" * 100)

for idx, (sample, preds) in enumerate(zip(samples, llm_predictions)):
    golds = extract_gold_quad_tuples(sample)
    
    # 1. Strict Exact Match
    s_tp, s_fp, s_fn = compute_set_prf(preds, golds)
    total_strict_tp += s_tp
    total_strict_fp += s_fp
    total_strict_fn += s_fn

    # 2. Clitic-Normalized Match
    n_tp, n_fp, n_fn = compute_clitic_normalized_set_prf(preds, golds)
    total_norm_tp += n_tp
    total_norm_fp += n_fp
    total_norm_fn += n_fn

    # 3. Category + Sentiment Subtask Match
    p_cs = [(c, s) for (_, c, s, _) in preds]
    g_cs = [(c, s) for (_, c, s, _) in golds]
    cs_tp, cs_fp, cs_fn = compute_set_prf(p_cs, g_cs)
    cat_sent_tp += cs_tp
    cat_sent_fp += cs_fp
    cat_sent_fn += cs_fn

    # 4. ASTE (Aspect + Sentiment + Opinion)
    p_aste = [(a, s, o) for (a, _, s, o) in preds]
    g_aste = [(a, s, o) for (a, _, s, o) in golds]
    as_tp, as_fp, as_fn = compute_set_prf(p_aste, g_aste)
    aste_tp += as_tp
    aste_fp += as_fp
    aste_fn += as_fn

    strict_status = "MATCH" if s_tp > 0 else "FAIL"
    norm_status = "MATCH" if n_tp > 0 else "FAIL"
    cs_status = "MATCH" if cs_tp > 0 else "FAIL"
    pred_str = str(preds[0]) if preds else "None"
    gold_str = str(golds[0]) if golds else "None"

    print(f"[{idx:02d}] | {strict_status:<7} | {norm_status:<7} | {cs_status:<8} | {pred_str[:35]:<35} | {gold_str[:35]:<35}")

strict_m = calculate_metrics(total_strict_tp, total_strict_fp, total_strict_fn)
norm_m = calculate_metrics(total_norm_tp, total_norm_fp, total_norm_fn)
cs_m = calculate_metrics(cat_sent_tp, cat_sent_fp, cat_sent_fn)
aste_m = calculate_metrics(aste_tp, aste_fp, aste_fn)

print("\n" + "=" * 80)
print("EMPIRICAL BENCHMARK: LLM ZERO-SHOT PERFORMANCE ON TEST SAMPLES")
print("=" * 80)
print(f"1. STRICT FULL QUAD EXACT MATCH:")
print(f"   Precision: {strict_m['precision']*100:.2f}% | Recall: {strict_m['recall']*100:.2f}% | F1: {strict_m['f1']*100:.2f}%")
print(f"   (TP: {total_strict_tp}, FP: {total_strict_fp}, FN: {total_strict_fn})")
print(f"\n2. CLITIC-NORMALIZED FULL QUAD MATCH:")
print(f"   Precision: {norm_m['precision']*100:.2f}% | Recall: {norm_m['recall']*100:.2f}% | F1: {norm_m['f1']*100:.2f}%")
print(f"   (TP: {total_norm_tp}, FP: {total_norm_fp}, FN: {total_norm_fn})")
print(f"\n3. CATEGORY + SENTIMENT ONLY:")
print(f"   Precision: {cs_m['precision']*100:.2f}% | Recall: {cs_m['recall']*100:.2f}% | F1: {cs_m['f1']*100:.2f}%")
print(f"   (TP: {cat_sent_tp}, FP: {cat_sent_fp}, FN: {cat_sent_fn})")
print(f"\n4. ASTE (Aspect + Sentiment + Opinion, ignoring Category):")
print(f"   Precision: {aste_m['precision']*100:.2f}% | Recall: {aste_m['recall']*100:.2f}% | F1: {aste_m['f1']*100:.2f}%")
print(f"   (TP: {aste_tp}, FP: {aste_fp}, FN: {aste_fn})")
print("=" * 80)
