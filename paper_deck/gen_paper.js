const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, AlignmentType, LevelFormat, convertInchesToTwip
} = require("docx");

const FONT = "Calibri";
const HEAD_FONT = "Cambria";

function h1(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 320, after: 160 } });
}
function h2(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 240, after: 120 } });
}
function p(text, opts = {}) {
  return new Paragraph({
    children: [new TextRun({ text, font: FONT, size: 22, italics: opts.italics || false, bold: opts.bold || false })],
    spacing: { after: 160, line: 276 },
    alignment: opts.center ? AlignmentType.CENTER : AlignmentType.JUSTIFIED,
  });
}
function bullet(text) {
  return new Paragraph({
    children: [new TextRun({ text, font: FONT, size: 22 })],
    numbering: { reference: "bullets", level: 0 },
    spacing: { after: 80 },
  });
}
function caption(text) {
  return new Paragraph({
    children: [new TextRun({ text, font: FONT, size: 18, italics: true })],
    spacing: { after: 240 },
    alignment: AlignmentType.CENTER,
  });
}

function cell(text, opts = {}) {
  return new TableCell({
    width: { size: opts.width, type: WidthType.DXA },
    shading: opts.header ? { type: ShadingType.CLEAR, fill: "1E2761" } : undefined,
    children: [new Paragraph({
      children: [new TextRun({ text, font: FONT, size: opts.header ? 19 : 19, bold: !!opts.header, color: opts.header ? "FFFFFF" : "000000" })],
    })],
    verticalAlign: "center",
  });
}

function makeTable(headers, rows, colWidths) {
  const tableWidth = colWidths.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: tableWidth, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [
      new TableRow({ children: headers.map((h, i) => cell(h, { header: true, width: colWidths[i] })) }),
      ...rows.map(r => new TableRow({ children: r.map((c, i) => cell(String(c), { width: colWidths[i] })) })),
    ],
  });
}

const doc = new Document({
  numbering: {
    config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] }],
  },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    children: [

      // Title block
      new Paragraph({
        children: [new TextRun({ text: "Amharic ACOS: A Pipeline Approach to Aspect-Category-Opinion-Sentiment", font: HEAD_FONT, size: 34, bold: true })],
        alignment: AlignmentType.CENTER, spacing: { after: 60 },
      }),
      new Paragraph({
        children: [new TextRun({ text: "Quadruple Extraction for Civic-Domain Amharic Text", font: HEAD_FONT, size: 34, bold: true })],
        alignment: AlignmentType.CENTER, spacing: { after: 200 },
      }),
      new Paragraph({
        children: [new TextRun({ text: "Progress Report", font: FONT, size: 24, italics: true })],
        alignment: AlignmentType.CENTER, spacing: { after: 360 },
      }),

      h1("Abstract"),
      p("Aspect-Based Sentiment Analysis (ABSA) for Amharic remains under-resourced, with prior work limited to aspect-term sentiment classification via classical machine learning and no existing resource for full Aspect-Category-Opinion-Sentiment (ACOS) quadruple extraction with implicit aspect and opinion handling. We present an in-progress ACOS pipeline for Amharic civic and governance text, built on a dataset of 39,261+ annotated sentences across a fixed, externally-mandated 22-category taxonomy. Our modular architecture decomposes ACOS extraction into six stages; four are implemented and evaluated here: (1) joint aspect/opinion term extraction via BIO tagging, (2) aspect-opinion pairing, (3) category classification, and (4) sentiment classification. Data analysis directly informed two key architectural decisions: a distance-based heuristic (F1 = 0.993) was adopted for pairing in place of a learned classifier after finding fewer than 1% of sentences contain genuine pairing ambiguity, and logit-adjusted loss was adopted over naive class weighting to address category imbalance so severe that several categories retain fewer than 30 -- or zero -- training examples. We report results to date, document an evaluation-metric pitfall that we identified and corrected, and outline the remaining implicit-detection and end-to-end evaluation stages as future work.", { }),

      h1("1. Introduction"),
      p("Aspect-Based Sentiment Analysis identifies fine-grained opinions toward specific aspects of an entity, rather than a single sentiment label for an entire document. The most complete formulation, ACOS quadruple extraction, requires identifying the tuple (aspect term, aspect category, opinion term, sentiment polarity) for every opinion expressed in a sentence, including cases where the aspect term or opinion term is not explicitly stated in the text (Cai et al., 2021)."),
      p("Amharic, spoken by over 30 million people as a first language and serving as Ethiopia's federal working language, has seen comparatively little ABSA research. Existing work is limited to aspect-level sentiment classification using classical machine learning (Naive Bayes, SVM, k-NN) on social media and telecom-domain text, with full aspect-category-opinion-sentiment extraction and implicit-element handling explicitly noted as open directions rather than addressed. No existing Amharic dataset or system performs full ACOS quadruple extraction to our knowledge."),
      p("This work makes three contributions: (1) a large-scale Amharic ACOS dataset for the civic and governance domain, generated from mT5/mC4/OSCAR-sourced text and annotated for full quadruples -- including implicit aspects and opinions -- using an LLM annotator; (2) a modular, six-stage extraction pipeline, of which four stages are implemented and evaluated; and (3) a set of data-driven architectural decisions -- a heuristic pairing stage justified by direct ambiguity analysis, and a logit-adjustment strategy for a category taxonomy too fine-grained and externally fixed to simplify by relabeling -- that we document in enough detail to be reproducible and revisited as the dataset grows."),

      h1("2. Related Work"),
      h2("2.1 ACOS and Quadruple Extraction"),
      p("Cai et al. (2021) introduced the ACOS task and the Restaurant-ACOS (2,286 sentences, 3,661 quadruples) and Laptop-ACOS (4,076 sentences, 5,773 quadruples) datasets, along with the Extract-Classify-ACOS baseline: BIO-tag aspect and opinion terms, use binary classifiers to detect implicit aspects/opinions, form candidate pairs via Cartesian product, then classify category and sentiment jointly per pair. Our pipeline's stage decomposition converges independently on a similar structure, though our Stage 2 pairing decision (Section 4.2) was reached through direct analysis of our own data's pairing ambiguity rather than assumed from precedent. An alternative paradigm, generative paraphrase-based extraction (Zhang et al., 2021), reformulates quadruple extraction as sequence-to-sequence generation; we did not adopt this approach given Amharic's small representation in the pretraining data of available multilingual generative models, which favors extraction/classification sub-tasks over free-form generation for this language."),
      h2("2.2 Amharic Language Resources"),
      p("We evaluate three pretrained Amharic-capable encoders as pipeline backbones: AfroXLMR-base (Alabi et al., 2022), an XLM-R model continually pretrained on 17 African languages including Amharic; bert-small-amharic, a compact (27.8M parameter) model pretrained from scratch on Amharic web text; and roberta-base-amharic (110M parameters), pretrained with an Amharic-specific vocabulary on the same source corpora (OSCAR, mC4) used to generate our dataset. On an independent Amharic sentiment benchmark, roberta-base-amharic (F1 = 0.88) outperforms both AfroXLMR-base (F1 = 0.83) and the substantially larger AfroXLMR-large (560M parameters, F1 = 0.86), motivating its inclusion as a comparison backbone in this work despite its smaller size."),
      h2("2.3 Long-Tail and Imbalanced Classification"),
      p("Naive inverse-frequency class weighting is a documented weak mitigation for severe class imbalance, correcting per-example loss magnitude without directly addressing the classifier's decision-boundary bias. We instead adopt logit-adjusted cross-entropy (Menon et al., 2021), which incorporates class priors directly into the training objective and is reported to more effectively counteract long-tailed label distributions than reweighting alone."),

      h1("3. Dataset"),
      p("Source text was drawn from mT5, mC4, and OSCAR Amharic web-crawl corpora, filtered to civic and governance content, and annotated for ACOS quadruples using an LLM annotator (Gemini). Human verification of a held-out sample is planned but not yet complete; this is a limitation we discuss in Section 7."),
      h2("3.1 Category Taxonomy"),
      p("Categories follow a fixed, externally-mandated two-level naming convention (DOMAIN#SUBCATEGORY, e.g. GOVERNANCE#TRANSPARENCY) spanning seven domains: ECONOMY, GOVERNANCE, HEALTHCARE, INFRASTRUCTURE, PUBLIC_SAFETY, PUBLIC_SERVICES, and SOCIAL. The taxonomy comprises 22 categories in total and cannot be merged or renamed for modeling convenience, as it reflects a stakeholder requirement rather than a modeling choice -- an early version of this pipeline collapsed overlapping fine-grained labels across domains for consistency, and this was reverted once the constraint was clarified."),
      h2("3.2 Scale and Statistics"),
      makeTable(
        ["Statistic", "Value"],
        [
          ["Training sentences", "39,261"],
          ["Training quadruples", "41,224"],
          ["Categories (fixed taxonomy)", "22"],
          ["Implicit aspects", "~32% of quadruples"],
          ["Implicit opinions", "~19% of quadruples"],
          ["Both aspect and opinion implicit", "~8% of quadruples"],
          ["Sentiment distribution (full dataset)", "63.8% negative / 27.5% positive / 8.7% neutral"],
        ],
        [5500, 3500]
      ),
      caption("Table 1: Dataset scale and key statistics."),
      p("Stages 3 and 4 (category and sentiment classification) train on the subset of quadruples where both the aspect and opinion are explicit spans -- 23,617 of 41,224 training quadruples (57.3%). Within this subset, sentiment shifts notably: neutral sentiment drops from 8.7% (full dataset) to 3.1%, indicating that neutral-sentiment quadruples disproportionately involve an implicit opinion term. This is, to our knowledge, a previously undocumented interaction between implicitness and sentiment class distribution."),
      p("Category frequency is severely long-tailed: the largest category alone accounts for roughly a quarter of all quadruples, while ten categories have fewer than 30 explicit-pair training examples. At least one category (ECONOMY#UTILITIES) has zero training examples in the full dataset despite appearing in the test split, and two further categories (PUBLIC_SERVICES#COMMUNITY_SUPPORT, PUBLIC_SERVICES#INFRASTRUCTURE) have zero examples specifically within the explicit-pairs subset used for classification, since they only co-occur with implicit spans in this dataset. These are documented as structural limitations rather than treated as classifier failures: no amount of algorithmic improvement can produce a correct prediction for a category with zero training signal."),

      h1("4. Pipeline Architecture"),
      makeTable(
        ["Stage", "Task", "Status", "Approach"],
        [
          ["1", "Aspect + opinion term extraction", "Implemented", "Joint BIO tagging, shared encoder, dual linear heads"],
          ["2", "Aspect-opinion pairing", "Implemented", "Nearest-token-distance heuristic (no training required)"],
          ["3", "Category classification", "Implemented", "Span-pair classifier, logit-adjusted loss, 22-way"],
          ["4", "Sentiment classification", "Implemented", "Same architecture as Stage 3, 3-way"],
          ["5", "Implicit aspect/opinion detection", "Planned", "Sentence-level classification + context-based category/sentiment"],
          ["6", "End-to-end quadruple assembly", "Planned", "Combine stages 1-5; exact-match quadruple F1"],
        ],
        [900, 2800, 1600, 3700]
      ),
      caption("Table 2: Pipeline stage overview."),

      h2("4.1 Stage 1: Aspect and Opinion Term Extraction"),
      p("Aspect and opinion terms are extracted jointly via BIO tagging over a shared pretrained encoder with two independent linear classification heads. Word-level spans are aligned to subword tokens using the tokenizer's word-id mapping, with the first subword of a word carrying the word's tag and continuation subwords inheriting an inside (I) tag. This stage covers only explicit spans by construction; implicit terms have no span to tag and are deferred to Stage 5."),
      h2("4.2 Stage 2: Aspect-Opinion Pairing"),
      p("Rather than training a neural pairing classifier, we analyzed the data directly and found that fewer than 1% of sentences (331 of 39,261) contain any genuine pairing ambiguity, with only 704 negative candidate pairs existing across the entire training set -- insufficient signal to reliably train a classifier. True aspect-opinion pairs sit at a median token distance of 1, versus 6 for false candidate pairs, and over 99% of quadruples are strict one-to-one aspect-opinion mappings. We therefore adopt a deterministic nearest-token-distance heuristic: each aspect span is paired with its nearest opinion span(s) by token distance, and vice versa, with the union taken across both directions. This achieves F1 = 0.993 on both training and test splits when evaluated against gold spans, with no gap between splits -- as expected of a fixed rule rather than a learned model."),
      h2("4.3 Stage 3: Category Classification"),
      p("A shared-encoder span-pair classifier pools the aspect span, opinion span, and full sentence (masked mean pooling over each span's subword tokens), concatenates the three representations, and classifies via a two-layer MLP into the 22-category space. Given the severity of the class imbalance described in Section 3.2, we use logit-adjusted cross-entropy by default: a class-prior-derived offset is added to each class's logit during training, directly shaping the decision boundary rather than only reweighting the loss magnitude of individual examples. An inverse-frequency class-weighted loss (capped to prevent instability from near-zero-count classes) is retained as a configurable alternative for comparison."),
      p("During evaluation we identified and corrected a metric artifact: averaging per-category F1 across all 22 categories forces a score of zero for any category with no examples in the evaluation split, which can substantially understate true performance when many categories are absent from a given split (9 of 22 in our test evaluation). We report two macro-F1 variants -- one over all 22 categories (for transparency about taxonomy coverage) and one restricted to categories actually present in the evaluation set (the operationally meaningful number) -- and use the latter for checkpoint selection."),
      h2("4.4 Stage 4: Sentiment Classification"),
      p("Stage 4 reuses the Stage 3 architecture unchanged apart from the output layer (3-way softmax over NEGATIVE/NEUTRAL/POSITIVE) and the label field. Given neutral sentiment's scarcity in the explicit-pairs training subset (3.1%, Section 3.2) -- a pattern consistently reported as the hardest case for ABSA sentiment classifiers under imbalance in the broader literature -- logit-adjusted loss is again the default, and per-class metrics, with particular attention to neutral-class recall, are logged every training epoch rather than only accuracy or an aggregate score, since a model could otherwise reach roughly 65% accuracy by never predicting the neutral class at all."),

      h1("5. Experimental Setup"),
      p("Three pretrained encoders are compared as pipeline backbones: AfroXLMR-base (278M parameters), bert-small-amharic (27.8M parameters), and roberta-base-amharic (110M parameters, Amharic-specific vocabulary). All stages are trained with AdamW, linear warmup followed by linear decay, gradient clipping (max norm 1.0), and mixed-precision (fp16) training. Experiment configurations are version-controlled as YAML files with fixed random seeds, and resolved run arguments are saved alongside each checkpoint for reproducibility."),

      h1("6. Results"),
      h2("6.1 Stage 1: Aspect and Opinion Term Extraction"),
      makeTable(
        ["Backbone", "Aspect F1", "Aspect P / R", "Opinion F1", "Opinion P / R"],
        [
          ["AfroXLMR-base", "0.599", "0.613 / 0.586", "0.458", "0.468 / 0.449"],
          ["bert-small-amharic", "0.551", "0.602 / 0.508", "0.389", "0.420 / 0.362"],
          ["roberta-base-amharic", "pending", "--", "pending", "--"],
        ],
        [3200, 1600, 2300, 1600, 1300]
      ),
      caption("Table 3: Stage 1 best-checkpoint results (span-level F1 against explicit gold spans only)."),
      p("AfroXLMR-base outperforms bert-small-amharic on both subtasks despite the latter's roughly 6x faster training time, a reasonable trade given the ~6-10x parameter difference. Opinion extraction lags aspect extraction for both backbones, consistent with opinion terms typically being longer, more variable expressions with less agreed-upon span boundaries than aspect nouns. Both backbones show diminishing returns after approximately epoch 4, suggesting a longer schedule with learning-rate decay, rather than simply more epochs, is the more promising next step. The roberta-base-amharic run is in progress at the time of writing."),
      h2("6.2 Stage 2: Pairing"),
      p("The heuristic pairing approach achieves precision 0.989 / recall 0.998 / F1 0.993 on the training split and precision 0.988 / recall 0.998 / F1 0.993 on the test split, evaluated against gold spans in isolation from Stage 1's extraction errors. This isolated result represents a ceiling for pairing quality alone; end-to-end pipeline performance additionally depends on Stage 1's span extraction accuracy, since an incorrect or missed span cannot be correctly paired regardless of pairing logic quality."),
      h2("6.3 Stage 3: Category Classification"),
      p("On bert-small-amharic with class-weighted loss (an earlier configuration, prior to the logit-adjustment default described in Section 4.3), the model reaches accuracy 0.622 and macro-F1 (present categories only) of approximately 0.59-0.64. Per-category F1 for categories with real test support ranges from 0.45 (ECONOMY#EMPLOYMENT) to 0.83 (PUBLIC_SERVICES#EDUCATION), with nine of the 22 categories absent from this particular test evaluation entirely -- a test-split coverage limitation rather than a modeling weakness. A comparable run on AfroXLMR-base produced similar overall performance. Re-evaluation with logit-adjusted loss, the current default, is in progress and expected to improve performance specifically on moderate-support categories such as ECONOMY#EMPLOYMENT."),
      h2("6.4 Stage 4: Sentiment Classification"),
      p("The Stage 4 architecture and training pipeline are implemented and verified against the dataset (23,617 training examples correctly extracted and label-distributed as expected: 15,363 negative, 7,513 positive, 741 neutral). Empirical training results are pending at the time of writing; the neutral-class recall metric described in Section 4.4 will be the primary indicator of whether logit-adjusted loss alone is sufficient or whether targeted data augmentation for the neutral class (Section 8) is required."),

      h1("7. Discussion and Limitations"),
      bullet("Annotation validity: the dataset is currently 100% LLM-annotated. Human verification of a held-out sample, with inter-annotator agreement reported per quadruple element, is necessary before annotation quality claims can be made with confidence -- particularly for implicit aspects and opinions, where LLM annotation is most prone to defaulting to a plausible-sounding rather than textually-grounded reading."),
      bullet("Fixed taxonomy constrains modeling options: because the 22-category taxonomy reflects an external stakeholder requirement, imbalance must be addressed through loss-function and training-strategy choices that preserve the full label space, rather than through category merging -- a more common mitigation in ABSA research where the taxonomy is a design choice rather than a fixed requirement."),
      bullet("Structurally zero-support categories are a data-collection problem: three categories currently cannot be predicted under any technique explored here, because they have no (or, in one case, only test-time) training examples. This limitation is orthogonal to model or algorithm quality."),
      bullet("Pairing heuristic ceiling: the nearest-distance heuristic's known failure mode is genuine semantic ambiguity where the correct pairing is not the nearest one. This is currently rare enough (<1% of sentences) not to matter empirically, but should be re-examined if future dataset expansion increases the rate of multi-quadruple sentences."),
      bullet("Small-sample classes limit achievable performance regardless of technique: with 741 neutral-sentiment and single-digit-count category examples, there is a data volume floor that no loss-function adjustment alone can overcome."),

      h1("8. Future Work"),
      bullet("Complete Stage 4 experiments, including the roberta-base-amharic backbone comparison and evaluation of logit-adjusted loss against the class-weighted baseline."),
      bullet("Implement Stage 5 (implicit aspect/opinion detection) and Stage 6 (end-to-end quadruple assembly and exact-match evaluation, reported separately for explicit-only, implicit-involving, and combined quadruples)."),
      bullet("Conduct a human-verification study with native Amharic speakers on a held-out sample, reporting inter-annotator agreement against the LLM annotations."),
      bullet("Explore targeted synthetic data augmentation (backtranslation, paraphrasing) for the neutral sentiment class and the lowest-support categories specifically."),

      h1("9. Conclusion"),
      p("We present an in-progress pipeline for Amharic ACOS quadruple extraction on a civic-domain dataset substantially larger than existing English-language ACOS resources, with four of six planned pipeline stages implemented and evaluated. Two architectural decisions -- heuristic rather than learned pairing, and logit-adjusted rather than naively-reweighted classification loss -- were reached through direct analysis of the dataset's actual statistical properties rather than assumed from precedent, and are documented here so they can be revisited as the dataset grows. The remaining implicit-detection and end-to-end evaluation stages, along with human annotation verification, are the immediate next steps toward a complete system."),

      h1("References"),
      p("Alabi, J. O., Adelani, D. I., Mosbach, M., & Klakow, D. (2022). Adapting Pre-trained Language Models to African Languages via Multilingual Adaptive Fine-Tuning. In Proceedings of the 29th International Conference on Computational Linguistics (COLING 2022), pp. 4336-4349."),
      p("Cai, H., Xia, R., & Yu, J. (2021). Aspect-Category-Opinion-Sentiment Quadruple Extraction with Implicit Aspects and Opinions. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (ACL-IJCNLP 2021), pp. 340-350."),
      p("Menon, A. K., Jayasumana, S., Rawat, A. S., Jain, H., Veit, A., & Kumar, S. (2021). Long-Tail Learning via Logit Adjustment. In International Conference on Learning Representations (ICLR 2021)."),
      p("Zhang, W., Deng, Y., Li, X., Yuan, Y., Bing, L., & Lam, W. (2021). Aspect Sentiment Quad Prediction as Paraphrase Generation. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP 2021)."),
      p("rasyosef. bert-small-amharic and roberta-base-amharic [pretrained language models]. Hugging Face Model Hub."),
    ],
  }],
});

Packer.toBuffer(doc).then(buf => {
  require("fs").writeFileSync("amharic_acos_paper.docx", buf);
  console.log("written");
});
