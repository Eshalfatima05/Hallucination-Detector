const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, AlignmentType, LevelFormat, convertInchesToTwip,
} = require("docx");
const fs = require("fs");

const INK = "1B2430";
const SOFT = "5B6472";
const RULE = "D8DAD3";
const ACCENT = "A83A3A";
const OK = "3D6B54";
const HEAD_SHADE = "EFEEE6";

function cell(text, opts = {}) {
  const { bold = false, width, shade, color = INK, align = AlignmentType.LEFT } = opts;
  return new TableCell({
    width: width ? { size: width, type: WidthType.DXA } : undefined,
    shading: shade ? { type: ShadingType.CLEAR, fill: shade } : undefined,
    margins: { top: 80, bottom: 80, left: 100, right: 100 },
    children: [new Paragraph({
      alignment: align,
      children: [new TextRun({ text: String(text), bold, color, size: 19 })],
    })],
  });
}

function table(headers, rows, widths) {
  const headerRow = new TableRow({
    tableHeader: true,
    children: headers.map((h, i) => cell(h, { bold: true, shade: HEAD_SHADE, width: widths[i], color: INK })),
  });
  const bodyRows = rows.map(r => new TableRow({
    children: r.map((v, i) => cell(v, { width: widths[i] })),
  }));
  return new Table({
    width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    columnWidths: widths,
    borders: {
      top: { style: BorderStyle.SINGLE, size: 4, color: RULE },
      bottom: { style: BorderStyle.SINGLE, size: 4, color: RULE },
      left: { style: BorderStyle.NONE },
      right: { style: BorderStyle.NONE },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: RULE },
      insideVertical: { style: BorderStyle.NONE },
    },
    rows: [headerRow, ...bodyRows],
  });
}

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 160 },
    children: [new TextRun({ text, color: INK, bold: true })],
  });
}
function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 260, after: 120 },
    children: [new TextRun({ text, color: INK, bold: true })],
  });
}
function p(text, opts = {}) {
  const { italic = false, size = 21, spacingAfter = 160, color = INK } = opts;
  return new Paragraph({
    spacing: { after: spacingAfter, line: 300 },
    children: [new TextRun({ text, italic, size, color })],
  });
}
function bullet(text) {
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: { after: 80, line: 290 },
    children: [new TextRun({ text, size: 21, color: INK })],
  });
}
function caption(text) {
  return new Paragraph({
    spacing: { before: 60, after: 220 },
    children: [new TextRun({ text, size: 17, italics: true, color: SOFT })],
  });
}

const doc = new Document({
  numbering: {
    config: [{
      reference: "bullets",
      levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2014", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 360, hanging: 260 } } } }],
    }],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1080, bottom: 1080, left: 1200, right: 1200 },
      },
    },
    children: [
      new Paragraph({
        spacing: { after: 40 },
        children: [new TextRun({ text: "RESEARCH REPORT", size: 17, bold: true, color: SOFT })],
      }),
      new Paragraph({
        spacing: { after: 80 },
        children: [new TextRun({
          text: "Claim-Level Hallucination Detection for LLM Outputs",
          size: 40, bold: true, color: INK,
        })],
      }),
      new Paragraph({
        spacing: { after: 260 },
        children: [new TextRun({
          text: "A retrieval-grounded, feature-based verification layer evaluated on RAGTruth",
          size: 24, color: SOFT, italics: true,
        })],
      }),
      p("September 10, 2026", { size: 18, color: SOFT, spacingAfter: 360 }),

      h1("Abstract"),
      p("Large language models frequently produce fluent text that is not actually supported by the source documents they were given, a failure mode generally called hallucination. This report describes an independent claim-level verification system: given an LLM response and its source document(s), the system segments the response into sentence-level claims, retrieves the most relevant evidence for each claim, scores how well-grounded each claim is, and flags likely hallucinations. The system is trained and evaluated on RAGTruth (Niu et al., 2024), a public benchmark of about 18,000 human-annotated LLM responses spanning question answering, news summarization, and data-to-text generation. Because this project's execution environment could not reach pretrained embedding or NLI model weights, the entailment step was built from classical retrieval (TF-IDF) and lexical grounding features \u2014 novel-word ratio, numeric/entity consistency, and retrieval similarity \u2014 combined in a gradient-boosted classifier, rather than an off-the-shelf NLI model. On RAGTruth's held-out test split, the system reaches a claim-level F1 of 0.36 (AUC 0.79) on the hallucination class and a response-level F1 of 0.64 for flagging whether a response contains any hallucination at all. Error analysis shows the detector is substantially better at catching baseless, invented content than at catching subtle contradictions that reuse the source's own wording. The report closes with a live annotated demo and a discussion of what a heavier, embedding- and NLI-based version of the same pipeline would likely add."),

      h1("1. Motivation"),
      p("Hallucination is the most commonly cited reason companies hesitate to deploy LLMs in legal, medical, financial, and journalistic settings, where a confidently stated but unsupported claim can cause real harm. The tractable version of this problem is narrower than open-ended fact-checking: given a specific set of source documents \u2014 for example the retrieved context in a retrieval-augmented generation (RAG) pipeline \u2014 the question is simply whether each claim in the model's response is actually entailed by those documents. That is a well-defined, checkable task, and it is the one this project builds and evaluates a system for."),

      h1("2. Related Work"),
      p("RAGTruth (Niu et al., 2024) is the primary benchmark used here: a corpus of nearly 18,000 LLM responses across QA, news summarization, and data-to-text generation, with hallucinated spans hand-annotated at the word level and typed as Evident Conflict, Subtle Conflict, Evident Baseless Info, or Subtle Baseless Info. The original RAGTruth paper also shows that a small LLM fine-tuned directly on this data can match prompting-based detection with state-of-the-art LLMs, and follow-on work such as LettuceDetect (a span-level detector built on encoder models trained on RAGTruth) and RAG-HAT (a fine-tuned Llama-based detector) report substantially higher scores than lexical baselines by relying on learned language representations rather than hand-built features. This project's classical, feature-based approach sits closer to an older family of methods \u2014 self-consistency and n-gram-overlap detectors such as SelfCheckGPT (Manakul et al., 2023) \u2014 which trade some accuracy for being fully local, fast, and free of any pretrained-model dependency."),

      h1("3. Data"),
      p("RAGTruth's response.jsonl gives each LLM-generated response plus character-level hallucination spans; source_info.jsonl gives the corresponding source document, which is a plain string for the Summary task, a question-plus-passages object for QA, and structured JSON (business attributes and review text) for Data2txt. All source_info records were flattened into plain text, and every response was segmented into sentences with NLTK's Punkt tokenizer. A sentence was labeled hallucinated if it overlapped any annotated hallucination span, and supported otherwise. Responses marked as truncated or as an incorrect refusal in RAGTruth's own quality field were excluded."),
      table(
        ["Split", "Responses", "Claims (sentences)", "Hallucinated claims", "Positive rate"],
        [
          ["Train", "14,942", "105,775", "12,296", "11.6%"],
          ["Test", "2,675", "17,975", "1,556", "8.7%"],
        ],
        [1800, 2400, 2600, 2400, 2000],
      ),
      caption("Table 1. Sentence-level dataset sizes after quality filtering and sentence segmentation."),

      h1("4. Method"),
      h2("4.1 Claim segmentation"),
      p("Each response is split into sentences via NLTK sentence tokenization; each sentence is treated as one checkable claim. This mirrors the RAGTruth annotation granularity closely enough that gold labels transfer cleanly (a sentence is positive if it overlaps any annotated span)."),

      h2("4.2 Evidence retrieval"),
      p("For each source document, retrieval chunks are built per task type: sentences for Summary, passage-then-sentence for QA, and individual \u201ckey: value\u201d lines for the flattened Data2txt JSON. A TF-IDF vectorizer (word 1\u20132-grams) is fit per source document over its own chunks, and the top-3 chunks by cosine similarity to the claim are retrieved as evidence."),

      h2("4.3 Grounding features in place of a pretrained NLI model"),
      p("This sandbox's network allowlist covers package registries (PyPI, npm, GitHub) but not huggingface.co, so no pretrained sentence-embedding or NLI model could be downloaded \u2014 confirmed directly (huggingface.co returned HTTP 403 through the egress proxy). Rather than skip the entailment step, it was rebuilt from features with a similar detection rationale to NLI, all computable offline:"),
      bullet("max_sim / mean_top3_sim \u2014 TF-IDF cosine similarity to the best-matching evidence chunk(s); low similarity suggests nothing in the source addresses the claim."),
      bullet("novel_word_ratio \u2014 the fraction of the claim's content words that appear nowhere in the source document at all; the single strongest feature in the final model."),
      bullet("num_mismatch_ratio / has_numbers \u2014 the fraction of numbers in the claim (prices, dates, percentages, counts) that do not appear verbatim anywhere in the source; a direct check for fabricated statistics."),
      bullet("cap_mismatch_ratio / has_caps \u2014 the same idea applied to capitalized-word spans, as a lightweight proxy for named entities."),
      bullet("claim_len_words, position_ratio, task-type one-hot \u2014 structural controls."),
      p("These features were chosen to approximate what an NLI model's entailment/contradiction/neutral judgment would pick up on \u2014 lexical grounding for \u201centailment,\u201d unsupported specifics for \u201cneutral,\u201d and numeric/entity mismatch for \u201ccontradiction\u201d \u2014 without requiring a downloaded model. Section 8 discusses what a true pretrained NLI model would likely add."),

      h2("4.4 Classifier"),
      p("A logistic regression and a gradient-boosted tree ensemble (scikit-learn) were trained on the 12 features above using RAGTruth's train split (85/15 train/validation), with the validation split used to pick both the better model and its decision threshold by maximizing F1 on the hallucination class. Gradient boosting won on validation (F1 0.41 vs. 0.35 for logistic regression) and was refit on the full train split before the single held-out evaluation on the test split reported below."),

      h1("5. Results"),
      p("All results are on RAGTruth's test split (2,675 responses, 17,975 claim sentences), which was not used for any training or threshold selection."),
      table(
        ["Metric", "Value"],
        [
          ["Precision (hallucination class)", "0.29"],
          ["Recall (hallucination class)", "0.49"],
          ["F1 (hallucination class)", "0.36"],
          ["AUC", "0.79"],
        ],
        [5200, 2200],
      ),
      caption("Table 2. Claim (sentence)-level results."),
      table(
        ["Metric", "Value"],
        [
          ["Precision", "0.55"],
          ["Recall", "0.76"],
          ["F1", "0.64"],
          ["Responses actually containing \u2265 1 hallucination", "35%"],
          ["Responses flagged by the system", "48%"],
        ],
        [5200, 2200],
      ),
      caption("Table 3. Response-level results: a response is \u201cflagged\u201d if the system marks at least one of its claims as hallucinated."),
      p("The gap between claim-level F1 (0.36) and response-level F1 (0.64) is expected and, in this application, the more useful number: a verification layer sitting in front of a chat or RAG product mainly needs to answer \u201cshould a human double-check this response,\u201d and it recovers 76% of responses that truly contain a hallucination, at the cost of also flagging some clean responses for review."),
      table(
        ["Task type", "N claims", "Hallucination rate", "Precision", "Recall", "F1"],
        [
          ["Summary", "4,821", "5.4%", "0.25", "0.15", "0.19"],
          ["QA", "5,175", "7.1%", "0.31", "0.62", "0.41"],
          ["Data2txt", "7,979", "11.7%", "0.28", "0.53", "0.37"],
        ],
        [1800, 1600, 2000, 1600, 1400, 1200],
      ),
      caption("Table 4. Breakdown by task type. Free-form summarization is both the hardest task and the one where this feature set does worst \u2014 summaries paraphrase more, which erodes the lexical-overlap features the model relies on most."),
      table(
        ["Feature", "Relative importance"],
        [
          ["novel_word_ratio", "0.46"],
          ["position_ratio", "0.11"],
          ["num_mismatch_ratio", "0.11"],
          ["claim_len_words", "0.08"],
          ["is_qa (task type)", "0.07"],
          ["mean_top3_sim", "0.06"],
          ["max_sim", "0.05"],
          ["all remaining features", "0.06 combined"],
        ],
        [4200, 3200],
      ),
      caption("Table 5. Gradient-boosting feature importances. The single feature measuring whether a claim's own words appear anywhere in the source document dominates \u2014 unsurprising for a lexical-overlap approach, but also a clue to its main blind spot (Section 6)."),

      h1("6. Error Analysis"),
      p("Breaking test-set errors down by RAGTruth's own hallucination-type taxonomy shows a consistent pattern: the detector is much better at catching invented content than at catching subtle contradictions."),
      table(
        ["Hallucination type", "N in test set", "Recall"],
        [
          ["Evident Baseless Info", "878", "0.53"],
          ["Evident Conflict", "626", "0.47"],
          ["Subtle Baseless Info", "174", "0.48"],
          ["Subtle Conflict", "16", "0.31"],
        ],
        [3400, 2200, 2000],
      ),
      caption("Table 6. Recall by hallucination type. \u201cBaseless\u201d claims (invented information absent from the source) are caught roughly half the time; \u201cSubtle Conflict\u201d \u2014 a claim that contradicts the source while reusing much of its wording \u2014 is the hardest category by a wide margin, though this category is also rare (16 examples)."),
      p("This gap follows directly from the feature set. Comparing missed hallucinations (false negatives) to caught ones (true positives): missed claims have a much higher retrieval similarity to their evidence (max_sim 0.43 vs. 0.32) and a much lower novel-word ratio (0.48 vs. 0.72). In plain terms, the claims the system misses are the ones that reuse the source's own words closely while quietly inverting or altering a detail \u2014 a negation flip, a swapped name, a changed outcome \u2014 rather than introducing new vocabulary. Lexical-overlap features are, by construction, poorly suited to catching a claim that is textually close to its evidence but semantically wrong. This is exactly the case a real NLI model (trained explicitly to distinguish entailment from contradiction, not just to measure overlap) is designed for, and is the clearest, most concrete argument in this report for why Section 8's proposed upgrade path matters."),
      p("A second, milder pattern: false positives (1,879 of them, versus 756 true positives) are concentrated in claims with unusually high novel-word ratio for benign reasons \u2014 e.g., a claim that legitimately synthesizes or reasonably infers across several evidence chunks rather than restating one of them. The system currently has no notion of multi-hop support; every claim is scored against its single best-matching evidence chunk(s) independently."),

      h1("7. Demo"),
      p("The accompanying demo (claim_verifier_demo.html) is a live, fully client-side implementation of the trained pipeline, not a lookup over precomputed results. The trained gradient-boosted classifier (200 trees, 12 features) was exported to JSON, and the entire retrieval-plus-scoring pipeline \u2014 sentence splitting, per-source TF-IDF indexing, feature computation, and tree-ensemble inference \u2014 was re-implemented in vanilla JavaScript so it runs directly in the browser with no server or API call. A person can paste any source document and any response into the page and get real, freshly computed per-sentence verdicts and evidence. Three RAGTruth examples are included as one-click presets: a weather Q&A, where the model correctly flags an invented statistic (\u201cwith an average annual temperature of 61\u00b0F,\u201d a figure absent from the source); a restaurant summary generated from structured Yelp data, where it flags a fabricated claim about music not being offered; and a news summary. The JS reimplementation was validated against the Python model on held-out feature vectors (identical output to 6 decimal places) and uses a lighter regex-based sentence splitter in place of NLTK, so sentence boundaries can differ slightly from the offline evaluation pipeline on unusual punctuation."),

      h1("8. Limitations and Future Work"),
      bullet("No pretrained NLI or embedding model. The largest single change likely to improve results is replacing the TF-IDF + lexical-feature stand-in with an actual pretrained NLI model (e.g., a DeBERTa-MNLI checkpoint) for the entailment step, and a sentence-embedding model for retrieval \u2014 both were unreachable in this sandboxed environment but would be the first thing to add given API/network access to a model hub."),
      bullet("Sentence-level granularity misses compound sentences that mix a true and a false claim in one clause; word- or span-level segmentation (as RAGTruth itself annotates) would be more precise but adds substantial complexity."),
      bullet("Single-chunk evidence scoring cannot verify claims that require combining two or more evidence chunks (multi-hop reasoning), a known source of false positives here."),
      bullet("The system is tuned and evaluated only on RAGTruth's three task types (QA, summarization, data-to-text); generalization to open-domain chat or multi-document RAG is untested."),
      bullet("As with any lexical-overlap method, it is more easily fooled by paraphrase than a semantic method would be \u2014 visible directly in the Summary task's low recall (0.15), since summaries paraphrase far more than QA or data-to-text responses do."),

      h1("9. Reproducibility"),
      p("All code, cached features, trained model, and result files (metrics.json, error_analysis.json, test_predictions.jsonl) accompany this report. Pipeline: src/data_prep.py (load + segment + label) \u2192 src/features.py (retrieval + feature functions) \u2192 src/build_dataset.py (featurize both splits) \u2192 src/train_eval.py (train, select, evaluate) \u2192 src/error_analysis.py. Source data: RAGTruth, https://github.com/ParticleMedia/RAGTruth (Apache-licensed)."),

      h1("References"),
      p("Niu, C., Wu, Y., Zhu, J., Xu, S., Shum, K., Zhong, R., Song, J., & Zhang, T. (2024). RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models. Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL 2024), 10862\u201310878.", { size: 19, spacingAfter: 100 }),
      p("Manakul, P., Liusie, A., & Gales, M. J. F. (2023). SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models. Proceedings of EMNLP 2023.", { size: 19, spacingAfter: 100 }),
      p("K\u00e1d\u00e1r, \u00c1., et al. LettuceDetect: A Hallucination Detection Framework for RAG Applications, built and evaluated on RAGTruth (arXiv:2502.17125).", { size: 19, spacingAfter: 100 }),
    ],
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync("/home/claude/project/writeup/report.docx", buf);
  console.log("wrote report.docx", buf.length, "bytes");
});
