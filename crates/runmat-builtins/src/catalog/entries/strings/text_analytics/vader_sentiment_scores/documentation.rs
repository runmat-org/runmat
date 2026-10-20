use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Supported forms", paragraphs: &[
        "compoundScores = vaderSentimentScores(documents)\ncompoundScores = vaderSentimentScores(documents, Name, Value)\n[compoundScores,positiveScores,negativeScores,neutralScores] = vaderSentimentScores(___)",
        "Numeric column vectors with one score per input document.",
        "`SentimentLexicon` must be a table with `Token` and `SentimentScore` variables.",
        "`Token` entries in a custom lexicon must be nonempty lowercase strings.",
        "`SentimentScore` entries must be finite single, double, or native integer values in [-4, 4].",
        "`Boosters`, `Dampeners`, and `Negations` must contain at least one word or n-gram when supplied.",
        "Only English `tokenizedDocument` inputs are accepted.",
    ] },
    BuiltinDocumentationSection { heading: "Inputs and sentiment rules", paragraphs: &[
        "vaderSentimentScores accepts an English RunMat tokenizedDocument. One output is the compound score; four requested outputs additionally return positive, negative and neutral proportions. Each is a host double column vector with one entry per document.",
        "Compound scores lie in [-1,1] and apply booster, dampener, negation, capitalization and punctuation modifiers to token sentiment. The compact default English lexicon covers common scripts; domain-specific or full-coverage scoring should pass a SentimentLexicon table. Exact MATLAB VADER scoring parity remains open.",
        "SentimentLexicon requires Token and SentimentScore variables. Tokens are nonempty lowercase strings. Scores are finite single, double or any native integer in [-4,4]; integer scores are range-checked exactly before floating computation.",
        "Boosters and Dampeners accept string, character, cell or string-array word lists. Column vectors are word lists; multi-column string arrays are row-wise n-grams with empty padding ignored. Negations accepts the same text forms as word lists. Supplied modifier lists must be nonempty.",
    ] },
    BuiltinDocumentationSection { heading: "Execution", paragraphs: &[
        "Text and object processing is host-only. Supported resident payloads within the supplied objects may gather through their owner; there is no provider scoring kernel.",
    ] },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "score-documents", title: "Score tokenized documents", program: "documents = tokenizedDocument([\"The book was VERY good!!!!\"; \"The book was not very good.\"]); compoundScores = vaderSentimentScores(documents)", display_output: Some("A double column vector with one score per document."), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(compoundScores), [2 1]));" } },
    BuiltinExample { id: "four-scores", title: "Return sentiment proportions", program: "documents = tokenizedDocument([\"good\"; \"bad\"]); [compoundScores,positiveScores,negativeScores,neutralScores] = vaderSentimentScores(documents)", display_output: Some("Four double column vectors."), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(compoundScores), size(positiveScores))); assert(isequal(size(negativeScores), size(neutralScores)));" } },
    BuiltinExample { id: "custom-lexicon", title: "Use a custom lexicon", program: "documents = tokenizedDocument(\"innovative risk\"); tbl = table([\"innovative\"; \"risk\"], [4; -3], \"VariableNames\", [\"Token\", \"SentimentScore\"]); compoundScores = vaderSentimentScores(documents, \"SentimentLexicon\", tbl)", display_output: Some("The custom table replaces the compact default lexicon."), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(compoundScores), [1 1]));" } },
];
const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/text_analytics/sentiment.rs") };
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("vaderSentimentScores"), slug: Some("vaderSentimentScores"), summary: "Score English tokenized documents with VADER-style rules.",
    description: "vaderSentimentScores returns compound and optional positive, negative and neutral host scores for English tokenizedDocument inputs.",
    keywords: &["vaderSentimentScores", "VADER", "sentiment", "tokenizedDocument", "text analytics"], related: &["tokenizedDocument", "tokenDetails"], sections: SECTIONS, examples: EXAMPLES, example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does RunMat include MATLAB's full VADER lexicon?", answer: "Not yet. It includes a compact default lexicon and supports custom SentimentLexicon tables." },
        BuiltinDocumentationFaq { question: "Can I provide modifier lists?", answer: "Yes. Boosters, Dampeners and Negations are applied to sentiment-bearing tokens." },
        BuiltinDocumentationFaq { question: "Does scoring execute on the GPU?", answer: "No. Scoring processes host text/object data and returns host double vectors." },
    ], links: &[BuiltinDocumentationLink { label: "tokenizedDocument", target: BuiltinDocumentationLinkTarget::Builtin("tokenizedDocument") }, SOURCE], media: &[],
    evidence: BuiltinDocumentationEvidence { implementation: &[SOURCE], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "sentiment options and scores", location: "builtins::strings::text_analytics::sentiment::tests" }], notes: &["The compact default lexicon and rules do not yet guarantee full MATLAB VADER scoring parity."] }, introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
