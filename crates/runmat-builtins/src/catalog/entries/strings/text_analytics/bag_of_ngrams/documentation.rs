use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("bagOfNgrams"),
    slug: Some("bagOfNgrams"),
    summary: "Create bag-of-n-grams model objects.",
    description: "`bagOfNgrams(documents)` creates a lightweight RunMat `bagOfNgrams` object from a `tokenizedDocument` compatibility object. `bagOfNgrams(uniqueNgrams, counts)` creates a model from explicit n-grams and count matrix data.",
    keywords: &[
        "bagOfNgrams",
        "text analytics",
        "n-grams",
        "word counts",
        "tokenizedDocument",
    ],
    related: &[
        "tokenizedDocument",
        "bagOfWords",
        "removeShortWords",
    ],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Supported forms",
            paragraphs: &[
                "bag = bagOfNgrams\nbag = bagOfNgrams(documents)\nbag = bagOfNgrams(words)\nbag = bagOfNgrams(___, 'NgramLengths', lengths)\nbag = bagOfNgrams(uniqueNgrams, counts)",
                "`bag` is a `bagOfNgrams` compatibility object.",
                "`documents` must be a RunMat `tokenizedDocument` object.",
                "`lengths` must contain positive integer values.",
                "`counts` must be a nonnegative integer-valued numeric matrix.",
                "The number of columns in `counts` must match the number of rows in `uniqueNgrams`.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "`bagOfNgrams` with no input creates an empty model using default bigram length metadata.",
                "`bagOfNgrams(documents)` accepts RunMat `tokenizedDocument` objects created by `tokenizedDocument`.",
                "`bagOfNgrams(words)` accepts a word vector as one pre-tokenized document.",
                "`bagOfNgrams(___, 'NgramLengths', lengths)` accepts a positive integer scalar or vector of n-gram lengths. Duplicate length values are collapsed in first-seen order. For explicit `uniqueNgrams, counts` input, this option keeps only rows whose nonempty word count matches the requested lengths.",
                "`bagOfNgrams(uniqueNgrams, counts)` accepts a string matrix or text cell matrix whose rows are unique n-grams and a numeric count matrix whose columns match the n-gram rows.",
                "Explicit `counts` and `NgramLengths` accept all eight integer classes. Values remain authoritative through exact validation and filtering; counts then cross one deliberate conversion into the double `Counts` property, and lengths become bounded host indices before returning as double metadata.",
                "When `NgramLengths` is not supplied for explicit `uniqueNgrams, counts` input, RunMat infers the property from the nonmissing n-gram rows in first-seen order.",
                "Duplicate nonmissing n-grams raise `RunMat:bagOfNgrams:InvalidInput`. Rows containing missing strings are ignored along with their corresponding count columns.",
                "The returned object stores `Ngrams`, `Counts`, `NgramLengths`, `Vocabulary`, `NumNgrams`, and `NumDocuments` properties.",
                "N-gram rows are stored in `Ngrams`; shorter n-grams are padded with empty strings when mixed lengths are present.",
                "Counts are stored as a dense host tensor with one row per document and one column per n-gram. Very large dense count matrices are rejected with a controlled error until native sparse bag storage is implemented.",
                "GPU-resident counts and lengths reject before provider access because `bagOfNgrams` exposes no interactive GPU-array input surface.",
                "Native sparse bag storage, object-array parity, and the full Text Analytics model method surface remain broader Text Analytics work.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Execution and residency",
            paragraphs: &[
                "`bagOfNgrams` builds host text-model metadata and currently stores dense host double counts.",
                "`bagOfNgrams` stores host text-model metadata and dense host double counts in this slice; resident inputs reject before provider access.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "retained-1",
            title: "Create A Bigram Bag",
            program: "documents = tokenizedDocument([\"a b a\"; \"a b c\"]);\nbag = bagOfNgrams(documents)",
            display_output: Some("`bag.Ngrams` contains the bigrams `[\"a\" \"b\"]`, `[\"b\" \"a\"]`, and `[\"b\" \"c\"]`."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(bag.Ngrams, [\"a\" \"b\"; \"b\" \"a\"; \"b\" \"c\"])); assert(isequal(bag.Counts, [1 1 0; 1 0 1]));",
            },
        },
        BuiltinExample {
            id: "retained-2",
            title: "Use Multiple Ngram Lengths",
            program: "documents = tokenizedDocument(\"a b c\", \"TokenizeMethod\", \"unicode\");\nbag = bagOfNgrams(documents, \"NgramLengths\", [1 3])",
            display_output: Some("`bag.NgramLengths` is `[1 3]`, and the model contains unigrams plus the three-word n-gram."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(bag.NgramLengths, [1 3])); assert(bag.NumNgrams == 4); assert(isequal(bag.Counts, [1 1 1 1]));",
            },
        },
        BuiltinExample {
            id: "retained-3",
            title: "Create A Bag From Counts",
            program: "uniqueNgrams = [\"a\" \"b\"; \"b\" \"c\"];\nbag = bagOfNgrams(uniqueNgrams, [2 1; 0 3])",
            display_output: Some("`bag.Counts` is the supplied `2 x 2` count matrix."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(bag.Counts, [2 1; 0 3])); assert(isequal(bag.Ngrams, uniqueNgrams));",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Does `bagOfNgrams` use sparse matrices?",
            answer: "Not in this slice. The compatibility object stores dense host counts and rejects very large dense count matrices. Sparse storage and provider-resident sparse paths remain broader runtime/model work.",
        },
        BuiltinDocumentationFaq {
            question: "Can `bagOfNgrams` consume raw document strings directly?",
            answer: "No. Raw document arrays should first be converted with `tokenizedDocument`. A raw word vector is accepted as a single pre-tokenized document.",
        },
        BuiltinDocumentationFaq {
            question: "Does `bagOfNgrams` execute on the GPU?",
            answer: "No. It builds host text-model metadata and has no `runmat-accelerate` provider path. Resident inputs reject rather than gathering.",
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "tokenizedDocument",
            target: BuiltinDocumentationLinkTarget::Builtin("tokenizedDocument")
        },
        BuiltinDocumentationLink {
            label: "bagOfWords",
            target: BuiltinDocumentationLinkTarget::Builtin("bagOfWords")
        },
        BuiltinDocumentationLink {
            label: "removeShortWords",
            target: BuiltinDocumentationLinkTarget::Builtin("removeShortWords")
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink {
                label: "`crates/runmat-runtime/src/builtins/strings/text_analytics/ngrams.rs`",
                target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/text_analytics/ngrams.rs")
            },
        ],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "unit",
                location: "builtins::strings::text_analytics::ngrams::tests",
            },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
