use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,

    title: Some("bagOfWords"),

    slug: Some("bagOfWords"),

    summary: "Create bag-of-words model objects.",

    description: "`bagOfWords(documents)` creates a lightweight RunMat `bagOfWords` object from a `tokenizedDocument` compatibility object. `bagOfWords(uniqueWords, counts)` creates a model from an explicit vocabulary and count matrix.",

    keywords: &["bagOfWords", "text analytics", "word counts", "vocabulary", "documents"],
    related: &["tokenizedDocument", "removeShortWords", "stopWords"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Supported forms",
            paragraphs: &["bag = bagOfWords\nbag = bagOfWords(documents)\nbag = bagOfWords(words)\nbag = bagOfWords(uniqueWords, counts)", "`bag` is a `bagOfWords` compatibility object.", "`documents` must be a RunMat `tokenizedDocument` object.", "`counts` must be a nonnegative integer-valued numeric matrix.", "The number of columns in `counts` must match the number of vocabulary words."],
        },
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "`bagOfWords` with no input creates an empty model.",
                "`bagOfWords(documents)` accepts RunMat `tokenizedDocument` objects created by `tokenizedDocument`.",
                "`bagOfWords(words)` accepts a word vector as a single-document model.",
                "`bagOfWords(uniqueWords, counts)` accepts a word vector and a nonnegative integer-valued count matrix whose columns match the vocabulary length. All eight integer classes are supported; values remain authoritative through exact validation and vocabulary-column filtering before one deliberate conversion into the double `Counts` property.",
                "`uniqueWords` must not contain duplicate nonmissing words. Missing string entries are ignored along with their corresponding count columns.",
                "The returned object stores `Vocabulary`, `Counts`, `NumWords`, and `NumDocuments` properties.",
                "Counts are stored as a dense host double tensor with one row per document and one column per vocabulary word. Very large dense count matrices are rejected with a controlled error until native sparse bag storage is implemented.",
                "GPU-resident inputs reject before provider access because `bagOfWords` exposes no interactive GPU-array input surface.",
                "Native sparse bag storage and sparse-preserving GPU residency remain outside this slice.",
                "Vocabulary order follows first token appearance for document input and the supplied order for explicit `uniqueWords` input.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Execution and residency",
            paragraphs: &["`bagOfWords` builds host text-model metadata and currently stores dense host double counts.", "`bagOfWords` stores host text-model metadata and dense host double counts in this slice; resident inputs reject before provider access."],
        },
        BuiltinDocumentationSection {
            heading: "Implementation boundary",
            paragraphs: &["Counts remain dense host double. The existing 50,000,000-cell guard is not native sparse bag storage or a performance guarantee for large corpora."],
        },
        BuiltinDocumentationSection {
            heading: "Finite host allocation",
            paragraphs: &["The current implementation requires an Unbounded host allocation policy. Under Managed finite allocation, a shared family guard rejects before gather, model cloning, mutation or filesystem work because complete text-model scratch and metadata ownership is not yet implemented. Authority is checked again after suspension and before publication, and failures are terminal rather than permission to replay. This refusal is an implementation frontier, not completion of finite resource support."],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "retained-1",
            title: "Create A Bag From Documents",
            program: "documents = tokenizedDocument([\"alpha beta\"; \"beta gamma\"]);\nbag = bagOfWords(documents)",
            display_output: Some("`bag.NumDocuments` is `2`, and `bag.Counts` has one column for each unique token."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(bag.NumDocuments == 2); assert(isequal(bag.Vocabulary, [\"alpha\" \"beta\" \"gamma\"])); assert(isequal(bag.Counts, [1 1 0; 0 1 1]));",
            },
        },
        BuiltinExample {
            id: "retained-2",
            title: "Create A Bag From Counts",
            program: "bag = bagOfWords([\"alpha\" \"beta\"], [2 1; 0 3])",
            display_output: Some("`bag.Vocabulary` is `[\"alpha\" \"beta\"]`, and `bag.Counts` is the supplied `2 x 2` count matrix."),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(isequal(bag.Vocabulary, [\"alpha\" \"beta\"])); assert(isequal(bag.Counts, [2 1; 0 3]));",
            },
        },
    ],
    example_exemption: None,

    faqs: &[
        BuiltinDocumentationFaq {
            question: "Does `bagOfWords` use sparse matrices?",
            answer: "Not in this slice. The compatibility object stores dense host counts and rejects very large dense count matrices. Sparse storage and provider-resident sparse paths remain broader runtime/model work.",
        },
        BuiltinDocumentationFaq {
            question: "Can `bagOfWords` consume raw document strings directly?",
            answer: "No. Raw document arrays should first be converted with `tokenizedDocument`. A raw word vector is accepted as a single-document model.",
        },
        BuiltinDocumentationFaq {
            question: "Does `bagOfWords` execute on the GPU?",
            answer: "No. It builds a host text-model object and has no `runmat-accelerate` provider path. Resident inputs reject rather than gathering.",
        },
    ],
    links: &[
        BuiltinDocumentationLink {
            label: "tokenizedDocument",
            target: BuiltinDocumentationLinkTarget::Builtin("tokenizedDocument"),
        },
        BuiltinDocumentationLink {
            label: "removeShortWords",
            target: BuiltinDocumentationLinkTarget::Builtin("removeShortWords"),
        },
        BuiltinDocumentationLink {
            label: "stopWords",
            target: BuiltinDocumentationLinkTarget::Builtin("stopWords"),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "`crates/runmat-runtime/src/builtins/strings/text_analytics/documents.rs`",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/text_analytics/documents.rs"),
        }],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "unit",
            location: "builtins::strings::text_analytics::documents::tests",
        }],
        notes: &[],
    },
    introduced: None,

    status: Some(BuiltinDocumentationStatus::Partial),
};
