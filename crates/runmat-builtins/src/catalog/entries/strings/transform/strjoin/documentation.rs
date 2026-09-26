use crate::*;
const SECTIONS: &[BuiltinDocumentationSection] = &[
BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
"The default delimiter is one space. Standard vector mode joins all vector elements into one string scalar, or one character row for cellstr input.",
"A text delimiter vector must have exactly one element per gap; scalar delimiters repeat. String or cell delimiter vectors are supported. Numeric input text is never implicitly formatted; standard vector mode also requires text delimiters.",
"The legacy row-wise extension additionally preserves scalar numeric, integer and logical delimiter formatting. Native integer scalars format exactly as decimal text, including uint64 maximum; this corrects the former sidecar claim that every numeric delimiter rejected. Integer tensors and provider-resident delimiters remain unsupported.",
"With RunMat extensions enabled, an explicit scalar delimiter and a string scalar, string array, or character array select the existing row-wise extension. String arrays join across columns, returning a rows-by-1 string array, including unchanged string column vectors.",
"In the row-wise extension, each character row is already one text element and is returned intact. Character rows ab and cd yield the strings ab and cd; the old sidecar example incorrectly showed a-b and c-d.",
"The row-wise extension keeps the existing empty 0-by-0 string-array result for zero rows or columns. Standard empty vector mode returns an empty string scalar or an empty character row.",
"Strict compatibility mode joins string column vectors into one scalar and rejects string matrices and direct character-array inputs. Cellstr input and omitted delimiters always use standard vector mode.",
"Inputs with non-singleton dimensions beyond two reject before any data can be truncated. Text resides on the host; there is no provider dispatch, gather or callback.",
] },
BuiltinDocumentationSection { heading: "Placement and effects", paragraphs: &[
"This portable host operation is pure apart from errors and never suspends. It performs no callbacks, workspace changes, filesystem, randomness, time services, provider hooks, or GPU fusion.",
"Interpreter and compiled calls share the default runtime binding. Zero requested outputs evaluates and discards the return value; more than one output is rejected before processing. Distributed input is unsupported. Device kernels and provider benchmarks do not apply to host-only text and symbolic values.",
] },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "Join a row vector of strings with commas",
        program: "labels = [\"alpha\" \"beta\" \"gamma\"];\nout = strjoin(labels, \", \")",
        display_output: Some("out = \"alpha, beta, gamma\""),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(out, \"alpha, beta, gamma\"));",
        },
    },
    BuiltinExample {
        id: "example-2",
        title: "Join each row of a string matrix independently",
        program:
            "names = [\"Ada\" \"Lovelace\"; \"Grace\" \"Hopper\"];\nout = strjoin(names, \" \")",
        display_output: Some("out = 2×1 string\n    \"Ada Lovelace\"\n    \"Grace Hopper\""),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(out, [\"Ada Lovelace\"; \"Grace Hopper\"]));",
        },
    },
    BuiltinExample {
        id: "example-3",
        title: "Join rows of a character array",
        program: "chars = ['ab'; 'cd'];\nout = strjoin(chars, \"-\")",
        display_output: Some("out = 2×1 string\n    \"ab\"\n    \"cd\""),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(out, [\"ab\"; \"cd\"]));",
        },
    },
    BuiltinExample {
        id: "example-4",
        title: "Join cellstr using the default delimiter",
        program: "out = strjoin({'one','two'})",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(ischar(out)); assert(strcmp(out,'one two'));",
        },
    },
    BuiltinExample {
        id: "example-5",
        title: "Use a different delimiter per gap",
        program: "out = strjoin([\"a\",\"b\",\"c\"], [\"-\",\":\"])",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(out,\"a-b:c\"));",
        },
    },
];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
 authority: BuiltinDocumentationAuthority::Catalog, title: Some("strjoin"), slug: Some("strjoin"),
 summary: "Join string vectors or cell vectors of character rows, with a RunMat row-wise extension.", description: "strjoin(text) joins a string vector or cell vector of character rows using spaces. strjoin(text, delimiter) supports scalar text or a delimiter vector with one element per gap. RunMat extension mode preserves established row-wise behavior for explicit scalar delimiters with string or character inputs.",
 keywords: &["strjoin","join strings","delimiter","row-wise join"], related: &["join","split","string","char"],
 sections: SECTIONS, examples: EXAMPLES, example_exemption: None,
 faqs: &[
 BuiltinDocumentationFaq { question: "Is the delimiter optional?", answer: "Yes. Omission joins a supported vector with spaces using standard vector semantics." },
BuiltinDocumentationFaq { question: "Does strjoin join by row or by column?", answer: "Standard vector mode joins all elements. The RunMat row-wise extension retains independent row joins for explicit scalar delimiters; strict compatibility disables that extension." },
BuiltinDocumentationFaq { question: "Can I pass a cell array of character vectors?", answer: "Yes, a cell vector of character rows is supported and returns a character row." },
 ],
 links: &[BuiltinDocumentationLink { label: "join", target: BuiltinDocumentationLinkTarget::Builtin("join") },
BuiltinDocumentationLink { label: "split", target: BuiltinDocumentationLinkTarget::Builtin("split") },
BuiltinDocumentationLink { label: "string", target: BuiltinDocumentationLinkTarget::Builtin("string") },
BuiltinDocumentationLink { label: "char", target: BuiltinDocumentationLinkTarget::Builtin("char") },
BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/transform/strjoin.rs") }], media: &[],
 evidence: BuiltinDocumentationEvidence {
 implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/transform/strjoin.rs") }], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "integration", location: "tests/strings/string_processing.m" },
BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Identity inference tests", location: "crates/runmat-builtins/src/catalog/entries/strings/transform/strjoin/inference.rs" }],
 notes: &["Legacy behaviors, syntax, examples, FAQs and links are reconciled here. Typed placement replaces GPU/fusion metadata. Example assertions verify semantic values rather than presentation. Integrated native, browser and documentation execution remains required; catalog metadata alone is not runtime evidence.",
"Sole-use implementation helpers formerly in crates/runmat-runtime/src/lib.rs moved into this identity. Incorrect MATLAB row-wise claims and character example output were corrected without changing established extension results."],
 },
 introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
