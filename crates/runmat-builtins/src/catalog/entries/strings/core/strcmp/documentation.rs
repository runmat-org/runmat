use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`strcmp(A, B)` compares text exactly and returns a logical scalar or array. It accepts string scalars and arrays, character vectors and matrices, and cell arrays of character vectors.",
        "Each character-array row is one text element. Character matrices therefore produce column-shaped comparison results; scalar operands expand against compatible text arrays, including N-D string arrays.",
        "Missing string elements compare false, even against another missing element. Comparison is case-sensitive; use `strcmpi` to ignore letter case.",
        "Direct integer, numeric, symbolic, logical, and provider-resident operands are unsupported text and return scalar logical false before provider access or validation of the other operand. When neither operand takes that direct-false route, malformed host cell elements report an input error rather than being silently converted.",
    ] },
    BuiltinDocumentationSection { heading: "Placement and effects", paragraphs: &[
        "The default binding compares text on the host in native and browser runtimes. It has no GPU text kernel or fusion form. It may throw for invalid text containers or incompatible shapes, but has no callback, workspace, filesystem, randomness, network, or time effects.",
        "An unused output is evaluated and discarded; more than one output is invalid. Text comparison does not create or mutate device allocations. Ordinary distribution of host text values is separate from distributed-input support for this builtin.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "Compare equal string scalars",
        program: "tf = strcmp(\"RunMat\", \"RunMat\")",
        display_output: Some("tf = logical 1"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, true));",
        },
    },
    BuiltinExample {
        id: "example-2",
        title: "Compare a string array with a scalar",
        program: "names = [\"red\" \"green\" \"blue\"]; tf = strcmp(names, \"green\")",
        display_output: Some("tf = [false true false]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, logical([0 1 0])));",
        },
    },
    BuiltinExample {
        id: "example-3",
        title: "Compare character-array rows",
        program: "labels = char(\"cat\", \"dog\", \"cat\"); tf = strcmp(labels, \"cat\")",
        display_output: Some("tf = [true; false; true]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, logical([1; 0; 1])));",
        },
    },
    BuiltinExample {
        id: "example-4",
        title: "Compare cell arrays",
        program:
            "C1 = {'apple', 'pear', 'grape'}; C2 = {'apple', 'peach', 'grape'}; tf = strcmp(C1, C2)",
        display_output: Some("tf = [true false true]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, logical([1 0 1])));",
        },
    },
    BuiltinExample {
        id: "example-5",
        title: "Missing strings compare unequal",
        program: "vals = [\"alpha\" missing]; tf = strcmp(vals, \"alpha\")",
        display_output: Some("tf = [true false]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, logical([1 0])));",
        },
    },
    BuiltinExample {
        id: "example-6",
        title: "Expand a character matrix against a string row",
        program:
            "patterns = char(\"north\", \"south\"); tf = strcmp(patterns, [\"north\" \"east\"])",
        display_output: Some("tf = [true false; false false]"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, logical([1 0; 0 0])));",
        },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("strcmp"), slug: Some("strcmp"),
    summary: "Compare text inputs for exact case-sensitive equality with implicit expansion.",
    description: "`strcmp(A, B)` compares text elementwise, preserving text-container broadcasting and returning logical results.",
    keywords: &["strcmp", "string compare", "text equality", "cell array", "character vector"],
    related: &["string", "char", "contains", "startswith", "strlength", "compose", "num2str", "sprintf", "str2double", "strcmpi", "string.empty", "strings", "strncmp"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "What types can I pass to `strcmp`?", answer: "String arrays, character vectors or matrices, and cell arrays of character vectors. Compatible scalar operands expand against array operands." },
        BuiltinDocumentationFaq { question: "Does `strcmp` ignore letter case?", answer: "No. Use `strcmpi` for case-insensitive comparison." },
        BuiltinDocumentationFaq { question: "What happens when an input contains missing strings?", answer: "A missing string compares unequal to every text element, including another missing string." },
        BuiltinDocumentationFaq { question: "Can `strcmp` compare character matrices?", answer: "Yes. It compares rows as text elements and returns a column-shaped logical result." },
        BuiltinDocumentationFaq { question: "Is the result numeric or logical?", answer: "It is logical. Scalar comparisons return a logical scalar; array comparisons return a logical array." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "string", target: BuiltinDocumentationLinkTarget::Builtin("string") },
        BuiltinDocumentationLink { label: "char", target: BuiltinDocumentationLinkTarget::Builtin("char") },
        BuiltinDocumentationLink { label: "contains", target: BuiltinDocumentationLinkTarget::Builtin("contains") },
        BuiltinDocumentationLink { label: "startswith", target: BuiltinDocumentationLinkTarget::Builtin("startswith") },
        BuiltinDocumentationLink { label: "strlength", target: BuiltinDocumentationLinkTarget::Builtin("strlength") },
        BuiltinDocumentationLink { label: "compose", target: BuiltinDocumentationLinkTarget::Builtin("compose") },
        BuiltinDocumentationLink { label: "num2str", target: BuiltinDocumentationLinkTarget::Builtin("num2str") },
        BuiltinDocumentationLink { label: "sprintf", target: BuiltinDocumentationLinkTarget::Builtin("sprintf") },
        BuiltinDocumentationLink { label: "str2double", target: BuiltinDocumentationLinkTarget::Builtin("str2double") },
        BuiltinDocumentationLink { label: "strcmpi", target: BuiltinDocumentationLinkTarget::Builtin("strcmpi") },
        BuiltinDocumentationLink { label: "string.empty", target: BuiltinDocumentationLinkTarget::Builtin("string.empty") },
        BuiltinDocumentationLink { label: "strings", target: BuiltinDocumentationLinkTarget::Builtin("strings") },
        BuiltinDocumentationLink { label: "strncmp", target: BuiltinDocumentationLinkTarget::Builtin("strncmp") },
        BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/strcmp.rs") },
    ], media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/strcmp.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "comparison semantics", location: "builtins::strings::core::strcmp::tests" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "static inference", location: "crates/runmat-builtins/src/catalog/entries/strings/core/strcmp/inference.rs" }],
        notes: &["All substantive legacy sidecar behavior, six examples, five FAQs, thirteen related links, and source link are represented. Examples assert values and shapes rather than matching display text. Integrated native, browser, and documentation execution are still required."],
    }, introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
