use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`strcmpi(A, B)` compares text elementwise without distinguishing letter case. It accepts string scalars and arrays, character vectors and matrices, and cell arrays of character vectors, including mixed text containers.",
        "Scalar operands expand against compatible text arrays. Each row of a character matrix is one text element and produces one comparison result. Non-scalar results are logical arrays with the broadcast shape.",
        "Missing string elements compare false even to other missing elements. Whitespace and text length remain significant. For case-sensitive equality, use `strcmp`; for leading-character comparison, use `strncmpi`.",
        "Unsupported direct numeric, integer, logical, symbolic, and provider-resident inputs return scalar logical false before provider access or validation of the other operand. Otherwise host cell containers are checked elementwise and malformed elements produce an input error.",
    ] },
    BuiltinDocumentationSection { heading: "Placement and effects", paragraphs: &[
        "Text comparison runs on the host in native and browser runtimes. There is no GPU text kernel or fusion form. The binding may throw on invalid containers or incompatible shapes, but has no callback, workspace, filesystem, randomness, network, or time effect.",
        "Zero requested outputs discards the evaluated result; more than one requested output is invalid. No device allocation is created or mutated. Distributed-input execution is unsupported; ordinary transfer of host text values is separate.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "Compare scalars ignoring case",
        program: "tf = strcmpi(\"RunMat\", \"runmat\")",
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
        title: "Search a string row",
        program: "colors = [\"Red\" \"GREEN\" \"blue\"]; mask = strcmpi(colors, \"green\")",
        display_output: Some("mask = [false true false]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(mask, logical([0 1 0])));",
        },
    },
    BuiltinExample {
        id: "example-3",
        title: "Compare character rows",
        program: "animals = char(\"Cat\", \"DOG\", \"cat\"); tf = strcmpi(animals, \"cAt\")",
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
            "C1 = {'north', 'East', 'SOUTH'}; C2 = {'NORTH', 'east', 'west'}; tf = strcmpi(C1, C2)",
        display_output: Some("tf = [true true false]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, logical([1 1 0])));",
        },
    },
    BuiltinExample {
        id: "example-5",
        title: "Compare a character matrix with a string row",
        program:
            "patterns = char(\"alpha\", \"DELTA\"); tf = strcmpi(patterns, [\"ALPHA\" \"delta\"])",
        display_output: Some("tf = [true false; false true]"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, logical([1 0; 0 1])));",
        },
    },
    BuiltinExample {
        id: "example-6",
        title: "Missing strings never compare equal",
        program: "values = [\"Active\" missing]; mask = strcmpi(values, \"active\")",
        display_output: Some("mask = [true false]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(mask, logical([1 0])));",
        },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("strcmpi"), slug: Some("strcmpi"),
    summary: "Compare text inputs for case-insensitive equality with implicit expansion.",
    description: "`strcmpi(A, B)` compares text values while ignoring letter case and returns an elementwise logical result.",
    keywords: &["strcmpi", "case insensitive compare", "string compare", "text equality", "cell array"],
    related: &["strcmp", "contains", "startswith", "endswith", "strip", "char", "compose", "num2str", "sprintf", "str2double", "string.empty", "string", "strings", "strlength", "strncmp"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Which text containers are accepted?", answer: "String arrays, character vectors and matrices, and cell arrays of character vectors. Mixed text containers and compatible scalar expansion are supported." },
        BuiltinDocumentationFaq { question: "Is whitespace significant?", answer: "Yes. Case is ignored, but trailing spaces and different lengths can make strings unequal. Use `strtrim` or `strip` when trimming is intended." },
        BuiltinDocumentationFaq { question: "Do missing strings compare equal?", answer: "No. Missing strings compare false even against another missing string." },
        BuiltinDocumentationFaq { question: "Can I compare complex or numeric inputs?", answer: "They are unsupported text operands; a direct numeric input returns scalar logical false instead of being converted to text." },
        BuiltinDocumentationFaq { question: "How are GPU inputs handled?", answer: "There is no GPU text representation. A direct provider-resident numeric input returns scalar logical false without gathering it or accessing the provider." },
        BuiltinDocumentationFaq { question: "What happens when both inputs are scalars?", answer: "The result is a logical scalar; array inputs produce logical arrays matching the broadcast dimensions." },
        BuiltinDocumentationFaq { question: "What does the `i` mean?", answer: "Insensitive to letter case. Use `strcmp` for case-sensitive equality and `strncmpi` to compare leading characters." },
        BuiltinDocumentationFaq { question: "How do cells and string arrays expand?", answer: "A scalar operand expands elementwise against a compatible array; arrays of the same shape compare element by element." },
        BuiltinDocumentationFaq { question: "Can character and string operands be mixed?", answer: "Yes. Character rows, string scalars, and cellstr values interoperate. For case-insensitive substring search, use `contains` with IgnoreCase or `regexpi`." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "strcmp", target: BuiltinDocumentationLinkTarget::Builtin("strcmp") },
        BuiltinDocumentationLink { label: "contains", target: BuiltinDocumentationLinkTarget::Builtin("contains") },
        BuiltinDocumentationLink { label: "startswith", target: BuiltinDocumentationLinkTarget::Builtin("startswith") },
        BuiltinDocumentationLink { label: "endswith", target: BuiltinDocumentationLinkTarget::Builtin("endswith") },
        BuiltinDocumentationLink { label: "strip", target: BuiltinDocumentationLinkTarget::Builtin("strip") },
        BuiltinDocumentationLink { label: "char", target: BuiltinDocumentationLinkTarget::Builtin("char") },
        BuiltinDocumentationLink { label: "compose", target: BuiltinDocumentationLinkTarget::Builtin("compose") },
        BuiltinDocumentationLink { label: "num2str", target: BuiltinDocumentationLinkTarget::Builtin("num2str") },
        BuiltinDocumentationLink { label: "sprintf", target: BuiltinDocumentationLinkTarget::Builtin("sprintf") },
        BuiltinDocumentationLink { label: "str2double", target: BuiltinDocumentationLinkTarget::Builtin("str2double") },
        BuiltinDocumentationLink { label: "string.empty", target: BuiltinDocumentationLinkTarget::Builtin("string.empty") },
        BuiltinDocumentationLink { label: "string", target: BuiltinDocumentationLinkTarget::Builtin("string") },
        BuiltinDocumentationLink { label: "strings", target: BuiltinDocumentationLinkTarget::Builtin("strings") },
        BuiltinDocumentationLink { label: "strlength", target: BuiltinDocumentationLinkTarget::Builtin("strlength") },
        BuiltinDocumentationLink { label: "strncmp", target: BuiltinDocumentationLinkTarget::Builtin("strncmp") },
        BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/strcmpi.rs") },
    ], media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/strcmpi.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "comparison semantics", location: "builtins::strings::core::strcmpi::tests" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "static inference", location: "crates/runmat-builtins/src/catalog/entries/strings/core/strcmpi/inference.rs" }],
        notes: &["All substantive legacy sidecar behavior, six examples, nine FAQs, fifteen related links, and source link are represented. Examples assert values and shapes rather than display text. Integrated native, browser, and documentation execution remain required."],
    }, introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
