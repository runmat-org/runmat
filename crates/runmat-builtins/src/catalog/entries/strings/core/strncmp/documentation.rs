use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`strncmp(A, B, N)` compares at most N leading Unicode scalar values from each corresponding text element, case-sensitively. It accepts string scalars and arrays, character vectors and matrices, and cell arrays of character vectors. Scalar text expands against compatible arrays; character-matrix rows are independent text elements.",
        "N accepts real scalar double, single, logical, and all eight fixed-width integer classes. Native integer storage is read exactly. A negative integral N is treated as zero; zero makes every element comparison true, including missing strings. Unrepresentable host counts and nonintegral or nonfinite floating values reject; a floating count is not rounded to a nearby integer.",
        "For positive N, missing string elements compare false. If both text elements end before N and all compared characters matched, the result is true; an extra character on either side before N makes it false.",
        "Unsupported direct numeric or provider-resident operands in A or B return scalar logical false before N parsing or provider access. N may be a supported resident scalar and is gathered before exact validation; no string comparison runs on a GPU.",
    ] },
    BuiltinDocumentationSection { heading: "Placement and effects", paragraphs: &[
        "This default binding performs host text comparison in native and browser runtimes. Gathering a supported resident N may suspend before host validation; there is no GPU text kernel or fusion form. It has no callback, workspace, filesystem, randomness, network, or time effects. Invalid N or incompatible text shape can throw.",
        "Zero requested outputs evaluates and discards the result; more than one output is invalid. Ordinary worker transfer of host text is separate from unsupported distributed-input execution.",
    ] },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "Compare leading characters",
        program: "tf = strncmp(\"RunMat\", \"Runway\", 3)",
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
        title: "Expand a prefix across strings",
        program: "names = [\"north\" \"south\" \"east\"]; tf = strncmp(names, \"no\", 2)",
        display_output: Some("tf = [true false false]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, logical([1 0 0])));",
        },
    },
    BuiltinExample {
        id: "example-3",
        title: "Compare rows of a character matrix",
        program: "animals = char(\"cat\", \"camel\", \"cow\"); tf = strncmp(animals, \"ca\", 2)",
        display_output: Some("tf = [true; true; false]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, logical([1; 1; 0])));",
        },
    },
    BuiltinExample {
        id: "example-4",
        title: "Compare cell arrays",
        program:
            "C1 = {'red', 'green', 'blue'}; C2 = {'rose', 'grey', 'black'}; tf = strncmp(C1, C2, 2)",
        display_output: Some("tf = [false true true]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, logical([0 1 1])));",
        },
    },
    BuiltinExample {
        id: "example-5",
        title: "A zero-length comparison succeeds",
        program: "tf = strncmp(\"alpha\", \"omega\", 0)",
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
        id: "exact-integer-count-runmat",
        title: "Use an exact integer prefix count",
        program: "tf = strncmp(\"RunMat\", \"Runway\", uint64(3))",
        display_output: Some("tf = logical 1"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, true));",
        },
    },
];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("strncmp"), slug: Some("strncmp"),
    summary: "Compare up to N leading characters of text, case-sensitively.",
    description: "`strncmp(A, B, N)` returns logical prefix-comparison results with compatible text-container expansion.",
    keywords: &["strncmp", "string compare", "prefix", "text equality", "case-sensitive"],
    related: &["strcmp", "strcmpi", "contains", "startswith", "strlength", "char", "compose", "num2str", "sprintf", "str2double", "string.empty", "string", "strings"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "What types can I pass to `strncmp`?", answer: "A and B accept string arrays, character vectors or matrices, and cell arrays of character vectors. N accepts real integral scalar double, single, or fixed-width integer; negative values act as zero." },
        BuiltinDocumentationFaq { question: "Is comparison case-sensitive?", answer: "Yes. Use `strncmpi` to ignore letter case." },
        BuiltinDocumentationFaq { question: "What if N is zero?", answer: "Every element comparison is true because no characters are compared." },
        BuiltinDocumentationFaq { question: "What if a string ends before N?", answer: "It compares equal only if the other string also ends at that point after the same characters." },
        BuiltinDocumentationFaq { question: "How are missing strings treated?", answer: "For positive N, any comparison involving a missing string is false. For zero N, every comparison is true." },
        BuiltinDocumentationFaq { question: "What is the result class?", answer: "Logical; scalar inputs yield a logical scalar and expanded inputs yield a logical array in column-major order." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "strcmp", target: BuiltinDocumentationLinkTarget::Builtin("strcmp") },
        BuiltinDocumentationLink { label: "strcmpi", target: BuiltinDocumentationLinkTarget::Builtin("strcmpi") },
        BuiltinDocumentationLink { label: "contains", target: BuiltinDocumentationLinkTarget::Builtin("contains") },
        BuiltinDocumentationLink { label: "startswith", target: BuiltinDocumentationLinkTarget::Builtin("startswith") },
        BuiltinDocumentationLink { label: "strlength", target: BuiltinDocumentationLinkTarget::Builtin("strlength") },
        BuiltinDocumentationLink { label: "char", target: BuiltinDocumentationLinkTarget::Builtin("char") },
        BuiltinDocumentationLink { label: "compose", target: BuiltinDocumentationLinkTarget::Builtin("compose") },
        BuiltinDocumentationLink { label: "num2str", target: BuiltinDocumentationLinkTarget::Builtin("num2str") },
        BuiltinDocumentationLink { label: "sprintf", target: BuiltinDocumentationLinkTarget::Builtin("sprintf") },
        BuiltinDocumentationLink { label: "str2double", target: BuiltinDocumentationLinkTarget::Builtin("str2double") },
        BuiltinDocumentationLink { label: "string.empty", target: BuiltinDocumentationLinkTarget::Builtin("string.empty") },
        BuiltinDocumentationLink { label: "string", target: BuiltinDocumentationLinkTarget::Builtin("string") },
        BuiltinDocumentationLink { label: "strings", target: BuiltinDocumentationLinkTarget::Builtin("strings") },
        BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/strncmp.rs") },
    ], media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/strncmp.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "prefix semantics", location: "builtins::strings::core::strncmp::tests" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "static inference", location: "crates/runmat-builtins/src/catalog/entries/strings/core/strncmp/inference.rs" }],
        notes: &["The legacy sidecar cell example incorrectly expected red/rose to match at N=2; the assertion now records its actual false result and green/grey and blue/black true. The former blanket MATLAB Unicode-parity claim is narrowed to RunMat's observed Unicode-scalar comparison. Five prior examples, six FAQs, thirteen related links and the source link are retained; a RunMat exact-integer example was added. Integrated execution remains required."],
    }, introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
