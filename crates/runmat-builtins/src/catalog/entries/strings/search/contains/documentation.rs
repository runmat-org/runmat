use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "Accepts text inputs as string scalars/arrays, character scalars/arrays, or cell arrays of character vectors.",
        "Accepts patterns in the same formats. Either input may be scalar; when sizes differ MATLAB-style implicit expansion applies.",
        "Typed missing string scalars (displayed as `<missing>`) never match any pattern.",
        "Empty pattern values (`\"\"` or `''`) always match non-missing text elements.",
        "The optional `'IgnoreCase', true|false` name-value pair controls case sensitivity (default is case-sensitive).",
        "Returns a logical scalar when the broadcasted size is one element, otherwise returns a logical array with the broadcasted shape.",
    ] },
    BuiltinDocumentationSection { heading: "Placement and effects", paragraphs: &[
        "Matching operates on host text. The builtin has no text GPU kernel or fusion form; numeric and resident text-role inputs reject before provider access. It may throw for malformed arguments and options, but does not mutate workspace state or access files, network, randomness, or time.",
        "An unused result is still evaluated and discarded. Ordinary host-text distribution is distinct from distributed-input support for this builtin.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1", title: "Check if a string contains a substring",
        program: "tf = contains(\"RunMat Accelerate\", \"Accelerate\")", display_output: Some("tf = logical\n   1"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(tf, true));" },
    },
    BuiltinExample {
        id: "example-2", title: "Perform a case-insensitive search",
        program: "tf = contains(\"RunMat\", \"run\", 'IgnoreCase', true)", display_output: Some("tf = logical\n   1"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(tf, true));" },
    },
    BuiltinExample {
        id: "example-3", title: "Apply a scalar pattern to every element of a string array",
        program: "labels = [\"alpha\" \"beta\" \"gamma\"];\ntf = contains(labels, \"a\")", display_output: Some("tf = 1×3 logical array\n   1   1   1"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(tf, logical([1 1 1])));" },
    },
    BuiltinExample {
        id: "example-4", title: "Match element-wise patterns with implicit expansion",
        program: "names = [\"hydrogen\"; \"helium\"; \"lithium\"];\npatterns = [\"gen\"; \"ium\"; \"iron\"];\ntf = contains(names, patterns)", display_output: Some("tf = 3×1 logical array\n   1\n   1\n   0"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(tf, logical([1; 1; 0])));" },
    },
    BuiltinExample {
        id: "example-5", title: "Search a cell array of character vectors",
        program: "C = {'Mercury', 'Venus', 'Mars'};\ntf = contains(C, 'us')", display_output: Some("tf = 1×3 logical array\n   0   1   0"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(tf, logical([0 1 0])));" },
    },
    BuiltinExample {
        id: "example-6", title: "Provide multiple patterns as a column vector",
        program: "tf = contains(\"saturn\", ['s'; 'n'; 'x'])", display_output: Some("tf = 3×1 logical array\n   1\n   1\n   0"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(tf, logical([1; 1; 0])));" },
    },
    BuiltinExample {
        id: "example-7", title: "Handle empty and missing text values",
        program: "texts = [\"\", string(missing)];\ntf = contains(texts, \"\")", display_output: Some("tf = 1×2 logical array\n   1   0"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(tf, logical([1 0])));" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("contains"), slug: Some("contains"),
    summary: "Test whether text inputs contain patterns in MATLAB and RunMat.", description: "`contains(str, pattern)` returns logical results indicating whether each element of `str` contains the corresponding text in `pattern`. It supports string arrays, character arrays, and cell text inputs with implicit expansion behavior matching MATLAB and RunMat.",
    keywords: &["contains", "substring", "text search", "ignorecase", "string array"],
    related: &["regexp", "regexpi", "endsWith", "startsWith", "strfind"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "What types can I pass to `contains`?", answer: "Use string scalars/arrays, character vectors/arrays, or cell arrays of character vectors for both the text and the pattern. Mixed combinations are accepted, and RunMat performs MATLAB-style implicit expansion when the array sizes differ." },
        BuiltinDocumentationFaq { question: "How do I ignore letter case?", answer: "Specify `'IgnoreCase', true` after the pattern argument. The option name is case-insensitive and the default is `false`. Positional, numeric, and textual flag conveniences are available only in RunMat compatibility mode." },
        BuiltinDocumentationFaq { question: "What happens with empty patterns?", answer: "Empty patterns (`\"\"` or `''`) always match non-missing text elements. When the text element is missing (`<missing>`), the result is `false`." },
        BuiltinDocumentationFaq { question: "Can I search for multiple patterns at once?", answer: "Yes. Provide `pattern` as a string array, character array, or cell array of character vectors. RunMat applies implicit expansion so that scalar inputs expand across the other argument automatically." },
        BuiltinDocumentationFaq { question: "How are missing strings treated?", answer: "Missing string scalars (displayed as `<missing>`) never match any pattern and produce `false` in the result. Use `ismissing` if you need to separate missing values before calling `contains`." },
        BuiltinDocumentationFaq { question: "Does `contains` run on the GPU?", answer: "Text matching runs on the host and has no GPU text kernel. Numeric or resident values in text roles are rejected before provider access; supported scalar controls are handled separately." },
        BuiltinDocumentationFaq { question: "Does `contains` preserve the input shape?", answer: "Yes. The output is a logical array whose shape is the implicit-expansion result of the input shapes. When the broadcasted shape has exactly one element, the builtin returns a logical scalar." }],
    links: &[BuiltinDocumentationLink { label: "`regexp`", target: BuiltinDocumentationLinkTarget::Builtin("regexp") },
        BuiltinDocumentationLink { label: "`regexpi`", target: BuiltinDocumentationLinkTarget::Builtin("regexpi") },
        BuiltinDocumentationLink { label: "endsWith", target: BuiltinDocumentationLinkTarget::Builtin("endsWith") },
        BuiltinDocumentationLink { label: "startsWith", target: BuiltinDocumentationLinkTarget::Builtin("startsWith") },
        BuiltinDocumentationLink { label: "strfind", target: BuiltinDocumentationLinkTarget::Builtin("strfind") },
        BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/search/contains.rs") }], media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/search/contains.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "runtime semantics", location: "builtins::strings::search::contains::tests" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "integrated behavior", location: "builtins::strings::search::contains::tests::contains_cell_array_patterns" }],
        notes: &["Legacy behavior, examples, FAQs, links, source and tested fields are reconciled here. Examples have value assertions; integrated native and browser execution remains to be verified."],
    }, introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
