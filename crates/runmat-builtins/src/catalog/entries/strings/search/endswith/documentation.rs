use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "Accepts text inputs as string scalars/arrays, character vectors/arrays, or cell arrays of character vectors.",
        "Accepts patterns in the same formats. Multiple patterns form an alternatives set rather than an element-wise array paired with `str`.",
        "Typed missing string scalars (displayed as `<missing>`) never match any pattern.",
        "Empty patterns (`\"\"` or `''`) always match non-missing text elements.",
        "Typed missing pattern elements never match and therefore return `false`; literal `<missing>` patterns are ordinary text.",
        "Character arrays treat each row as an independent element; zero-row character arrays yield empty outputs.",
        "MATLAB-compatible mode accepts the documented `'IgnoreCase', logical` name-value form. RunMat mode additionally supports a positional flag and numeric or textual predicate values behind explicit extension gates.",
        "Returns a logical scalar for a scalar subject and otherwise returns a logical array with the shape of `str`.",
    ] },
    BuiltinDocumentationSection { heading: "Placement and effects", paragraphs: &[
        "Matching operates on host text. The builtin has no text GPU kernel or fusion form; numeric and resident text-role inputs reject before provider access. It may throw for malformed arguments and options, but does not mutate workspace state or access files, network, randomness, or time.",
        "An unused result is still evaluated and discarded. Ordinary host-text distribution is distinct from distributed-input support for this builtin.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1", title: "Check if a filename ends with an extension",
        program: "tf = endsWith(\"report.pdf\", \".pdf\")", display_output: Some("tf = logical\n   1"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(tf, true));" },
    },
    BuiltinExample {
        id: "example-2", title: "Perform a case-insensitive suffix test",
        program: "tf = endsWith(\"RunMat\", \"MAT\", 'IgnoreCase', true)", display_output: Some("tf = logical\n   1"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(tf, true));" },
    },
    BuiltinExample {
        id: "example-3", title: "Apply a scalar suffix to every element of a string array",
        program: "labels = [\"alpha\" \"beta\" \"gamma\"];\ntf = endsWith(labels, \"ma\")", display_output: Some("tf = 1×3 logical array\n   0   0   1"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(tf, logical([0 0 1])));" },
    },
    BuiltinExample {
        id: "example-4", title: "Test every subject against a set of suffixes",
        program: "names = [\"hydrogen\"; \"helium\"; \"lithium\"];\nsuffixes = [\"gen\"; \"ium\"];\ntf = endsWith(names, suffixes)", display_output: Some("tf = 3×1 logical array\n   1\n   1\n   1"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(tf, logical([1; 1; 1])));" },
    },
    BuiltinExample {
        id: "example-5", title: "Test suffixes for a cell array of character vectors",
        program: "C = {'Mercury', 'Venus', 'Mars'};\ntf = endsWith(C, 's')", display_output: Some("tf = 1×3 logical array\n   0   1   1"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(tf, logical([0 1 1])));" },
    },
    BuiltinExample {
        id: "example-6", title: "Provide multiple suffixes",
        program: "tf = endsWith(\"saturn\", [\"n\" \"x\" \"r\"])", display_output: Some("tf = logical\n   1"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(tf, true));" },
    },
    BuiltinExample {
        id: "example-7", title: "Handle empty patterns and missing values",
        program: "texts = [\"\", string(missing)];\ntf = endsWith(texts, \"\")", display_output: Some("tf = 1×2 logical array\n   1   0"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(tf, logical([1 0])));" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("endsWith"), slug: Some("endsWith"),
    summary: "Test whether text inputs end with patterns in MATLAB and RunMat.", description: "`endsWith(str, pattern)` returns one logical result for each element of `str`. When `pattern` has multiple elements, each element is an alternative and a subject matches if any pattern is a suffix.",
    keywords: &["endswith", "suffix", "string array", "ignorecase", "text search"],
    related: &["contains", "startsWith", "regexp", "regexpi", "strfind"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "What types can I pass to `endsWith`?", answer: "Use string arrays, character vectors, or cell arrays of character vectors for both arguments. Every pattern is tested as an alternative against each subject element." },
        BuiltinDocumentationFaq { question: "How do I ignore letter case?", answer: "MATLAB-compatible mode uses `'IgnoreCase', true` or false. RunMat mode additionally accepts textual values such as `'on'` and a positional flag behind explicit extension gates. The option name is case-insensitive." },
        BuiltinDocumentationFaq { question: "What happens with empty patterns?", answer: "Empty patterns (`\"\"` or `''`) always match non-missing text elements. When the text element is missing (`<missing>`), the result is `false`." },
        BuiltinDocumentationFaq { question: "Can I provide multiple suffixes at once?", answer: "Yes. Provide `pattern` as a string array, character vector, cell array of character vectors, or supported pattern object. A subject returns true when any supplied pattern is a suffix." },
        BuiltinDocumentationFaq { question: "How are missing strings treated?", answer: "Missing string scalars (displayed as `<missing>`) never match any pattern and produce `false` in the result. Use `ismissing` if you need to handle missing values separately." },
        BuiltinDocumentationFaq { question: "Does `endsWith` run on the GPU?", answer: "Text matching runs on the host and has no GPU text kernel. Numeric or resident values in text roles are rejected before provider access; supported scalar controls are handled separately." },
        BuiltinDocumentationFaq { question: "Does `endsWith` preserve the input shape?", answer: "Yes. The output has the shape of `str`; pattern-array shape does not change it. A scalar subject produces a logical scalar." }],
    links: &[BuiltinDocumentationLink { label: "MathWorks endsWith documentation", target: BuiltinDocumentationLinkTarget::Source("https://www.mathworks.com/help/matlab/ref/string.endswith.html") },
        BuiltinDocumentationLink { label: "`contains`", target: BuiltinDocumentationLinkTarget::Builtin("contains") },
        BuiltinDocumentationLink { label: "`startsWith`", target: BuiltinDocumentationLinkTarget::Builtin("startsWith") },
        BuiltinDocumentationLink { label: "`regexp`", target: BuiltinDocumentationLinkTarget::Builtin("regexp") },
        BuiltinDocumentationLink { label: "`regexpi`", target: BuiltinDocumentationLinkTarget::Builtin("regexpi") },
        BuiltinDocumentationLink { label: "strfind", target: BuiltinDocumentationLinkTarget::Builtin("strfind") },
        BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/search/endswith.rs") }], media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/search/endswith.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "runtime semantics", location: "builtins::strings::search::endswith::tests" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "integrated behavior", location: "builtins::strings::search::endswith::tests::endswith_cell_array_patterns" }],
        notes: &["Legacy behavior, examples, FAQs, links, source and tested fields are reconciled here. Examples have value assertions; integrated native and browser execution remains to be verified."],
    }, introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
