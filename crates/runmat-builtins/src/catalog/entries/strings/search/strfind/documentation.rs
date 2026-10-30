use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "Accepts text inputs as string scalars/arrays, character vectors/arrays, or cell arrays of character vectors. Mixed combinations are permitted.",
        "Applies MATLAB-style implicit expansion when either argument is non-scalar.",
        "Returns a numeric row vector when both inputs are scalar text (character vectors or string scalars) and `'ForceCellOutput'` is `false`. If either input is a cell array or the broadcasted result contains multiple elements, the output is a cell array with the broadcasted shape, and each cell contains a row vector of double indices.",
        "Pattern matches are case-sensitive and may overlap. For example, `strfind(\"aaaa\",\"aa\")` yields `[1 2 3]`.",
        "An empty pattern returns an empty index vector. Typed missing strings (displayed as `<missing>`) never match any pattern.",
        "Specify `'ForceCellOutput', true` or double 1 to always obtain a cell array, even for scalar inputs.",
        "Typed integer `ForceCellOutput` values are available in RunMat mode behind a compatibility gate; MATLAB-compatible mode retains logical and double zero/one.",
    ] },
    BuiltinDocumentationSection { heading: "Placement and effects", paragraphs: &[
        "Matching operates on host text. The builtin has no text GPU kernel or fusion form; numeric and resident text-role inputs reject before provider access. It may throw for malformed arguments and options, but does not mutate workspace state or access files, network, randomness, or time.",
        "An unused result is still evaluated and discarded. Ordinary host-text distribution is distinct from distributed-input support for this builtin.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1", title: "Find substring positions in a character vector",
        program: "idx = strfind('abracadabra', 'abra')", display_output: Some("idx = [1 8]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(idx, [1 8]));" },
    },
    BuiltinExample {
        id: "example-2", title: "Locate overlapping matches",
        program: "idx = strfind(\"aaaa\", \"aa\")", display_output: Some("idx = [1 2 3]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(idx, [1 2 3]));" },
    },
    BuiltinExample {
        id: "example-3", title: "Return matches for each element of a string array",
        program: "words = [\"hydrogen\"; \"helium\"; \"lithium\"];\nidx = strfind(words, \"i\")", display_output: Some("idx = 3×1 cell array\n    {[]}\n    {[4]}\n    {[2 5]}"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(idx), [3 1])); assert(isempty(idx{1})); assert(isequal(idx{2}, 4)); assert(isequal(idx{3}, [2 5]));" },
    },
    BuiltinExample {
        id: "example-4", title: "Search with multiple patterns against one subject",
        program: "idx = strfind(\"saturn\", [\"sat\", \"turn\", \"moon\"])", display_output: Some("idx = 1×3 cell array\n    {[1]}    {[3]}    {[]}"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(idx), [1 3])); assert(isequal(idx{1}, 1)); assert(isequal(idx{2}, 3)); assert(isempty(idx{3}));" },
    },
    BuiltinExample {
        id: "example-5", title: "Force cell output for scalar inputs",
        program: "idx = strfind('mission', 's', 'ForceCellOutput', true)", display_output: Some("idx = 1×1 cell array\n    {[3 4]}"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(idx), [1 1])); assert(isequal(idx{1}, [3 4]));" },
    },
    BuiltinExample {
        id: "example-6", title: "Handle empty patterns and missing strings",
        program: "idxEmpty = strfind(\"abc\", \"\");\nidxMissing = strfind(string(missing), \"abc\")", display_output: Some("idxEmpty = [];\nidxMissing = []"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isempty(idxEmpty)); assert(isempty(idxMissing));" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("strfind"), slug: Some("strfind"),
    summary: "Locate starting indices of substring matches in text inputs with MATLAB-compatible output forms.", description: "`k = strfind(str, pattern)` returns starting positions of each occurrence of `pattern` within `str`. It follows MATLAB-compatible behavior across character vectors, strings, and cell arrays of character vectors.",
    keywords: &["strfind", "substring", "index", "positions", "forcecelloutput"],
    related: &["contains", "startsWith", "endsWith", "regexp"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "What types can I pass to `strfind`?", answer: "Use string scalars/arrays, character vectors/arrays, or cell arrays of character vectors for either argument. Mixed combinations are allowed; MATLAB-style implicit expansion aligns the inputs automatically." },
        BuiltinDocumentationFaq { question: "How are the results formatted?", answer: "When both inputs are scalar text (character vectors or string scalars) and `'ForceCellOutput'` is `false`, `strfind` returns a numeric row vector with the starting indices. In all other cases — for example, if either input is a cell array or the broadcasted size exceeds one element — the result is a cell array whose shape matches the broadcasted size, with each cell containing a row vector of doubles." },
        BuiltinDocumentationFaq { question: "Do matches overlap?", answer: "Yes. The builtin considers every occurrence of the pattern, including overlapping matches." },
        BuiltinDocumentationFaq { question: "What happens with empty patterns?", answer: "An empty pattern returns an empty index vector. A missing subject or pattern also produces no matches." },
        BuiltinDocumentationFaq { question: "How do I always get a cell array?", answer: "Supply `'ForceCellOutput', true`. This mirrors MATLAB and is useful when you want consistent cell outputs regardless of input sizes." },
        BuiltinDocumentationFaq { question: "Does `strfind` support case-insensitive matching?", answer: "No. `strfind` is case-sensitive like MATLAB. Use `contains`, `startsWith`, or the regular expression functions when you need case-insensitive searches." }],
    links: &[BuiltinDocumentationLink { label: "`contains`", target: BuiltinDocumentationLinkTarget::Builtin("contains") },
        BuiltinDocumentationLink { label: "`startsWith`", target: BuiltinDocumentationLinkTarget::Builtin("startsWith") },
        BuiltinDocumentationLink { label: "`endsWith`", target: BuiltinDocumentationLinkTarget::Builtin("endsWith") },
        BuiltinDocumentationLink { label: "`regexp`", target: BuiltinDocumentationLinkTarget::Builtin("regexp") },
        BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/search/strfind.rs") }], media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/search/strfind.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "runtime semantics", location: "builtins::strings::search::strfind::tests" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "integrated behavior", location: "integer_pattern_and_string_transform_semantics" }],
        notes: &["Legacy behavior, examples, FAQs, links, source and tested fields are reconciled here. Examples have value assertions; integrated native and browser execution remains to be verified."],
    }, introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
