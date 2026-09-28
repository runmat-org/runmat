use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "Case-insensitive matching is the default; include `'matchcase'` when you need case-sensitive behaviour.",
        "With one output, `regexpi` returns a numeric row vector of 1-based match start indices.",
        "With multiple outputs, the default order is match starts, match ends, matched substrings.",
        "When the input is a string array or cell array of character vectors, outputs are cell arrays whose shape matches the input container.",
        "`'forceCellOutput'` forces cell outputs even for scalar inputs, matching MATLAB semantics.",
        "`'once'` limits each element to its first match, influencing every requested output.",
        "`'emptymatch','allow'` keeps zero-length matches; `'emptymatch','remove'` is the default filter.",
        "Named tokens (using `(?<name>...)`) return scalar struct values per match when `'names'` is requested. Unmatched names resolve to empty strings for MATLAB compatibility.",
        "Integer semantics: numeric match locations are outputs only. Numeric, logical, symbolic, and provider-resident inputs or options are rejected by the shared regular-expression admission boundary before gather or provider access.",
    ] },
    BuiltinDocumentationSection { heading: "Placement and effects", paragraphs: &[
        "Matching operates on host text. The builtin has no text GPU kernel or fusion form; numeric and resident text-role inputs reject before provider access. It may throw for malformed arguments and options, but does not mutate workspace state or access files, network, randomness, or time.",
        "An unused result is still evaluated and discarded. Ordinary host-text distribution is distinct from distributed-input support for this builtin.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1", title: "Finding indices regardless of case",
        program: "idx = regexpi('Abracadabra', 'a')", display_output: Some("idx =\n     1     4     6     8    11"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(idx, [1 4 6 8 11]));" },
    },
    BuiltinExample {
        id: "example-2", title: "Returning matched substrings ignoring case",
        program: "matches = regexpi('abcXYZ123', '[a-z]{3}', 'match')", display_output: Some("matches =\n  1×2 cell array\n    {'abc'}    {'XYZ'}"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(numel(matches), 2)); assert(strcmp(matches{1}, 'abc')); assert(strcmp(matches{2}, 'XYZ'));" },
    },
    BuiltinExample {
        id: "example-3", title: "Extracting capture tokens case-insensitively",
        program: "tokens = regexpi('ID:AB12', '(?<prefix>[a-z]+)(?<digits>\\d+)', 'tokens');\nfirst = tokens{1}{1};\nsecond = tokens{1}{2}", display_output: Some("first =\n    'AB'\nsecond =\n    '12'"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(first, 'AB')); assert(strcmp(second, '12'));" },
    },
    BuiltinExample {
        id: "example-4", title: "Limiting `regexpi` to the first match",
        program: "firstMatch = regexpi('aXaXaX', 'ax', 'match', 'once')", display_output: Some("firstMatch =\n    'aX'"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(firstMatch, 'aX'));" },
    },
    BuiltinExample {
        id: "example-5", title: "Splitting a string array without worrying about letter case",
        program: "parts = regexpi([\"Color:Red\"; \"COLOR:Blue\"], 'color:', 'split')", display_output: Some("parts =\n  2×1 cell array\n    {1×2 cell}\n    {1×2 cell}\n\nparts{2}{2}\nans =\n    'Blue'"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(parts), [2 1])); assert(strcmp(parts{2}{2}, 'Blue'));" },
    },
    BuiltinExample {
        id: "example-6", title: "Enforcing case-sensitive matches with `'matchcase'`",
        program: "idx = regexpi('CaseTest', 'case', 'matchcase')", display_output: Some("idx =\n     []"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isempty(idx));" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("regexpi"), slug: Some("regexpi"),
    summary: "Run case-insensitive regular-expression matching with MATLAB-compatible outputs and options.", description: "`regexpi(text, pattern)` evaluates regular-expression matches while ignoring case by default. It supports MATLAB-compatible outputs for indices, matches, capture tokens, token extents, named tokens, and split text, plus flags like `'once'`, `'emptymatch'`, and case/newline toggles.",
    keywords: &["regexpi", "regular expression", "ignore case", "pattern", "match"],
    related: &["regexp", "regexprep", "contains", "split", "strfind"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "How are the outputs ordered when I request several?", answer: "If you do not specify explicit output flags, the default order is match starts, match ends, and matched substrings—identical to MATLAB. Providing flags such as `'match'` or `'tokens'` returns only the requested outputs." },
        BuiltinDocumentationFaq { question: "Can I make `regexpi` behave like `regexp` with case sensitivity?", answer: "Yes. Include the `'matchcase'` flag to disable the default case-insensitive mode. You can also pass `'ignorecase'` explicitly to emphasise the default." },
        BuiltinDocumentationFaq { question: "Does `regexpi` support string arrays and cell arrays?", answer: "Yes. Outputs mirror the input container shape, and each element stores results for the corresponding string or character vector." },
        BuiltinDocumentationFaq { question: "How do zero-length matches behave?", answer: "By default (`'emptymatch','remove'`), zero-length matches are omitted. Use `'emptymatch','allow'` to keep them, which is helpful when inspecting optional pattern components." },
        BuiltinDocumentationFaq { question: "Does `regexpi` run on the GPU?", answer: "Text matching runs on the host and has no GPU text kernel. Numeric or resident values in text roles are rejected before provider access; supported scalar controls are handled separately." },
        BuiltinDocumentationFaq { question: "Are named tokens supported?", answer: "Yes. Use the `(?<name>...)` syntax and request the `'names'` output flag. Each match produces a scalar struct with fields for every named group." },
        BuiltinDocumentationFaq { question: "What happens with `'once'`?", answer: "`'once'` restricts each input element to the first match. All requested outputs honour that limit, returning scalars instead of per-match cells." },
        BuiltinDocumentationFaq { question: "Can I keep scalar outputs in cells?", answer: "Yes. Pass `'forceCellOutput'` to wrap even scalar results in cells, which is useful when writing code that must treat scalar and array inputs uniformly." }],
    links: &[BuiltinDocumentationLink { label: "regexp", target: BuiltinDocumentationLinkTarget::Builtin("regexp") },
        BuiltinDocumentationLink { label: "regexprep", target: BuiltinDocumentationLinkTarget::Builtin("regexprep") },
        BuiltinDocumentationLink { label: "contains", target: BuiltinDocumentationLinkTarget::Builtin("contains") },
        BuiltinDocumentationLink { label: "split", target: BuiltinDocumentationLinkTarget::Builtin("split") },
        BuiltinDocumentationLink { label: "strfind", target: BuiltinDocumentationLinkTarget::Builtin("strfind") },
        BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/regex/regexpi.rs") }], media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/regex/regexpi.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "runtime semantics", location: "builtins::strings::regex::regexpi::tests" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "integrated behavior", location: "builtins::strings::regex::regexpi::tests::regexpi_builtin_match_output" }],
        notes: &["Legacy behavior, examples, FAQs, links, source and tested fields are reconciled here. Examples have value assertions; integrated native and browser execution remains to be verified."],
    }, introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
