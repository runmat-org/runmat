use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "Case-insensitive matching is the default; include `'matchcase'` when you need case-sensitive behaviour.",
        "With one output and no output selector, `regexpi` returns a numeric row vector of 1-based match start indices.",
        "Without output keywords, successive outputs are match starts, match ends, token extents, matched substrings, capture tokens, named tokens, and split text.",
        "When the input is a string array or cell array of character vectors, outputs are cell arrays whose shape matches the input container.",
        "`'forceCellOutput'` forces cell outputs even for scalar inputs, matching MATLAB semantics.",
        "`'once'` limits each element to its first match, influencing every requested output.",
        "Zero-length matches are omitted by default; `'emptymatch'` includes them and `'noemptymatch'` restores the default. RunMat also accepts its older two-argument allow/remove form.",
        "Dot (`.`) matches newlines by default. Pass `'dotexceptnewline'` to exclude them. The Rust regex engine does not support MATLAB lookaround, backreferences, atomic groups, or dynamic expressions.",
        "Named tokens (using `(?<name>...)`) produce a cell of scalar structs for scalar text when `'names'` is requested, or one struct with `'once'`. Array subjects add an outer cell matching the input shape. Unmatched names resolve to empty strings for MATLAB compatibility.",
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
        program: "tokens = regexpi('ID:AB12', '(?<prefix>[a-z]+)(?<digits>\\d+)', 'tokens');\nfirst = tokens{1}{1}\nsecond = tokens{1}{2}", display_output: Some("first =\n    'AB'\nsecond =\n    '12'"),
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
    summary: "Match text case-insensitively with regular expressions.", description: "`regexpi(text, pattern)` ignores case by default and can return indices, matches, capture tokens, token extents, named tokens, and split text. Common MATLAB output and option forms are supported; advanced MATLAB pattern syntax is not yet available.",
    keywords: &["regexpi", "regular expression", "ignore case", "pattern", "match"],
    related: &["regexp", "regexprep", "contains", "split", "strfind"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "How are the outputs ordered when I request several?", answer: "Without output keywords, the default order is match starts, match ends, token extents, matched substrings, capture tokens, named tokens, then split text. Output keywords such as `'match'` or `'tokens'` select only the requested outputs." },
        BuiltinDocumentationFaq { question: "Can I make `regexpi` behave like `regexp` with case sensitivity?", answer: "Yes. Include the `'matchcase'` flag to disable the default case-insensitive mode. You can also pass `'ignorecase'` explicitly to emphasise the default." },
        BuiltinDocumentationFaq { question: "Does `regexpi` support string arrays and cell arrays?", answer: "Yes. Outputs mirror the input container shape, and each element stores results for the corresponding string or character vector." },
        BuiltinDocumentationFaq { question: "How do zero-length matches behave?", answer: "Zero-length matches are omitted by default. Use `'emptymatch'` to include them or `'noemptymatch'` to restore the default. The older two-argument allow/remove form also works in RunMat." },
        BuiltinDocumentationFaq { question: "Does `regexpi` run on the GPU?", answer: "Text matching runs on the host and has no GPU text kernel. Numeric or resident values in text roles are rejected before provider access; supported scalar controls are handled separately." },
        BuiltinDocumentationFaq { question: "Are named tokens supported?", answer: "Yes. Use the `(?<name>...)` syntax and request the `'names'` output flag. Each match produces a scalar struct with fields for every named group." },
        BuiltinDocumentationFaq { question: "What happens with `'once'`?", answer: "`'once'` restricts each input element to the first match. For scalar text with a match, start/end indices, matched text, and named tokens become scalar values; capture tokens and splits remain cells, while token extents remain a matrix." },
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
    }, introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
