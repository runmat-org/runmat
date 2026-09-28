use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "Single character vectors and string scalars return a numeric row vector of 1-based match start indices by default.",
        "String arrays and cell arrays always produce cell outputs that mirror the input shape, with each cell holding the result for the corresponding element.",
        "`'match'` returns matched substrings, `'tokens'` returns nested cells of capture-group substrings, `'tokenExtents'` returns `n × 2` double matrices with start/end indices for each token, `'names'` returns scalar struct values keyed by named tokens, and `'split'` yields the text segments between matches.",
        "`'once'` stops after the first match (per element), and every requested output honours that limit.",
        "`'emptymatch','remove'` (default) filters zero-length matches; `'emptymatch','allow'` keeps them so callers can observe optional patterns.",
        "`'forceCellOutput'` forces cell-array containers even for scalar inputs so downstream code can rely on uniform dimensions. MATLAB-compatible `'warnings','on'/'off'` flags are accepted but currently informational only.",
        "`'matchcase'` and `'ignorecase'` toggle case sensitivity, while `'lineanchors'` (`^`/`$`) and `'dotall'`/`'dotExceptNewline'` control how `.` interacts with newlines, mirroring MATLAB flags.",
        "Integer semantics: numeric match locations are outputs only. Numeric, logical, symbolic, and provider-resident inputs or options are rejected before gather or provider access.",
    ] },
    BuiltinDocumentationSection { heading: "Placement and effects", paragraphs: &[
        "Matching operates on host text. The builtin has no text GPU kernel or fusion form; numeric and resident text-role inputs reject before provider access. It may throw for malformed arguments and options, but does not mutate workspace state or access files, network, randomness, or time.",
        "An unused result is still evaluated and discarded. Ordinary host-text distribution is distinct from distributed-input support for this builtin.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1", title: "Find all 1-based match positions in a character vector",
        program: "idx = regexp('abracadabra', 'a')", display_output: Some("idx =\n     1     4     6     8    11"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(idx, [1 4 6 8 11]));" },
    },
    BuiltinExample {
        id: "example-2", title: "Return matched substrings using `'match'`",
        program: "matches = regexp('abc123xyz', '\\d+', 'match')", display_output: Some("matches =\n  1×1 cell array\n    {'123'}"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(numel(matches), 1)); assert(strcmp(matches{1}, '123'));" },
    },
    BuiltinExample {
        id: "example-3", title: "Extract capture tokens",
        program: "tokens = regexp('2024-03-14', '(\\d{4})-(\\d{2})-(\\d{2})', 'tokens');\nyear = tokens{1}{1};\nmonth = tokens{1}{2};\nday = tokens{1}{3}", display_output: Some("year =\n    '2024'\nmonth =\n    '03'\nday =\n    '14'"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(year, '2024')); assert(strcmp(month, '03')); assert(strcmp(day, '14'));" },
    },
    BuiltinExample {
        id: "example-4", title: "Split a string array around commas",
        program: "parts = regexp([\"a,b,c\"; \"1,2,3\"], ',', 'split')", display_output: Some("parts =\n  2×1 cell array\n    {1×3 cell}\n    {1×3 cell}"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(parts), [2 1])); assert(strcmp(parts{1}{1}, 'a')); assert(strcmp(parts{2}{3}, '3'));" },
    },
    BuiltinExample {
        id: "example-5", title: "Return only the first match with `'once'`",
        program: "first_idx = regexp('abababa', 'ba', 'once')", display_output: Some("first_idx =\n     2"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(first_idx, 2));" },
    },
    BuiltinExample {
        id: "example-6", title: "Work with named tokens",
        program: "matches = regexp('X=42; Y=7;', '(?<name>[A-Z])=(?<value>\\d+)', 'names');\nvalues = cellfun(@(s) str2double(s.value), matches)", display_output: Some("values =\n     42     7"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(values, [42 7]));" },
    },
    BuiltinExample {
        id: "example-7", title: "Keep zero-length matches with `'emptymatch','allow'`",
        program: "idx = regexp('aba', 'b*', 'emptymatch', 'allow')", display_output: Some("idx =\n     1     2     3     4"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(idx, [1 2 3 4]));" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("regexp"), slug: Some("regexp"),
    summary: "Match text with regular expressions and return indices, matches, tokens, names, or splits in MATLAB-compatible forms.", description: "`regexp(text, pattern)` searches character vectors, string arrays, and cell arrays of text and can return start/end indices, matched substrings, capture tokens, token extents, named tokens, or split text. Options such as `'once'` and `'emptymatch'` follow MATLAB-compatible behavior.",
    keywords: &["regexp", "regular expression", "pattern", "match", "split", "tokens"],
    related: &["regexpi", "regexprep", "contains", "split", "strfind"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "What outputs does `regexp` return by default?", answer: "With a single output argument, `regexp` returns a numeric row vector of 1-based match starts. When the call site asks for multiple outputs (e.g. `[startIdx, endIdx, matchStr] = regexp(...)`), RunMat returns match starts, match ends, and matched substrings in that order, just like MATLAB." },
        BuiltinDocumentationFaq { question: "How can I request tokens or splits instead of indices?", answer: "Specify the desired output types as string flags, for example `regexp(str, pat, 'match')`, `regexp(str, pat, 'tokens')`, or `regexp(str, pat, 'split')`. Multiple flags combine, so `regexp(str, pat, 'match', 'tokens')` returns both outputs." },
        BuiltinDocumentationFaq { question: "Does `regexp` support case-insensitive matching?", answer: "Yes. Use `'ignorecase'` (or call `regexpi`) to enable case-insensitive matching, and `'matchcase'` to revert to the default case-sensitive behaviour." },
        BuiltinDocumentationFaq { question: "How are string arrays and cell arrays handled?", answer: "For string arrays and cell arrays of char vectors, every output is a cell array whose shape matches the input. Each cell contains the result for the corresponding element, which mirrors MATLAB's container semantics." },
        BuiltinDocumentationFaq { question: "How do zero-length matches behave?", answer: "By default (`'emptymatch','remove'`), zero-length matches are filtered out so loops do not stall. Specify `'emptymatch','allow'` to keep them, matching MATLAB's `'emptymatch'` flag." },
        BuiltinDocumentationFaq { question: "Can I force cell output even for character vectors?", answer: "Yes. Pass `'forceCellOutput'` to force the outputs into cell arrays, which is useful when writing code that handles both scalar and array inputs uniformly." },
        BuiltinDocumentationFaq { question: "Does `regexp` run on the GPU?", answer: "Text matching runs on the host and has no GPU text kernel. Numeric or resident values in text roles are rejected before provider access; supported scalar controls are handled separately." },
        BuiltinDocumentationFaq { question: "What happens when I ask for more outputs than I requested via flags?", answer: "RunMat follows MATLAB's rules: if you do not supply explicit output flags, the default multi-output order is start indices, end indices, and matched substrings. Extra requested outputs beyond what you specified become numeric zeros." }],
    links: &[BuiltinDocumentationLink { label: "regexpi", target: BuiltinDocumentationLinkTarget::Builtin("regexpi") },
        BuiltinDocumentationLink { label: "regexprep", target: BuiltinDocumentationLinkTarget::Builtin("regexprep") },
        BuiltinDocumentationLink { label: "contains", target: BuiltinDocumentationLinkTarget::Builtin("contains") },
        BuiltinDocumentationLink { label: "split", target: BuiltinDocumentationLinkTarget::Builtin("split") },
        BuiltinDocumentationLink { label: "strfind", target: BuiltinDocumentationLinkTarget::Builtin("strfind") },
        BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/regex/regexp.rs") }], media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/regex/regexp.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "runtime semantics", location: "builtins::strings::regex::regexp::tests" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "integrated behavior", location: "builtins::strings::regex::regexp::tests::regexp_multi_output_default" }],
        notes: &["Legacy behavior, examples, FAQs, links, source and tested fields are reconciled here. Examples have value assertions; integrated native and browser execution remains to be verified."],
    }, introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
