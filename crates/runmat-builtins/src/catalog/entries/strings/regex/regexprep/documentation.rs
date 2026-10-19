use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "With scalar text inputs, the result is a scalar of the same kind (`char` inputs stay `char`, string scalars stay string scalars).",
        "String arrays and cell arrays are processed element-wise and produce containers with matching shapes. When patterns and replacements are provided as arrays of the same size, they are paired element-by-element; otherwise, scalar patterns/replacements broadcast to every element.",
        "Cell or string arrays of patterns apply sequentially: each `(pattern, replacement)` pair is applied in order to every element.",
        "The `'once'` flag limits each element to the first match. Zero-length matches are skipped by default; `'emptymatch'` includes them, and `'noemptymatch'` restores the default. The older two-argument allow/remove form also works in RunMat.",
        "`'ignorecase'` and `'matchcase'` control case sensitivity. Dot (`.`) matches newlines by default; `'dotexceptnewline'` excludes them. `'lineanchors'` enables line-boundary matching for `^` and `$`.",
        "The Rust regex engine does not support MATLAB lookaround, backreferences, atomic groups, or dynamic expressions. Unsupported patterns report an error.",
        "`'preservecase'` adjusts the replacement text so the case pattern of the first match (all upper, all lower, or title case) is preserved.",
        "A positive integer scalar option `N` replaces only the Nth eligible occurrence and accepts every integer class through exact native validation.",
        "Integer values in the subject, pattern, or replacement text roles reject without implicit decimal formatting or provider access.",
    ] },
    BuiltinDocumentationSection { heading: "Placement and effects", paragraphs: &[
        "Replacement operates on the host and has no GPU regex kernel or fusion form. Native integer text roles reject before gathering; explicit GPU inputs require a RunMat compatibility extension, then gather and undergo normal text validation. It may throw for malformed arguments and options, but does not mutate workspace state or access files, network, randomness, or time.",
        "An unused result is still evaluated and discarded. Ordinary host-text distribution is distinct from distributed-input support for this builtin.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1", title: "Replacing vowels in a character vector",
        program: "clean = regexprep('abracadabra', '[aeiou]', 'X')", display_output: Some("clean =\n    'XbrXcXdXbrX'"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(clean, 'XbrXcXdXbrX'));" },
    },
    BuiltinExample {
        id: "example-2", title: "Applying multiple pattern/replacement pairs sequentially",
        program: "result = regexprep(\"Color: red\", {'Color', 'red'}, {'Shade', 'blue'})", display_output: Some("result = \"Shade: blue\""),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(result, \"Shade: blue\"));" },
    },
    BuiltinExample {
        id: "example-3", title: "Performing case-insensitive replacements",
        program: "names = regexprep([\"Alpha\"; \"beta\"; \"GaMmA\"], 'a', '_', 'ignorecase')", display_output: Some("names =\n  3×1 string array\n    \"_lph_\"\n    \"bet_\"\n    \"G_Mm_\""),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(names, [\"_lph_\"; \"bet_\"; \"G_Mm_\"]));" },
    },
    BuiltinExample {
        id: "example-4", title: "Preserving case of the original match",
        program: "words = regexprep('MATLAB and matlab', 'matlab', 'runmat', 'preservecase')", display_output: Some("words =\n    'RUNMAT and runmat'"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(words, 'RUNMAT and runmat'));" },
    },
    BuiltinExample {
        id: "example-5", title: "Limiting replacements to the first match with `'once'`",
        program: "out = regexprep(\"abababa\", 'ba', 'XY', 'once')", display_output: Some("out = \"aXYbaba\""),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(out, \"aXYbaba\"));" },
    },
    BuiltinExample {
        id: "example-6", title: "Using element-wise patterns for a string array",
        program: "expr = [\"foo\", \"bar\"];\npat = [\"f\", \"ar\"];\nrep = [\"F\", \"AR\"];\nexact = regexprep(expr, pat, rep)", display_output: Some("exact = 1×2 string\n    \"Foo\"    \"bAR\""),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(exact, [\"Foo\", \"bAR\"]));" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("regexprep"), slug: Some("regexprep"),
    summary: "Replace text with regular expressions across char, string, and cell-text inputs.", description: "`regexprep(str, pattern, replacement)` replaces regular-expression matches in character vectors, string arrays, and cell arrays of text. Common MATLAB container and option forms are supported; advanced MATLAB pattern syntax is not yet available.",
    keywords: &["regexprep", "regular expression", "replace", "substitute", "regex"],
    related: &["regexp", "regexpi", "strrep", "replace", "contains"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "How are container outputs shaped?", answer: "Outputs mirror the input container. Character vectors return character vectors, string arrays return string arrays with the same size, and cell arrays return cell arrays with matching shape." },
        BuiltinDocumentationFaq { question: "Can I supply multiple patterns at once?", answer: "Yes. Provide `pattern` and `replacement` as equally-sized cell or string arrays. Each pair is applied sequentially to every element, mirroring MATLAB's behaviour." },
        BuiltinDocumentationFaq { question: "What if my replacement needs capture-group text?", answer: "Use `$1`, `$2`, … for numbered groups or `${name}` for named groups inside the replacement text. These tokens expand to the corresponding captured substrings." },
        BuiltinDocumentationFaq { question: "Does `regexprep` support case-insensitive matching?", answer: "Yes. Pass the `'ignorecase'` option (or `'matchcase'` to revert). You can also request multiline anchors or dot behaviour changes with `'lineanchors'`, `'dotall'`, or `'dotExceptNewline'`." },
        BuiltinDocumentationFaq { question: "What does `'preservecase'` do?", answer: "When enabled, `regexprep` inspects the alphabetic characters in the matched text. If they are all uppercase, lowercase, or title-cased, the replacement text is adjusted to match that style." },
        BuiltinDocumentationFaq { question: "Does `regexprep` execute on the GPU?", answer: "No GPU regex kernel is used. Explicit GPU inputs require the RunMat fallback extension; accepted values gather for host replacement and outputs remain host text. Integer text inputs reject rather than being formatted implicitly." }],
    links: &[BuiltinDocumentationLink { label: "regexp", target: BuiltinDocumentationLinkTarget::Builtin("regexp") },
        BuiltinDocumentationLink { label: "regexpi", target: BuiltinDocumentationLinkTarget::Builtin("regexpi") },
        BuiltinDocumentationLink { label: "strrep", target: BuiltinDocumentationLinkTarget::Builtin("strrep") },
        BuiltinDocumentationLink { label: "replace", target: BuiltinDocumentationLinkTarget::Builtin("replace") },
        BuiltinDocumentationLink { label: "contains", target: BuiltinDocumentationLinkTarget::Builtin("contains") },
        BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/regex/regexprep.rs") }], media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/regex/regexprep.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "runtime semantics", location: "builtins::strings::regex::regexprep::tests" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "integrated behavior", location: "crates/runmat-vm/tests/integer_label_and_text_control_semantics.rs" }],
        notes: &["Legacy behavior, examples, FAQs, links, source and tested fields are reconciled here. Examples have value assertions; integrated native and browser execution remains to be verified."],
    }, introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
