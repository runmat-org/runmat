use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`strlength(str)` returns double character counts. String arrays preserve their element shape; a string scalar produces one double scalar. Character matrices produce one count per row in a column vector, while cell arrays of text scalars retain the cell-container shape.",
        "Character matrices formed from unequal rows use trailing spaces as row padding, and those padded spaces are excluded from the row count. Explicit trailing spaces in a single character vector remain part of its count.",
        "A missing string element produces `NaN`. Empty text produces zero lengths or an empty double array with the corresponding element shape. RunMat counts Unicode scalar values, not UTF-8 bytes; parity of every Unicode grapheme case with MATLAB is not asserted.",
        "Numeric and integer arrays are not interpreted as character codes. Convert to text first when the intended operation is to count formatted characters. Provider-resident numeric inputs reject before provider access.",
    ] },
    BuiltinDocumentationSection { heading: "Placement and effects", paragraphs: &[
        "The default binding measures host text in native and browser runtimes. No text GPU kernel, provider preparation, or fusion form exists. It may throw for nontext input or malformed cell elements but has no callback, workspace, filesystem, randomness, network, or time effect.",
        "An unused result is evaluated and discarded. More than one requested output is invalid. Distributed-input execution is unsupported, separately from ordinary transferability of host text values.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "example-1", title: "Count a string scalar", program: "len = strlength(\"RunMat\")", display_output: Some("len = 6"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(len, 6)); assert(strcmp(class(len), 'double'));" } },
    BuiltinExample { id: "example-2", title: "Count each string-array element", program: "labels = [\"North\" \"South\" \"East\" \"West\"]; counts = strlength(labels)", display_output: Some("counts = [5 5 4 4]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(counts, [5 5 4 4]));" } },
    BuiltinExample { id: "example-3", title: "Count character-array rows", program: "names = char(\"cat\", \"giraffe\"); row_counts = strlength(names)", display_output: Some("row_counts = [3; 7]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(row_counts, [3; 7]));" } },
    BuiltinExample { id: "example-4", title: "Count empty and blank strings", program: "mixed = [\"\" \"   \"]; len = strlength(mixed)", display_output: Some("len = [0 3]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(len, [0 3]));" } },
    BuiltinExample { id: "example-5", title: "Count cell-array text", program: "C = {'red', 'green', 'blue'}; L = strlength(C)", display_output: Some("L = [3 5 4]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(L, [3 5 4]));" } },
    BuiltinExample { id: "example-6", title: "Missing string length is NaN", program: "values = string([\"alpha\" \"beta\" \"gamma\"]); values(2) = string(missing); lengths = strlength(values)", display_output: Some("lengths = [5 NaN 5]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(lengths), [1 3])); assert(lengths(1) == 5 && isnan(lengths(2)) && lengths(3) == 5);" } },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog, title: Some("strlength"), slug: Some("strlength"),
    summary: "Return double character counts for each text element.",
    description: "`strlength(str)` counts characters in strings, character rows, and cell-array text elements while preserving their element shape.",
    keywords: &["strlength", "string length", "character count", "text analytics", "cell array"],
    related: &["string", "char", "strtrim", "length", "size", "compose", "num2str", "sprintf", "str2double", "strcmp", "strcmpi", "string.empty", "strings", "strncmp"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Which numeric class does `strlength` return?", answer: "Double, including when every count is a whole number." },
        BuiltinDocumentationFaq { question: "Why are some trailing spaces ignored?", answer: "Padding of shorter rows in a multirow character matrix is excluded. Explicit spaces in a one-row character vector remain counted." },
        BuiltinDocumentationFaq { question: "What happens with a missing string?", answer: "Its count is `NaN`. Use `ismissing` or `fillmissing` when a default is needed." },
        BuiltinDocumentationFaq { question: "Can I pass numeric data?", answer: "No. Convert numbers to text first with `num2str` or `string` if that is the intended operation." },
        BuiltinDocumentationFaq { question: "How does RunMat count Unicode text?", answer: "It counts Unicode scalar values rather than UTF-8 bytes. Combining sequences may contain more than one scalar value." },
        BuiltinDocumentationFaq { question: "Does this run on a GPU?", answer: "No. Text is host data; unsupported resident numeric input rejects before provider access." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "string", target: BuiltinDocumentationLinkTarget::Builtin("string") },
        BuiltinDocumentationLink { label: "char", target: BuiltinDocumentationLinkTarget::Builtin("char") },
        BuiltinDocumentationLink { label: "strtrim", target: BuiltinDocumentationLinkTarget::Builtin("strtrim") },
        BuiltinDocumentationLink { label: "length", target: BuiltinDocumentationLinkTarget::Builtin("length") },
        BuiltinDocumentationLink { label: "size", target: BuiltinDocumentationLinkTarget::Builtin("size") },
        BuiltinDocumentationLink { label: "compose", target: BuiltinDocumentationLinkTarget::Builtin("compose") },
        BuiltinDocumentationLink { label: "num2str", target: BuiltinDocumentationLinkTarget::Builtin("num2str") },
        BuiltinDocumentationLink { label: "sprintf", target: BuiltinDocumentationLinkTarget::Builtin("sprintf") },
        BuiltinDocumentationLink { label: "str2double", target: BuiltinDocumentationLinkTarget::Builtin("str2double") },
        BuiltinDocumentationLink { label: "strcmp", target: BuiltinDocumentationLinkTarget::Builtin("strcmp") },
        BuiltinDocumentationLink { label: "strcmpi", target: BuiltinDocumentationLinkTarget::Builtin("strcmpi") },
        BuiltinDocumentationLink { label: "string.empty", target: BuiltinDocumentationLinkTarget::Builtin("string.empty") },
        BuiltinDocumentationLink { label: "strings", target: BuiltinDocumentationLinkTarget::Builtin("strings") },
        BuiltinDocumentationLink { label: "strncmp", target: BuiltinDocumentationLinkTarget::Builtin("strncmp") },
        BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/strlength.rs") },
    ], media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/strlength.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "length semantics", location: "builtins::strings::core::strlength::tests" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "static inference", location: "crates/runmat-builtins/src/catalog/entries/strings/core/strlength/inference.rs" }],
        notes: &["Legacy sidecar behavior, six examples, six FAQs, fourteen related links and source link are retained or corrected. The old unconditional MATLAB Unicode-parity claim is narrowed to observed RunMat scalar-counting behavior. Assertions test values, class and shape; integrated execution remains required."],
    }, introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
