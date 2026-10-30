use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Text and controls", paragraphs: &[
        "strip(text) removes leading and trailing whitespace by default. Direction is case-insensitive: 'left' removes from the beginning, 'right' from the end, and 'both' from both sides. RunMat extension mode also accepts 'leading' and 'trailing' as aliases.",
        "Supply a scalar stripCharacters value to remove a custom character instead of whitespace. RunMat extension mode also accepts multi-character sets and shape-matched string or cell arrays of per-element sets.",
        "String scalars and arrays preserve their shape and typed missing elements. Character arrays are stripped row by row; the row count stays fixed and shorter rows are space-padded to the longest resulting width. Cell arrays preserve their layout and contain string scalars or character vectors.",
    ] },
    BuiltinDocumentationSection { heading: "Placement and errors", paragraphs: &[
        "strip is a host text operation and is a fusion boundary. Numeric, logical, symbolic, and provider-resident inputs are not interpreted as characters and reject before provider access. A textual control or nested cell cannot make a numeric or resident value into a supported text input.",
        "Invalid direction, strip-character type, per-element shape, or cell element raises a typed error. The operation has no provider kernel, callbacks, or external resource dependency.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "legacy-scalar", title: "Remove surrounding spaces", program: "name = \"   RunMat   \";\nclean = strip(name)", display_output: Some("clean = \"RunMat\""), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(clean, \"RunMat\"));" } },
    BuiltinExample { id: "legacy-right", title: "Strip only the right side", program: "labels = [\"   Alpha   \"; \"   Beta     \"; \"   Gamma    \"];\nright_stripped = strip(labels, 'right')", display_output: Some("right_stripped = [\"   Alpha\"; \"   Beta\"; \"   Gamma\"]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(right_stripped, [\"   Alpha\"; \"   Beta\"; \"   Gamma\"]));" } },
    BuiltinExample { id: "legacy-zeroes", title: "Remove leading zeroes", program: "codes = [\"00095\"; \"00137\"; \"00420\"];\nnumeric = strip(codes, 'left', '0')", display_output: Some("numeric = [\"95\"; \"137\"; \"420\"]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(numeric, [\"95\"; \"137\"; \"420\"]));" } },
    BuiltinExample { id: "legacy-char", title: "Keep character rows rectangular", program: "animals = char(\"   cat  \", \" dog\", \"cow   \");\ntrimmed = strip(animals)", display_output: Some("trimmed = ['cat'; 'dog'; 'cow']"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(ischar(trimmed)); assert(isequal(size(trimmed), [3 3])); assert(strcmp(trimmed(1,:), 'cat')); assert(strcmp(trimmed(2,:), 'dog')); assert(strcmp(trimmed(3,:), 'cow'));" } },
    BuiltinExample { id: "legacy-per-element", title: "Use one character set per element", program: "metrics = [\"##pass##\"; \"--warn--\"; \"**fail**\"];\nper_char = [\"#\"; \"-\"; \"*\"];\nnormalized = strip(metrics, 'both', per_char)", display_output: Some("normalized = [\"pass\"; \"warn\"; \"fail\"]"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(normalized, [\"pass\"; \"warn\"; \"fail\"]));" } },
    BuiltinExample { id: "legacy-cell", title: "Strip a mixed text cell", program: "pieces = {'  GPU  ', \"   Accelerate\", 'RunMat   '};\nout = strip(pieces)", display_output: Some("out = {'GPU', \"Accelerate\", 'RunMat'}"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(out), [1 3])); assert(strcmp(out{1}, 'GPU')); assert(isequal(out{2}, \"Accelerate\")); assert(strcmp(out{3}, 'RunMat'));" } },
];

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/transform/strip.rs"),
};
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("strip"), slug: Some("strip"),
    summary: "Remove leading and trailing characters from text with side and character-set controls.",
    description: "strip(text) trims whitespace by default; optional direction and stripCharacters controls select a side and custom character set for supported text containers.",
    keywords: &["strip", "trim", "whitespace", "leading characters", "trailing characters", "character arrays"],
    related: &["lower", "upper", "string", "char", "compose", "erase", "eraseBetween", "extractBetween", "join", "pad", "replace", "split", "strcat", "strrep", "strtrim"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Which direction keywords are supported?", answer: "'left', 'right', and 'both' are the standard directions. RunMat extension mode also accepts 'leading' and 'trailing'. Matching is case-insensitive." },
        BuiltinDocumentationFaq { question: "How can I remove characters other than whitespace?", answer: "Pass one strip character in the standard mode, for example strip(text, 'x'). RunMat extension mode also accepts a set such as 'xyz'; every matching character is then removed from the chosen edge." },
        BuiltinDocumentationFaq { question: "Can each element use different strip characters?", answer: "In RunMat extension mode, a string or cell character-set array can match the input shape; each element then uses its corresponding set." },
        BuiltinDocumentationFaq { question: "What happens to missing strings?", answer: "Typed missing subject elements remain missing. Literal `<missing>` text is ordinary text; missing direction or character controls are rejected." },
        BuiltinDocumentationFaq { question: "Does strip change character-array shape?", answer: "Only width may change. Rows are stripped independently and padded to the longest resulting row." },
        BuiltinDocumentationFaq { question: "Does strip execute on the GPU?", answer: "No. It operates on host text; provider-resident numeric values are not automatically reinterpreted as text." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "lower", target: BuiltinDocumentationLinkTarget::Builtin("lower") },
        BuiltinDocumentationLink { label: "upper", target: BuiltinDocumentationLinkTarget::Builtin("upper") },
        BuiltinDocumentationLink { label: "string", target: BuiltinDocumentationLinkTarget::Builtin("string") },
        BuiltinDocumentationLink { label: "char", target: BuiltinDocumentationLinkTarget::Builtin("char") },
        BuiltinDocumentationLink { label: "compose", target: BuiltinDocumentationLinkTarget::Builtin("compose") },
        BuiltinDocumentationLink { label: "erase", target: BuiltinDocumentationLinkTarget::Builtin("erase") },
        BuiltinDocumentationLink { label: "eraseBetween", target: BuiltinDocumentationLinkTarget::Builtin("eraseBetween") },
        BuiltinDocumentationLink { label: "extractBetween", target: BuiltinDocumentationLinkTarget::Builtin("extractBetween") },
        BuiltinDocumentationLink { label: "join", target: BuiltinDocumentationLinkTarget::Builtin("join") },
        BuiltinDocumentationLink { label: "pad", target: BuiltinDocumentationLinkTarget::Builtin("pad") },
        BuiltinDocumentationLink { label: "replace", target: BuiltinDocumentationLinkTarget::Builtin("replace") },
        BuiltinDocumentationLink { label: "split", target: BuiltinDocumentationLinkTarget::Builtin("split") },
        BuiltinDocumentationLink { label: "strcat", target: BuiltinDocumentationLinkTarget::Builtin("strcat") },
        BuiltinDocumentationLink { label: "strrep", target: BuiltinDocumentationLinkTarget::Builtin("strrep") },
        BuiltinDocumentationLink { label: "strtrim", target: BuiltinDocumentationLinkTarget::Builtin("strtrim") },
        SOURCE,
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[SOURCE],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Retained strip verification", location: "builtins::strings::transform::strip::tests::strip_cell_array_mixed_content" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "container, direction and character-set semantics", location: "builtins::strings::transform::strip::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "typed text facts", location: "crates/runmat-builtins/src/catalog/entries/strings/transform/strip/inference.rs" },
        ],
        notes: &["The older sidecar described automatic gathering of device text, but this runtime rejects resident numeric inputs before provider access; no provider text kernel is implemented."],
    },
    introduced: None, status: Some(BuiltinDocumentationStatus::Partial),
};
