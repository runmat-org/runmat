use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "The first argument must be scalar text: a string scalar or a character vector. Unlike `split`, `strsplit` does not accept string arrays, character matrices, or cell arrays of text.",
            "When you omit the delimiter, `strsplit` splits on MATLAB's default whitespace set (`' '`, `\\f`, `\\n`, `\\r`, `\\t`, `\\v`) and collapses consecutive delimiters.",
            "Explicit delimiters default to `'CollapseDelimiters', true`, so adjacent delimiters are grouped unless you opt out.",
            "Set `'DelimiterType', 'RegularExpression'` to interpret delimiters as regex patterns. In `'Simple'` mode, delimiters are matched literally and special regex characters are escaped.",
            "String input returns a row string array. Character-vector input returns a row cell array of text elements.",
            "Integer values have no text or option role and are not interpreted as delimiters or character codes.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Placement and effects",
        paragraphs: &[
            "Splitting executes on host text. Numeric and provider-resident text or delimiter values reject before provider access.",
            "This pure host operation may throw and may suspend at admitted gathers. It performs no callbacks, workspace/environment mutation, filesystem operations, randomness, or Clock reads. There is no GPU kernel, fusion route, or distributed support. VM and native calls share the canonical portable Runtime binding; these declarations are not execution evidence.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Current implementation frontier",
        paragraphs: &[
            "Character arrays with non-singleton dimensions beyond two, including those inside cells and delimiter controls, reject before row extraction. String and cell array geometry follows the per-entry behavior described here.",
            "The API accepts scalar text and returns row tokens, plus optional row delimiter matches. Exactly zero, one, or two outputs are admitted; no extra zero-valued outputs are fabricated. Numeric option values are rejected; CollapseDelimiters uses logical values. Managed text allocation remains incomplete.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "Split A Character Vector On Whitespace",
        program: "txt = 'The rain in Spain.';\nparts = strsplit(txt)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(parts,{'The','rain','in','Spain.'}));",
        },
    },
    BuiltinExample {
        id: "example-2",
        title: "Split Using A Literal Delimiter",
        program: "csv = \"alpha,beta,gamma\";\nparts = strsplit(csv, \",\")",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(parts,[\"alpha\",\"beta\",\"gamma\"]));",
        },
    },
    BuiltinExample {
        id: "example-3",
        title: "Return The Matched Delimiters",
        program: "[parts, matches] = strsplit(\"a,,b,\", \",\")",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(parts,[\"a\",\"b\",\"\"])); assert(isequal(matches,[\",,\",\",\"]));",
        },
    },
    BuiltinExample {
        id: "example-4",
        title: "Use A Regular Expression Delimiter",
        program: "[parts, matches] = strsplit(\"1.21m/s 1.985 m/s\", \"\\s*m/s\\s*\", \"DelimiterType\", \"RegularExpression\")",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(parts,[\"1.21\",\"1.985\",\"\"])); assert(isequal(matches,[\"m/s \",\" m/s\"]));",
        },
    },
    BuiltinExample {
        id: "example-5",
        title: "Preserve Empty Fields",
        program: "parts = strsplit(\"a,,b\", \",\", \"CollapseDelimiters\", false)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(parts,[\"a\",\"\",\"b\"]));",
        },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("strsplit"),
    slug: Some("strsplit"),
    summary: "Split scalar text into substrings using simple or regular-expression delimiters with MATLAB-compatible options.",
    description: "`strsplit(str)` splits string-scalar or character-vector text into substrings. It supports MATLAB-compatible delimiter modes, delimiter collapsing, and optional matched-delimiter outputs.",
    keywords: &["strsplit", "split string", "character vector split", "CollapseDelimiters", "DelimiterType", "matches"],
    related: &["split", "strjoin", "regexp", "join"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "How is `strsplit` different from `split`?",
            answer: "`strsplit` is the scalar-text API that mirrors MATLAB's older function shape, including optional `matches` output and regex delimiter mode. `split` is the vectorized string-array builtin.",
        },
        BuiltinDocumentationFaq {
            question: "What does the second output contain?",
            answer: "The optional `matches` output returns the delimiter text that produced each split. When delimiters are collapsed, adjacent delimiter runs are grouped into the matched text.",
        },
        BuiltinDocumentationFaq {
            question: "Can `strsplit` use regular expressions?",
            answer: "Yes. Pass `'DelimiterType', 'RegularExpression'` and provide delimiter patterns as regex fragments.",
        },
        BuiltinDocumentationFaq {
            question: "Does `strsplit` accept string arrays?",
            answer: "No. RunMat currently restricts `strsplit` to scalar text, matching MATLAB's documented scalar-input API.",
        },
    ],
    links: &[
        BuiltinDocumentationLink { label: "split", target: BuiltinDocumentationLinkTarget::Builtin("split") },
        BuiltinDocumentationLink { label: "strjoin", target: BuiltinDocumentationLinkTarget::Builtin("strjoin") },
        BuiltinDocumentationLink { label: "regexp", target: BuiltinDocumentationLinkTarget::Builtin("regexp") },
        BuiltinDocumentationLink { label: "join", target: BuiltinDocumentationLinkTarget::Builtin("join") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "`crates/runmat-runtime/src/builtins/strings/transform/split.rs`", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/transform/split.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Runtime text semantics", location: "crates/runmat-runtime/src/builtins/strings/transform/split.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed identity facts", location: "crates/runmat-builtins/src/catalog/entries/strings/transform/strsplit/inference.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Original unit source reference (not execution evidence)", location: "builtins::strings::transform::split::tests::strsplit_string_scalar_returns_string_array" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Original regex source reference (not execution evidence)", location: "builtins::strings::transform::split::tests::strsplit_regular_expression_mode" },
        ],
        notes: &["Original examples are preserved as standalone asserted programs except the impossible GPU text example, whose host equivalent is explicit. Historical display outputs were unverified presentation claims and are retired rather than invented as transcripts. Source-only migration requires centralized compiler, catalog, documentation, and backend qualification."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
