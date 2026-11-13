use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "Return true when text fully matches a pattern.",
            "Integer and other numeric inputs have no text or pattern role and reject without conversion. The current binding accepts exactly two inputs; the IgnoreCase option is not implemented.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Placement and effects",
        paragraphs: &[
            "Text and pattern inputs are host-only; numeric resident handles reject without implicit text conversion or provider access.",
            "This pure host operation may throw and may suspend at admitted gathers. It performs no callbacks, workspace/environment mutation, filesystem operations, randomness, or Clock reads. There is no GPU kernel, fusion route, or distributed support. VM and native calls share the canonical portable Runtime binding; these declarations are not execution evidence.",
            "Not eligible for fusion; this builtin materializes host text or logical results.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Current implementation frontier",
        paragraphs: &[
            "Character arrays with non-singleton dimensions beyond two, including those inside cells and delimiter controls, reject before row extraction. String and cell array geometry follows the per-entry behavior described here.",
            "The current binding accepts exactly text and one scalar text/pattern object. IgnoreCase and pattern-array overloads are not implemented. Matching is whole-text and case-sensitive; character rows each contribute one logical result. Typed missing string subjects produce false; literal `<missing>` text is ordinary text. Managed text allocation and independent MATLAB parity remain incomplete.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Syntax and argument roles",
        paragraphs: &[
            "tf = matches(text, pattern)",
            "text: Input text.",
            "pattern: Required scalar text or pattern object.",
            "tf: Logical result.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "example-1",
    title: "matches(\"A12\", regexpPattern(\"A\\d+\")) returns true.",
    program: "out = matches(\"A12\", regexpPattern(\"A\\d+\"));",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isequal(out,true));",
    },
}];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("matches"),
    slug: Some("matches"),
    summary: "Return true when text fully matches a pattern.",
    description: "tf = matches(text, pattern) - Return true when text fully matches a pattern.",
    keywords: &["matches", "strings", "search", "matlab", "compatibility"],
    related: &["string", "char"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[

    ],
    links: &[
        BuiltinDocumentationLink { label: "string", target: BuiltinDocumentationLinkTarget::Builtin("string") },
        BuiltinDocumentationLink { label: "char", target: BuiltinDocumentationLinkTarget::Builtin("char") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "`crates/runmat-runtime/src/builtins/strings/transform/compat.rs`", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/transform/compat.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Runtime text semantics", location: "crates/runmat-runtime/src/builtins/strings/transform/compat/tests.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed identity facts", location: "crates/runmat-builtins/src/catalog/entries/strings/search/matches/inference.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Original unit source reference (not execution evidence)", location: "builtins::strings::transform::compat::tests::splitlines_reverse_deblank_and_matches_work" },
        ],
        notes: &["Original examples are preserved as standalone asserted programs except the impossible GPU text example, whose host equivalent is explicit. Historical display outputs were unverified presentation claims and are retired rather than invented as transcripts. Source-only migration requires centralized compiler, catalog, documentation, and backend qualification."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
