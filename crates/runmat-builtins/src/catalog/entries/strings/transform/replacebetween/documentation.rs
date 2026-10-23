use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "Replace text between positions or boundary markers.",
            "Numeric boundaries are one-based inclusive character positions and accept every integer class through exact native validation.",
            "The source defines the output text shape. Start, end, and replacement inputs must each be scalar or have exactly the same text-element shape as the source; they cannot expand a scalar source.",
            "Integer source or replacement values reject without implicit decimal formatting or provider access.",
            "Text and pattern boundaries replace the text between the matched delimiters and leave the delimiters in place.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Placement and effects",
        paragraphs: &[
            "No GPU kernel is registered. Automatic residency for numeric positions gathers transparently; explicit gpuArray host fallback is a declared RunMat extension.",
            "This pure host operation may throw and may suspend at admitted gathers. It performs no callbacks, workspace/environment mutation, filesystem operations, randomness, or Clock reads. There is no GPU kernel, fusion route, or distributed support. VM and native calls share the canonical portable Runtime binding; these declarations are not execution evidence.",
            "Not eligible for fusion; this builtin materializes host text or logical results.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Current implementation frontier",
        paragraphs: &[
            "Character arrays with non-singleton dimensions beyond two, including those inside cells and delimiter controls, reject before row extraction. String and cell array geometry follows the per-entry behavior described here.",
            "Exactly four inputs are required. Position boundaries are one-based inclusive and parsed exactly from every integer class; marker boundaries preserve the delimiters. The source text cannot be expanded by a nonscalar boundary/replacement. Explicit resident position fallback is a gated extension. Broader option forms, managed text allocation, and independent MATLAB parity remain incomplete.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Syntax and argument roles",
        paragraphs: &[
            "s = replaceBetween(text, start, end, newText)",
            "text: Input text.",
            "start, end, newText: Exactly three required controls; no extra options.",
            "out: Output value.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "example-1",
    title: "replaceBetween(\"a[old]b\", \"[\", \"]\", \"new\") returns \"a[new]b\".",
    program: "out = replaceBetween(\"a[old]b\", \"[\", \"]\", \"new\");",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isequal(out,\"a[new]b\"));",
    },
}];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("replaceBetween"),
    slug: Some("replacebetween"),
    summary: "Replace text between positions or boundary markers.",
    description: "s = replaceBetween(text, start, end, newText) - Replace text between positions or boundary markers.",
    keywords: &["replaceBetween", "strings", "transform", "matlab", "compatibility"],
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
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed identity facts", location: "crates/runmat-builtins/src/catalog/entries/strings/transform/replacebetween/inference.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Original unit source reference (not execution evidence)", location: "builtins::strings::transform::compat::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Original integration source reference (not execution evidence)", location: "crates/runmat-vm/tests/integer_label_and_text_control_semantics.rs" },
        ],
        notes: &["Original examples are preserved as standalone asserted programs except the impossible GPU text example, whose host equivalent is explicit. Historical display outputs were unverified presentation claims and are retired rather than invented as transcripts. Source-only migration requires centralized compiler, catalog, documentation, and backend qualification."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
