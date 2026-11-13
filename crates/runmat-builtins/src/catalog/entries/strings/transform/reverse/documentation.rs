use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "Reverse characters in text values.",
            "Integer and other numeric inputs have no semantic role and reject without conversion; this includes resident numeric handles.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Placement and effects",
        paragraphs: &[
            "Text is host-only; numeric resident handles reject without implicit text conversion or provider access.",
            "This pure host operation may throw and may suspend at admitted gathers. It performs no callbacks, workspace/environment mutation, filesystem operations, randomness, or Clock reads. There is no GPU kernel, fusion route, or distributed support. VM and native calls share the canonical portable Runtime binding; these declarations are not execution evidence.",
            "Not eligible for fusion; this builtin materializes host text or logical results.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Current implementation frontier",
        paragraphs: &[
            "Character arrays with non-singleton dimensions beyond two, including those inside cells and delimiter controls, reject before row extraction. String and cell array geometry follows the per-entry behavior described here.",
            "Unicode scalar values are reversed independently in each character row or string element; recursive cells preserve shape. This is not grapheme-cluster reversal. Managed text allocation, full missing-value representation, and independent MATLAB parity remain incomplete.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Syntax and argument roles",
        paragraphs: &[
            "s = reverse(text)",
            "text: Input text.",
            "out: Output value.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "example-1",
    title: "reverse(\"abc\") returns \"cba\".",
    program: "out = reverse(\"abc\");",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isequal(out,\"cba\"));",
    },
}];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("reverse"),
    slug: Some("reverse"),
    summary: "Reverse characters in text values.",
    description: "s = reverse(text) - Reverse characters in text values.",
    keywords: &["reverse", "strings", "transform", "matlab", "compatibility"],
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
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed identity facts", location: "crates/runmat-builtins/src/catalog/entries/strings/transform/reverse/inference.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Original unit source reference (not execution evidence)", location: "builtins::strings::transform::compat::tests" },
        ],
        notes: &["Original examples are preserved as standalone asserted programs except the impossible GPU text example, whose host equivalent is explicit. Historical display outputs were unverified presentation claims and are retired rather than invented as transcripts. Source-only migration requires centralized compiler, catalog, documentation, and backend qualification."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
