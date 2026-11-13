use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "Split text at newline boundaries.",
            "Integer semantics: `splitlines` has no numeric input role. Numeric, logical, symbolic, and provider-resident values are rejected before gather or provider access.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Placement and effects",
        paragraphs: &[
            "Text is host-only. Numeric, logical, symbolic, and provider-resident values reject before gather or provider access.",
            "This pure host operation may throw and may suspend at admitted gathers. It performs no callbacks, workspace/environment mutation, filesystem operations, randomness, or Clock reads. There is no GPU kernel, fusion route, or distributed support. VM and native calls share the canonical portable Runtime binding; these declarations are not execution evidence.",
            "Not eligible for fusion; this builtin materializes host text or logical results.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Current implementation frontier",
        paragraphs: &[
            "Character arrays with non-singleton dimensions beyond two, including those inside cells and delimiter controls, reject before row extraction. String and cell array geometry follows the per-entry behavior described here.",
            "CRLF and CR normalize to LF. Scalar text returns a string column; nonscalar text returns a cell array whose elements are string columns, preserving text-element geometry. Typed missing strings remain missing; literal `<missing>` text is ordinary text. This container behavior and missing-value policy are not claimed as independently verified MATLAB parity. Managed text allocation remains incomplete.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Syntax and argument roles",
        paragraphs: &[
            "s = splitlines(text)",
            "text: Input text.",
            "out: Output value.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "example-1",
    title: "splitlines(\"a\\nb\") returns a two-element string array.",
    program: "out = splitlines(sprintf(\"a\\nb\"));",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isequal(out,[\"a\";\"b\"]));",
    },
}];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("splitlines"),
    slug: Some("splitlines"),
    summary: "Split text at newline boundaries.",
    description: "s = splitlines(text) - Split text at newline boundaries.",
    keywords: &["splitlines", "strings", "transform", "matlab", "compatibility"],
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
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed identity facts", location: "crates/runmat-builtins/src/catalog/entries/strings/transform/splitlines/inference.rs" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Original unit source reference (not execution evidence)", location: "builtins::strings::transform::compat::tests" },
        ],
        notes: &["Original examples are preserved as standalone asserted programs except the impossible GPU text example, whose host equivalent is explicit. Historical display outputs were unverified presentation claims and are retired rather than invented as transcripts. Source-only migration requires centralized compiler, catalog, documentation, and backend qualification."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
