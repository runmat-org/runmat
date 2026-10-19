use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("j"),
    slug: Some("j"),
    summary: "Imaginary unit.",
    description: concat!(
        "The complex double scalar with real part zero ",
        "and imaginary part one. A workspace variable ",
        "named j shadows this constant; use 1i when a ",
        "literal imaginary unit is required.",
    ),
    keywords: &["j", "constant"],
    related: &[],

    sections: &[BuiltinDocumentationSection {
        heading: "Value and execution",
        paragraphs: &[concat!(
            "Evaluating this bare constant produces a host ",
            "scalar without arguments, allocation of array ",
            "payloads, provider access, ambient services, ",
            "or effects. Native and WASM consumers share ",
            "the constant's static kind; runtime ",
            "registration supplies its value. Constant ",
            "lookup does not create a callable overload.",
        )],
    }],

    examples: &[BuiltinExample {
        id: "value",
        title: "Read the constant",
        program: "value = j",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: concat!(
                "assert(isa(value, 'double')); ",
                "assert(isscalar(value)); assert(real(value) ",
                "== 0); assert(imag(value) == 1); assert(value ",
                "* value == -1);",
            ),
        },
    }],

    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq {
        question: "Does this constant select GPU residency?",
        answer: concat!(
            "No. It is a host scalar that may be consumed ",
            "or transferred by ordinary numeric ",
            "operations.",
        ),
    }],

    links: &[],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "Runtime constant binding",
            target: BuiltinDocumentationLinkTarget::Source(concat!(
                "https://github.com/runmat-org/runmat/blob/mai",
                "n/crates/runmat-runtime/src/builtins/constant",
                "s/core/j.rs",
            )),
        }],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Constant registry and values",
            location: "crates/runmat-runtime/src/builtins/constants/tests.rs",
        }],
        notes: &[concat!(
            "No legacy editable documentation sidecar or ",
            "runtime shadow existed for this constant.",
        )],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
