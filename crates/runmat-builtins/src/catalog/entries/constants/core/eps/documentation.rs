use crate::*;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("eps"),
    slug: Some("eps"),
    summary: "Double-precision spacing at one.",
    description: concat!(
        "The difference between 1 and the next larger ",
        "representable double, exactly 2^-52. This ",
        "constant is not an overload for spacing at an ",
        "arbitrary input.",
    ),
    keywords: &["eps", "constant"],
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
        program: "value = eps",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: concat!(
                "assert(isa(value, 'double')); ",
                "assert(isscalar(value)); assert(value == ",
                "2^-52); assert(1 + value > 1);",
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
                "s/core/eps.rs",
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
