mod examples;
mod faqs;
mod sections;

use crate::*;

use super::super::documentation::REFERENCE_LINKS;

const RELATED: &[&str] = &[
    "rdivide", "times", "power", "mldivide", "gpuArray", "gather",
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("ldivide"),
    slug: Some("ldivide"),
    summary: "Divide the second array by the first, element by element.",
    description: "`ldivide(A, B)` and `A .\\ B` compute `B ./ A`. The implementation applies compatible singleton expansion and covers floating-point, complex, fixed-width integer, logical, character, symbolic, and provider-resident operands under their documented class rules.",
    keywords: &[
        "ldivide",
        "element-wise left division",
        ".\\",
        "implicit expansion",
        "integer",
        "gpu",
    ],
    related: RELATED,
    sections: sections::SECTIONS,
    examples: examples::EXAMPLES,
    example_exemption: None,
    faqs: faqs::FAQS,
    links: REFERENCE_LINKS,
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "Element-wise left-division runtime",
            target: BuiltinDocumentationLinkTarget::Source(
                "https://github.com/runmat-org/runmat/tree/main/crates/runmat-runtime/src/builtins/math/elementwise/binary_arithmetic/ldivide",
            ),
        }],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "Host, complex, integer, provider, and WGPU left-division behavior",
                location: "builtins::math::elementwise::binary_arithmetic::ldivide::tests",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "Executable catalog examples",
                location: "scripts/runtime/verify-builtin-examples.mjs",
            },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
