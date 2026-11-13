mod examples;
mod faqs;
mod sections;

use crate::*;

use super::super::documentation::REFERENCE_LINKS;

const RELATED: &[&str] = &[
    "ldivide", "times", "power", "mrdivide", "gpuArray", "gather",
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("rdivide"),
    slug: Some("rdivide"),
    summary: "Divide arrays element by element with compatible singleton expansion.",
    description: "`rdivide(A, B)` and `A ./ B` divide each element of `A` by the corresponding element of `B`. The implementation covers floating-point, complex, fixed-width integer, logical, character, symbolic, and provider-resident operands under their documented class rules.",
    keywords: &[
        "rdivide",
        "element-wise division",
        "./",
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
            label: "Element-wise right-division runtime",
            target: BuiltinDocumentationLinkTarget::Source(
                "https://github.com/runmat-org/runmat/tree/main/crates/runmat-runtime/src/builtins/math/elementwise/binary_arithmetic/rdivide",
            ),
        }],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "Host, complex, integer, provider, and WGPU right-division behavior",
                location: "builtins::math::elementwise::binary_arithmetic::rdivide::tests",
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
