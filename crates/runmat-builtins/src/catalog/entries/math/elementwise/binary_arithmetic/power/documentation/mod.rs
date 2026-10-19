mod examples;
mod faqs;
mod sections;

use crate::*;

use super::super::documentation::REFERENCE_LINKS;

const RELATED: &[&str] = &[
    "times", "rdivide", "ldivide", "mpower", "pow2", "sqrt", "gpuArray", "gather",
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("power"),
    slug: Some("power"),
    summary: "Raise arrays to element-wise powers with compatible singleton expansion.",
    description: "`power(A, B)` and `A .^ B` raise each element of `A` to the corresponding element of `B`. The implementation covers floating-point, complex, fixed-width integer, logical, character, symbolic, and provider-resident operands under their documented domain and class rules.",
    keywords: &[
        "power",
        "element-wise power",
        "dot caret",
        ".^",
        "broadcasting",
        "complex",
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
            label: "Element-wise power runtime",
            target: BuiltinDocumentationLinkTarget::Source(
                "https://github.com/runmat-org/runmat/tree/main/crates/runmat-runtime/src/builtins/math/elementwise/binary_arithmetic/power",
            ),
        }],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "Host, symbolic, complex, integer, provider, and WGPU power behavior",
                location: "builtins::math::elementwise::binary_arithmetic::power::tests",
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
