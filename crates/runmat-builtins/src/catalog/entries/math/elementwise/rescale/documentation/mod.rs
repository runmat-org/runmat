mod examples;

use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference,
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Ranges and clipping",
        paragraphs: &[
            "`R = rescale(A)` maps the minimum and maximum finite observations across all of `A` to 0 and 1. `rescale(A, l, u)` selects another output interval.",
            "`InputMin` and `InputMax` replace the detected input range. Values outside that range are clipped before scaling. Bounds may be arrays and use MATLAB-compatible implicit expansion with `A`.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Classes and exceptional values",
        paragraphs: &[
            "Single input returns single storage. Double input returns double. Logical and fixed-width integer input returns double; integer values and bounds must be exactly representable at this explicit floating-point boundary.",
            "NaN values do not contribute to the default input range and remain NaN in their output positions. An all-NaN input produces NaN output. A constant input range maps to the lower output bound, except that an infinite requested interval produces NaN.",
            "Complex, character, string, cell, structure, object, and function-handle inputs are rejected.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Accelerated execution",
        paragraphs: &[
            "When any operand is provider-resident, RunMat gathers through the exact owning provider, performs the complete range calculation on the host, and restores the result to that same owner. Operands owned by different providers are rejected.",
            "The current implementation is a provider-preserving fallback rather than a fused kernel. `rescale` is a fusion boundary.",
        ],
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Does rescale(A) operate column-wise?", answer: "No. The default range uses all elements of A. Supply vector InputMin and InputMax bounds, such as min(A) and max(A), for independent column scaling." },
    BuiltinDocumentationFaq { question: "What happens when all input values are equal?", answer: "The result is the lower output bound. If either requested output bound is infinite, the result is NaN." },
    BuiltinDocumentationFaq { question: "Can InputMin and InputMax be arrays?", answer: "Yes. Each bound may be a scalar, vector, matrix, or N-D array that is implicitly expandable with A and the other bounds." },
    BuiltinDocumentationFaq { question: "Does a gpuArray result remain resident?", answer: "Yes. Supported calls restore the host-computed result to the exact provider that owned the resident operand." },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "bounds",
        target: BuiltinDocumentationLinkTarget::Builtin("bounds"),
    },
    BuiltinDocumentationLink {
        label: "min",
        target: BuiltinDocumentationLinkTarget::Builtin("min"),
    },
    BuiltinDocumentationLink {
        label: "max",
        target: BuiltinDocumentationLinkTarget::Builtin("max"),
    },
];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/rescale") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Range, broadcast, class, NaN, and diagnostic behavior", location: "builtins::math::elementwise::rescale::tests::host" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::ProviderTest, label: "Exact provider ownership and restored output", location: "builtins::math::elementwise::rescale::tests::provider" },
    ],
    notes: &[],
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("rescale"),
    slug: Some("rescale"),
    summary: "Scale a real array from a selected input range to a selected output range.",
    description: "`rescale` clips values to an explicit or detected input range, then maps that range to the requested output interval with implicit expansion across array bounds.",
    keywords: &["rescale", "normalize", "range", "InputMin", "InputMax", "gpuArray"],
    related: &["bounds", "min", "max"],
    sections: SECTIONS,
    examples: examples::EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: EVIDENCE,
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
