use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference,
};

mod examples;
use examples::SQRT_EXAMPLES;

const SQRT_SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "`sqrt(X)` applies the principal square root elementwise and preserves the input shape. Documented single inputs remain single. Negative real values promote the complete result to the corresponding complex floating class.",
            "Typed integer input is a RunMat extension. Each authoritative integer value must be exactly representable at the double-precision square-root boundary; values that would round are rejected. Logical values convert to double zeros and ones, while character arrays use their numeric code points and produce dense double results.",
            "Complex input uses the principal branch of the complex square root. Signed zeros, infinities, and NaNs follow IEEE arithmetic.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "GPU execution",
        paragraphs: &[
            "RunMat can keep real input on the active provider when the provider implements the required reduction and square-root operations and the values are known to be nonnegative. Otherwise it gathers the value and computes the same principal result on the host. Explicit `gpuArray` and `gather` calls remain available when placement is part of the program.",
            "`sqrt` is not fused into a real-only shader expression. Negative real input must promote the complete result to complex storage instead of producing NaN. Complex provider input is gathered through the value-aware boundary when the provider cannot produce the required complex representation directly.",
        ],
    },
];

const SQRT_FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Does `sqrt` return complex results for negative input?",
        answer: "Yes. Negative real values produce imaginary components, and a mixed-sign array promotes to a complex result.",
    },
    BuiltinDocumentationFaq {
        question: "How does `sqrt` handle logical input?",
        answer: "Logical arrays convert to double before the square root is applied, so true becomes 1 and false becomes 0.",
    },
    BuiltinDocumentationFaq {
        question: "Can a GPU provider handle negative input?",
        answer: "RunMat gathers when the active provider cannot produce the required complex representation. The host calculation uses the same principal-root contract.",
    },
    BuiltinDocumentationFaq {
        question: "Does `sqrt` preserve shape?",
        answer: "Yes. The output has the same shape as the input.",
    },
    BuiltinDocumentationFaq {
        question: "How are NaN and infinity handled?",
        answer: "They follow IEEE square-root behavior: NaN remains NaN, positive infinity remains positive infinity, and negative infinity produces a positive imaginary infinity.",
    },
    BuiltinDocumentationFaq {
        question: "How are complex values near an axis handled?",
        answer: "RunMat uses the principal complex branch and retains small nonzero components. Exact zero components are normalized to positive zero.",
    },
    BuiltinDocumentationFaq {
        question: "Can providers add direct complex support?",
        answer: "Yes. Provider execution is selected through typed capabilities, so an implementation that supports the required complex representation can keep the operation resident.",
    },
];

const SQRT_EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink {
        label: "Runtime implementation",
        target: BuiltinDocumentationLinkTarget::Source(
            "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/roots/sqrt/mod.rs",
        ),
    }],
    verification: &[
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "CPU and representation tests",
            location: "builtins::math::elementwise::roots::sqrt::tests",
        },
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::ProviderTest,
            label: "Provider round-trip test",
            location: "builtins::math::elementwise::roots::sqrt::tests::provider::sqrt_gpu_provider_roundtrip",
        },
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::ProviderTest,
            label: "WGPU conformance test",
            location: "builtins::math::elementwise::roots::sqrt::tests::provider::sqrt_wgpu_matches_cpu_elementwise",
        },
    ],
    notes: &[],
};

pub(super) const SQRT_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("sqrt"),
    slug: Some("sqrt"),
    summary: "Compute principal square roots elementwise with compatible real and complex behavior.",
    description: "`Y = sqrt(X)` computes the principal square root of each element in `X`, preserving single precision for documented single inputs and promoting negative real values to the corresponding complex floating class.",
    keywords: &["sqrt", "square root", "elementwise", "gpu", "complex"],
    related: &[
        "abs", "angle", "conj", "double", "exp", "expm1", "factorial", "gamma",
        "gather", "gpuArray", "hypot", "imag", "ldivide", "log", "log1p", "log2",
        "log10", "minus", "plus", "pow2", "power", "rdivide", "real", "realsqrt",
        "sign", "single", "times",
    ],
    sections: SQRT_SECTIONS,
    examples: SQRT_EXAMPLES,
    example_exemption: None,
    faqs: SQRT_FAQS,
    links: &[],
    media: &[],
    evidence: SQRT_EVIDENCE,
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
