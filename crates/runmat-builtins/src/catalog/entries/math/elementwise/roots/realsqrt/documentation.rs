use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};

const REALSQRT_SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "`realsqrt(X)` applies the real square root elementwise and preserves shape. Only real single and double input is accepted; logical, character, fixed-width integer, and every complex representation are rejected before coercion or provider dispatch.",
            "Sparse input remains sparse: stored values are square-rooted and implicit zeros remain zero. A negative stored or dense value raises a domain error because its square root would require a complex result. NaNs and positive infinities follow IEEE real square-root arithmetic.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "GPU execution",
        paragraphs: &[
            "RunMat rejects complex provider buffers first, then uses provider reduction to prove that real input is nonnegative. When the proof and square-root operation are available, the result remains provider-resident. Otherwise RunMat gathers and applies the same real-domain validation on the host.",
            "`realsqrt` is not fused. A negative value must raise the documented domain error instead of becoming NaN inside a fused real shader.",
        ],
    },
];

const REALSQRT_EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "positive-scalar",
        title: "Take the real square root of a scalar",
        program: "y = realsqrt(9)",
        display_output: Some("y = 3"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(y == 3);",
        },
    },
    BuiltinExample {
        id: "matrix",
        title: "Take real square roots elementwise across a matrix",
        program: "A = [1 4 9; 16 25 36];\nR = realsqrt(A)",
        display_output: Some("R = [1 2 3; 4 5 6]"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(R, [1 2 3; 4 5 6]));",
        },
    },
    BuiltinExample {
        id: "negative-domain-error",
        title: "Reject values whose square root is not real",
        program: "realsqrt([-1 4])",
        display_output: Some("Error using realsqrt\nInput must be nonnegative."),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::ExpectedError {
            identifier: "RunMat:realsqrt:ComplexResult",
        },
    },
    BuiltinExample {
        id: "sparse-storage",
        title: "Preserve sparse storage",
        program: "S = sparse([1 3], [1 2], [4 9], 3, 2);\nR = realsqrt(S)",
        display_output: Some("R = sparse([1 3], [1 2], [2 3], 3, 2)"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(issparse(R));\nassert(isequal(full(R), [2 0; 0 0; 0 3]));",
        },
    },
    BuiltinExample {
        id: "gpu-array",
        title: "Take real square roots of provider-resident data",
        program: "G = gpuArray([0 1; 4 9]);\nout = realsqrt(G);\nresult = gather(out)",
        display_output: Some("result = [0 1; 2 3]"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Wgpu,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(result, [0 1; 2 3]));",
        },
    },
];

const REALSQRT_FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "How does `realsqrt` differ from `sqrt`?",
        answer: "`sqrt` promotes negative real values to complex results. `realsqrt` stays in the real domain and raises an error when any value is negative.",
    },
    BuiltinDocumentationFaq {
        question: "Does `realsqrt` accept complex input with zero imaginary parts?",
        answer: "No. Every complex representation is rejected because `realsqrt` requires real storage as well as real values.",
    },
    BuiltinDocumentationFaq {
        question: "Can `realsqrt` run on GPU data?",
        answer: "Yes. RunMat uses provider execution after proving that a real provider buffer is nonnegative; otherwise it gathers and validates the same contract on the host.",
    },
    BuiltinDocumentationFaq {
        question: "Can `realsqrt` be fused?",
        answer: "No. It remains a fusion boundary so negative input raises the documented error instead of producing a shader NaN.",
    },
    BuiltinDocumentationFaq {
        question: "What happens to NaN values?",
        answer: "NaN remains NaN. A provider may need to gather when a reduction cannot establish the nonnegative-domain condition.",
    },
];

const REALSQRT_EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink {
        label: "Runtime implementation",
        target: BuiltinDocumentationLinkTarget::Source(
            "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/roots/realsqrt/mod.rs",
        ),
    }],
    verification: &[
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "CPU, sparse, and domain tests",
            location: "builtins::math::elementwise::realsqrt::tests",
        },
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::ProviderTest,
            label: "Provider round-trip test",
            location: "builtins::math::elementwise::realsqrt::tests::gpu_provider_roundtrip",
        },
    ],
    notes: &[],
};

pub(super) const REALSQRT_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("realsqrt"),
    slug: Some("realsqrt"),
    summary: "Compute real square roots elementwise and reject values that require complex results.",
    description: "`Y = realsqrt(X)` computes `sqrt(X)` for real nonnegative input. Unlike `sqrt`, it rejects negative real values and complex input instead of producing a complex result.",
    keywords: &[
        "realsqrt",
        "square root",
        "real square root",
        "elementwise",
        "gpu",
    ],
    related: &["abs", "gather", "gpuArray", "power", "real", "sqrt"],
    sections: REALSQRT_SECTIONS,
    examples: REALSQRT_EXAMPLES,
    example_exemption: None,
    faqs: REALSQRT_FAQS,
    links: &[],
    media: &[],
    evidence: REALSQRT_EVIDENCE,
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
