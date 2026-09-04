use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};

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

const SQRT_EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "positive-scalar",
        title: "Take the square root of a positive scalar",
        program: "y = sqrt(9)",
        display_output: Some("y = 3"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(y == 3);",
        },
    },
    BuiltinExample {
        id: "matrix",
        title: "Take square roots elementwise across a matrix",
        program: "A = [1 4 9; 16 25 36];\nR = sqrt(A)",
        display_output: Some("R = [1 2 3; 4 5 6]"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(R, [1 2 3; 4 5 6]));",
        },
    },
    BuiltinExample {
        id: "negative-real-input",
        title: "Promote negative real input to a complex result",
        program: "values = [-1 -4 9];\nroots = sqrt(values)",
        display_output: Some("roots = [0.0000 + 1.0000i, 0.0000 + 2.0000i, 3.0000 + 0.0000i]"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(max(abs(roots - [1i 2i 3])) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "gpu-array",
        title: "Take square roots of provider-resident data",
        program: "G = gpuArray([0 1; 4 9]);\nout = sqrt(G);\nresult = gather(out)",
        display_output: Some("result = [0 1; 2 3]"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Wgpu,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(result, [0 1; 2 3]));",
        },
    },
    BuiltinExample {
        id: "complex-values",
        title: "Take principal square roots of complex values",
        program: "z = [3 + 4i, -1 + 2i];\nw = sqrt(z)",
        display_output: Some("w = [2 + 1i, 0.7862 + 1.2720i]"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(max(abs(w .* w - z)) < 1e-10);",
        },
    },
    BuiltinExample {
        id: "character-codes",
        title: "Use the numeric code points of a character array",
        program: "C = 'AB';\nnumericRoots = sqrt(C)",
        display_output: Some("numericRoots = [8.0623 8.2462]"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(max(abs(numericRoots - sqrt([65 66]))) < 1e-12);",
        },
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
        answer: "RunMat uses the principal complex branch and normalizes negligible components to avoid displaying negative-zero artifacts.",
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
            location: "builtins::math::elementwise::sqrt::tests",
        },
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::ProviderTest,
            label: "Provider round-trip test",
            location: "builtins::math::elementwise::sqrt::tests::sqrt_gpu_provider_roundtrip",
        },
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::ProviderTest,
            label: "WGPU conformance test",
            location: "builtins::math::elementwise::sqrt::tests::sqrt_wgpu_matches_cpu_elementwise",
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
