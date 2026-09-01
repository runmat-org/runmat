use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "`Y = asinh(X)` computes the principal inverse hyperbolic sine element by element and preserves the input shape. Every real input has a real result; complex input follows the principal branch.",
            "Real and complex single input retain single precision. Real and complex double input retain double precision. Fixed-width integer, logical, and character input are RunMat extensions and produce double output.",
            "Real NaN propagates. Positive and negative infinity produce infinity with the same sign. Sparse, table, and timetable input are not currently supported.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "GPU execution",
        paragraphs: &[
            "Supported real floating input can execute through the owning provider's unary operation and participate in real elementwise fusion. Successful provider output is validated before it becomes the result.",
            "When the provider reports that the operation is unsupported, RunMat gathers through the concrete owner, computes on the host, and restores the result to that owner and device. Other provider failures remain errors.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "scalar",
        title: "Compute a scalar inverse hyperbolic sine",
        program: "y = asinh(0.5)",
        display_output: Some("y = 0.4812"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(abs(y - 0.481211825059603) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "vector",
        title: "Apply asinh to a vector",
        program: "x = [-2 -1 0 1 2];\ny = asinh(x)",
        display_output: Some("y = [-1.4436 -0.8814 0 0.8814 1.4436]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "expected = [-1.443635475178810 -0.881373587019543 0 0.881373587019543 1.443635475178810];\nassert(max(abs(y - expected)) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "matrix",
        title: "Apply asinh to a matrix",
        program: "A = [0 -0.5 1; 1.5 -2 3];\nB = asinh(A)",
        display_output: Some("B preserves the 2-by-3 shape"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(B), [2 3]));\nassert(max(abs(sinh(B) - A), [], \"all\") < 1e-12);",
        },
    },
    BuiltinExample {
        id: "gpu",
        title: "Evaluate provider-resident input",
        program: "G = gpuArray([0.25 0.5; 0.75 1]);\nresult_gpu = asinh(G);\nresult = gather(result_gpu)",
        display_output: Some("result = [0.2475 0.4812; 0.6931 0.8814]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        verification: BuiltinExampleVerification::Assertions {
            source: "expected = [0.247466461547263 0.481211825059603; 0.693147180559945 0.881373587019543];\nassert(max(abs(result - expected), [], \"all\") < 1e-12);",
        },
    },
    BuiltinExample {
        id: "complex",
        title: "Evaluate complex values on the principal branch",
        program: "z = [1 + 2i, -0.5 + 0.75i];\nw = asinh(z)",
        display_output: Some("w contains principal complex results"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(w), [1 2]));\nassert(max(abs(sinh(w) - z)) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "fused-gpu",
        title: "Use asinh in a provider expression",
        program: "G = gpuArray([0.125 0.25 0.5 1]);\nY = sinh(G) + asinh(G);\nresult = gather(Y)",
        display_output: Some("result contains the combined elementwise values"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        verification: BuiltinExampleVerification::Assertions {
            source: "x = [0.125 0.25 0.5 1];\nexpected = sinh(x) + asinh(x);\nassert(max(abs(result - expected)) < 1e-12);",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Does asinh return complex values for real input?",
        answer: "No. Every real input produces a real result. Complex output arises from complex input.",
    },
    BuiltinDocumentationFaq {
        question: "How are logical and integer inputs handled?",
        answer: "RunMat mode accepts logical input and all eight fixed-width integer classes as extensions and produces double output. MATLAB compatibility mode rejects these forms.",
    },
    BuiltinDocumentationFaq {
        question: "What happens when a provider lacks unary asinh?",
        answer: "A typed unsupported result triggers an owner-preserving host fallback. Other provider failures are reported.",
    },
    BuiltinDocumentationFaq {
        question: "Do CPU and GPU precision agree?",
        answer: "Both paths retain the input's supported single or double precision. Small rounding differences can reflect backend implementations.",
    },
    BuiltinDocumentationFaq {
        question: "Can asinh participate in fusion?",
        answer: "Yes, for supported real floating elementwise expressions. Complex and conversion paths remain runtime boundaries.",
    },
    BuiltinDocumentationFaq {
        question: "How are character arrays processed?",
        answer: "In RunMat mode, character code points enter the double-precision computation boundary and retain their array shape.",
    },
    BuiltinDocumentationFaq {
        question: "What happens with NaN and infinity?",
        answer: "Real NaN propagates, and positive or negative infinity returns infinity with the same sign.",
    },
    BuiltinDocumentationFaq {
        question: "Can complex output remain provider-resident?",
        answer: "Supported complex provider behavior is operation-specific. Host fallback restores supported complex results through the input's concrete owner.",
    },
    BuiltinDocumentationFaq {
        question: "Does asinh preserve shape?",
        answer: "Yes. Scalar, vector, matrix, empty, and N-D inputs retain their shape.",
    },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "sinh",
        target: BuiltinDocumentationLinkTarget::Builtin("sinh"),
    },
    BuiltinDocumentationLink {
        label: "tanh",
        target: BuiltinDocumentationLinkTarget::Builtin("tanh"),
    },
    BuiltinDocumentationLink {
        label: "sin",
        target: BuiltinDocumentationLinkTarget::Builtin("sin"),
    },
    BuiltinDocumentationLink {
        label: "gpuArray",
        target: BuiltinDocumentationLinkTarget::Builtin("gpuArray"),
    },
    BuiltinDocumentationLink {
        label: "gather",
        target: BuiltinDocumentationLinkTarget::Builtin("gather"),
    },
    BuiltinDocumentationLink {
        label: "acos",
        target: BuiltinDocumentationLinkTarget::Builtin("acos"),
    },
    BuiltinDocumentationLink {
        label: "acosh",
        target: BuiltinDocumentationLinkTarget::Builtin("acosh"),
    },
    BuiltinDocumentationLink {
        label: "asin",
        target: BuiltinDocumentationLinkTarget::Builtin("asin"),
    },
    BuiltinDocumentationLink {
        label: "atan",
        target: BuiltinDocumentationLinkTarget::Builtin("atan"),
    },
    BuiltinDocumentationLink {
        label: "atan2",
        target: BuiltinDocumentationLinkTarget::Builtin("atan2"),
    },
    BuiltinDocumentationLink {
        label: "atanh",
        target: BuiltinDocumentationLinkTarget::Builtin("atanh"),
    },
    BuiltinDocumentationLink {
        label: "cos",
        target: BuiltinDocumentationLinkTarget::Builtin("cos"),
    },
    BuiltinDocumentationLink {
        label: "cosh",
        target: BuiltinDocumentationLinkTarget::Builtin("cosh"),
    },
    BuiltinDocumentationLink {
        label: "tan",
        target: BuiltinDocumentationLinkTarget::Builtin("tan"),
    },
    BuiltinDocumentationLink {
        label: "Implementation",
        target: BuiltinDocumentationLinkTarget::Source(
            "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/trigonometry/asinh.rs",
        ),
    },
];

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink {
        label: "Inverse-hyperbolic-sine runtime",
        target: BuiltinDocumentationLinkTarget::Source(
            "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/trigonometry/asinh.rs",
        ),
    }],
    verification: &[
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Real, complex, typed, shape, extension, and error behavior",
            location: "crates/runmat-runtime/src/builtins/math/trigonometry/asinh.rs::tests",
        },
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::ProviderTest,
            label: "Owner-preserving provider execution and fallback",
            location: "crates/runmat-runtime/src/builtins/math/trigonometry/asinh.rs::tests::asinh_gpu_provider_roundtrip",
        },
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::WgpuTest,
            label: "Actual WGPU parity",
            location: "crates/runmat-runtime/src/builtins/math/trigonometry/asinh.rs::tests::asinh_wgpu_matches_cpu",
        },
    ],
    notes: &[],
};

pub(crate) const ASINH_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("asinh"),
    slug: Some("asinh"),
    summary: "Compute elementwise principal inverse hyperbolic sine for real and complex values.",
    description: "`asinh` preserves shape and floating precision, keeps every real input real, supports principal complex input, and retains supported provider ownership.",
    keywords: &[
        "asinh",
        "inverse hyperbolic sine",
        "arcsinh",
        "trigonometry",
        "complex",
        "gpu",
    ],
    related: &[
        "sinh", "tanh", "sin", "gpuArray", "gather", "acos", "acosh", "asin", "atan",
        "atan2", "atanh", "cos", "cosh", "tan",
    ],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: EVIDENCE,
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
