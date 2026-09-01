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
            "`Y = atanh(X)` computes the principal inverse hyperbolic tangent element by element and preserves the input shape. Real values in the closed interval from minus one through one remain real; real values outside that interval produce complex output.",
            "The endpoints are exact: `atanh(1)` is positive infinity and `atanh(-1)` is negative infinity. Real values outside the interval have an imaginary component of positive pi over two on RunMat's MATLAB-compatible real-input branch.",
            "Real and complex single input retain single precision. Real and complex double input retain double precision. Fixed-width integer, logical, and character input are RunMat extensions and enter an explicit double-precision computation boundary.",
            "Real NaN propagates as real NaN. Sparse, table, and timetable input are not currently supported. Empty arrays retain their shape and floating representation.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "GPU execution",
        paragraphs: &[
            "A provider can execute real floating input directly when exact domain inspection proves that every value lies in the closed interval from minus one through one. Successful provider output is validated before it becomes the result.",
            "When domain inspection or the unary operation is unsupported, RunMat gathers through the concrete owner, computes on the host, and restores the result to that owner and device. Other provider failures remain errors.",
            "MATLAB-compatible GPU execution requires potentially complex output to start from explicitly complex input. RunMat mode additionally permits resident real input outside the real domain and restores the promoted complex result through the same provider.",
            "Ordinary real elementwise fusion is disabled because input outside the closed unit interval must produce a complex result rather than a device NaN.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "scalar",
        title: "Compute a scalar inverse hyperbolic tangent",
        program: "y = atanh(0.5)",
        display_output: Some("y = 0.5493"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(abs(y - 0.549306144334055) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "vector",
        title: "Apply atanh to a vector",
        program: "x = [-0.9 -0.45 0 0.45 0.9];\ny = atanh(x)",
        display_output: Some("y = [-1.4722 -0.4847 0 0.4847 1.4722]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "expected = [-1.472219489583220 -0.484700278594052 0 0.484700278594052 1.472219489583220];\nassert(max(abs(y - expected)) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "endpoints",
        title: "Evaluate the real-domain endpoints",
        program: "A = [0.99 1 -1; 0 0.5 -0.5];\nB = atanh(A)",
        display_output: Some("The values at 1 and -1 are positive and negative infinity"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isinf(B(1, 2)) && B(1, 2) > 0);\nassert(isinf(B(1, 3)) && B(1, 3) < 0);\nfinite_mask = [true false false; true true true];\nassert(max(abs(tanh(B(finite_mask)) - A(finite_mask))) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "complex-promotion",
        title: "Promote real values outside the unit interval",
        program: "values = [2 -3];\nresult = atanh(values)",
        display_output: Some("result = [0.5493 + 1.5708i, -0.3466 + 1.5708i]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "expected = [0.549306144334055 + 1.570796326794897i, -0.346573590279973 + 1.570796326794897i];\nassert(max(abs(result - expected)) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "complex-input",
        title: "Evaluate complex input on the principal branch",
        program: "Z = [1 + 2i, -0.5 + 0.75i];\nW = atanh(Z)",
        display_output: Some("W contains the principal complex inverse hyperbolic tangents"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(W), [1 2]));\nassert(max(abs(tanh(W) - Z)) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "gpu-real-domain",
        title: "Keep real-domain input on its provider",
        program: "G = gpuArray([-0.8 -0.4 0.4 0.8]);\ngpu_result = atanh(G);\nresult = gather(gpu_result)",
        display_output: Some("result = [-1.0986 -0.4236 0.4236 1.0986]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        verification: BuiltinExampleVerification::Assertions {
            source: "expected = [-1.098612288668110 -0.423648930193602 0.423648930193602 1.098612288668110];\nassert(max(abs(result - expected)) < 1e-12);",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "When does atanh return complex values?",
        answer: "Real input produces a complex result when any element is strictly less than minus one or greater than one. Complex input follows the principal complex branch.",
    },
    BuiltinDocumentationFaq {
        question: "How are the endpoints handled?",
        answer: "atanh(1) returns positive infinity and atanh(-1) returns negative infinity. Both are real results.",
    },
    BuiltinDocumentationFaq {
        question: "What happens with NaN and infinity?",
        answer: "Real NaN remains real NaN. Positive and negative infinity produce principal complex results with an imaginary component of positive pi over two.",
    },
    BuiltinDocumentationFaq {
        question: "How are logical and integer inputs handled?",
        answer: "RunMat mode accepts logical input and all eight fixed-width integer classes as extensions and computes a double or complex-double result. MATLAB compatibility mode rejects these forms.",
    },
    BuiltinDocumentationFaq {
        question: "Can real input outside the unit interval remain provider-resident?",
        answer: "RunMat mode can gather through the concrete owner, compute the promoted complex result, and restore it through that owner. MATLAB compatibility mode requires potentially complex GPU input to be explicitly complex.",
    },
    BuiltinDocumentationFaq {
        question: "What happens when a provider lacks a required operation?",
        answer: "A typed unsupported domain reduction or unary operation triggers an owner-preserving host fallback. Other provider failures are reported.",
    },
    BuiltinDocumentationFaq {
        question: "Can atanh participate in fusion?",
        answer: "Not in ordinary real elementwise fusion. The result domain depends on runtime values, and a real fused kernel cannot represent required complex promotion.",
    },
    BuiltinDocumentationFaq {
        question: "Does atanh preserve shape and precision?",
        answer: "Yes. It preserves shape and retains single or double precision for floating input. RunMat extension inputs use double precision.",
    },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "tanh",
        target: BuiltinDocumentationLinkTarget::Builtin("tanh"),
    },
    BuiltinDocumentationLink {
        label: "asinh",
        target: BuiltinDocumentationLinkTarget::Builtin("asinh"),
    },
    BuiltinDocumentationLink {
        label: "acosh",
        target: BuiltinDocumentationLinkTarget::Builtin("acosh"),
    },
    BuiltinDocumentationLink {
        label: "atan",
        target: BuiltinDocumentationLinkTarget::Builtin("atan"),
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
        label: "asin",
        target: BuiltinDocumentationLinkTarget::Builtin("asin"),
    },
    BuiltinDocumentationLink {
        label: "atan2",
        target: BuiltinDocumentationLinkTarget::Builtin("atan2"),
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
        label: "sin",
        target: BuiltinDocumentationLinkTarget::Builtin("sin"),
    },
    BuiltinDocumentationLink {
        label: "sinh",
        target: BuiltinDocumentationLinkTarget::Builtin("sinh"),
    },
    BuiltinDocumentationLink {
        label: "tan",
        target: BuiltinDocumentationLinkTarget::Builtin("tan"),
    },
    BuiltinDocumentationLink {
        label: "Implementation",
        target: BuiltinDocumentationLinkTarget::Source(
            "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/trigonometry/atanh.rs",
        ),
    },
];

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink {
        label: "Inverse-hyperbolic-tangent runtime",
        target: BuiltinDocumentationLinkTarget::Source(
            "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/trigonometry/atanh.rs",
        ),
    }],
    verification: &[
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Real, complex, typed, domain, shape, extension, and error behavior",
            location: "crates/runmat-runtime/src/builtins/math/trigonometry/atanh.rs::tests",
        },
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::ProviderTest,
            label: "Domain inspection, owner-preserving execution, and complex fallback",
            location: "crates/runmat-runtime/src/builtins/math/trigonometry/atanh.rs::tests::atanh_gpu_provider_roundtrip",
        },
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::WgpuTest,
            label: "Actual WGPU real-domain parity",
            location: "crates/runmat-runtime/src/builtins/math/trigonometry/atanh.rs::tests::atanh_wgpu_matches_cpu_elementwise",
        },
    ],
    notes: &[],
};

pub(crate) const ATANH_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("atanh"),
    slug: Some("atanh"),
    summary: "Compute elementwise principal inverse hyperbolic tangent with exact real-to-complex domain handling.",
    description: "`atanh` preserves shape and floating precision, treats the closed real unit interval exactly, promotes other real values to principal complex results, and retains supported provider ownership.",
    keywords: &[
        "atanh",
        "inverse hyperbolic tangent",
        "artanh",
        "trigonometry",
        "gpu",
        "complex",
    ],
    related: &[
        "tanh", "asinh", "acosh", "atan", "gpuArray", "gather", "acos", "asin", "atan2",
        "cos", "cosh", "sin", "sinh", "tan",
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
