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
            "`Y = sinh(X)` computes the hyperbolic sine element by element and preserves the shape of scalar, vector, matrix, empty, and N-D input.",
            "Real and complex single input retain single precision. Real and complex double input retain double precision. Fixed-width integer, logical, and character input are RunMat extensions and produce double output.",
            "Complex input follows `sinh(a + bi) = sinh(a) cos(b) + i cosh(a) sin(b)`. Character input is evaluated from its numeric code points. NaN and infinity follow the floating-point hyperbolic operation.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "GPU execution",
        paragraphs: &[
            "Supported real floating input can execute through the owning provider's unary operation and participate in elementwise fusion. Successful provider output is validated before it becomes the result.",
            "When the provider reports that the operation is unsupported, RunMat gathers through the concrete owner, computes on the host, and restores the result to that owner and device. Other provider failures remain errors.",
            "You normally do not need an explicit `gpuArray` conversion: placement and fusion may keep eligible expressions resident. Explicit `gpuArray` and `gather` remain available when a program needs to control the boundary.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "scalar",
        title: "Compute a scalar hyperbolic sine",
        program: "y = sinh(1)",
        display_output: Some("y = 1.1752"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(abs(y - 1.175201193643801) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "vector",
        title: "Apply sinh to a vector",
        program: "x = linspace(-1, 1, 5);\ny = sinh(x)",
        display_output: Some("y = [-1.1752 -0.5211 0 0.5211 1.1752]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "expected = [-1.175201193643801 -0.521095305493747 0 0.521095305493747 1.175201193643801];\nassert(max(abs(y - expected)) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "matrix",
        title: "Evaluate a matrix element by element",
        program: "A = [0 1; 2 3];\nB = sinh(A)",
        display_output: Some("B = [0 1.1752; 3.6269 10.0179]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "expected = [0 1.175201193643801; 3.626860407847019 10.017874927409903];\nassert(isequal(size(B), [2 2]));\nassert(max(abs(B - expected), [], \"all\") < 1e-12);",
        },
    },
    BuiltinExample {
        id: "gpu",
        title: "Evaluate provider-resident input",
        program: "G = gpuArray([0.25 0.5; 0.75 1]);\nresult_gpu = sinh(G);\nresult = gather(result_gpu)",
        display_output: Some("result = [0.2526 0.5211; 0.8223 1.1752]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        verification: BuiltinExampleVerification::Assertions {
            source: "expected = [0.252612316808168 0.521095305493747; 0.822316731935830 1.175201193643801];\nassert(max(abs(result - expected), [], \"all\") < 1e-12);",
        },
    },
    BuiltinExample {
        id: "complex",
        title: "Evaluate a complex value",
        program: "z = 1 + 2i;\nw = sinh(z)",
        display_output: Some("w = -0.4891 + 1.4031i"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "expected = -0.489056259041294 + 1.403119250622041i;\nassert(abs(w - expected) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "empty-shape",
        title: "Preserve an empty matrix shape",
        program: "X = zeros(0, 3);\nY = sinh(X)",
        display_output: Some("Y is a 0-by-3 double matrix"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(Y), [0 3]));\nassert(isempty(Y));",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "When should I use sinh?",
        answer: "Use `sinh` where a model or transformation is expressed with the hyperbolic sine, including analytic continuation and signal-processing formulas.",
    },
    BuiltinDocumentationFaq {
        question: "Does sinh accept complex values?",
        answer: "Yes. Complex input is evaluated element by element with the analytic hyperbolic-sine definition.",
    },
    BuiltinDocumentationFaq {
        question: "How are integer and logical inputs handled?",
        answer: "RunMat mode accepts logical input and all eight fixed-width integer classes as extensions and produces double output. Integer values must be exactly representable at the binary64 computation boundary. MATLAB compatibility mode rejects these forms.",
    },
    BuiltinDocumentationFaq {
        question: "How are character arrays handled?",
        answer: "RunMat mode evaluates each character's numeric code point and returns a shape-preserving double array.",
    },
    BuiltinDocumentationFaq {
        question: "What happens when a provider lacks unary sinh?",
        answer: "A typed unsupported result triggers an owner-preserving host fallback. Other provider failures are reported.",
    },
    BuiltinDocumentationFaq {
        question: "Can sinh participate in fusion?",
        answer: "Yes, for supported real floating elementwise expressions. Complex and conversion paths remain explicit runtime boundaries.",
    },
    BuiltinDocumentationFaq {
        question: "What happens for NaN or infinity?",
        answer: "Real NaN propagates and real positive or negative infinity returns infinity with the same sign. Complex special values follow the complex floating-point formula.",
    },
    BuiltinDocumentationFaq {
        question: "Does sinh preserve shape and precision?",
        answer: "Yes. It preserves shape, and supported single or double floating input retains its precision.",
    },
    BuiltinDocumentationFaq {
        question: "Is provider warmup required?",
        answer: "A provider may prepare pipelines during its own initialization. If its unary operation is unavailable, the typed fallback path preserves correctness and ownership.",
    },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "cosh", target: BuiltinDocumentationLinkTarget::Builtin("cosh") },
    BuiltinDocumentationLink { label: "tanh", target: BuiltinDocumentationLinkTarget::Builtin("tanh") },
    BuiltinDocumentationLink { label: "asinh", target: BuiltinDocumentationLinkTarget::Builtin("asinh") },
    BuiltinDocumentationLink { label: "acosh", target: BuiltinDocumentationLinkTarget::Builtin("acosh") },
    BuiltinDocumentationLink { label: "atanh", target: BuiltinDocumentationLinkTarget::Builtin("atanh") },
    BuiltinDocumentationLink { label: "acos", target: BuiltinDocumentationLinkTarget::Builtin("acos") },
    BuiltinDocumentationLink { label: "asin", target: BuiltinDocumentationLinkTarget::Builtin("asin") },
    BuiltinDocumentationLink { label: "atan", target: BuiltinDocumentationLinkTarget::Builtin("atan") },
    BuiltinDocumentationLink { label: "atan2", target: BuiltinDocumentationLinkTarget::Builtin("atan2") },
    BuiltinDocumentationLink { label: "sin", target: BuiltinDocumentationLinkTarget::Builtin("sin") },
    BuiltinDocumentationLink { label: "cos", target: BuiltinDocumentationLinkTarget::Builtin("cos") },
    BuiltinDocumentationLink { label: "tan", target: BuiltinDocumentationLinkTarget::Builtin("tan") },
    BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") },
    BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Builtin("gather") },
    BuiltinDocumentationLink {
        label: "Implementation",
        target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/trigonometry/sinh.rs"),
    },
];

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink {
        label: "Hyperbolic-sine runtime",
        target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/trigonometry/sinh.rs"),
    }],
    verification: &[
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Real, complex, typed, shape, extension, and error behavior",
            location: "crates/runmat-runtime/src/builtins/math/trigonometry/sinh.rs::tests",
        },
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::ProviderTest,
            label: "Owner-preserving provider execution and fallback",
            location: "crates/runmat-runtime/src/builtins/math/trigonometry/sinh.rs::tests::sinh_gpu_provider_roundtrip",
        },
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::WgpuTest,
            label: "Actual WGPU parity",
            location: "crates/runmat-runtime/src/builtins/math/trigonometry/sinh.rs::tests::sinh_wgpu_matches_cpu_elementwise",
        },
    ],
    notes: &[],
};

pub(crate) const SINH_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("sinh"),
    slug: Some("sinh"),
    summary: "Compute elementwise hyperbolic sine for real and complex values.",
    description: "`sinh` preserves shape and floating precision, supports complex input, and retains supported provider ownership.",
    keywords: &["sinh", "hyperbolic sine", "trigonometry", "complex", "gpu"],
    related: &[
        "cosh", "tanh", "asinh", "acosh", "atanh", "sin", "cos", "tan", "asin", "acos",
        "atan", "atan2", "gpuArray", "gather",
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
