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
            "`Y = cosh(X)` computes hyperbolic cosine element by element and preserves the shape of scalar, vector, matrix, empty, and N-D input.",
            "Real and complex single input retain single precision; double input retains double precision. Fixed-width integer, logical, and character input are RunMat extensions and produce double output. Integers must be exactly representable at the binary64 boundary; aligned values above `flintmax` remain eligible when their significant bits fit.",
            "Complex input follows `cosh(a + bi) = cosh(a) cos(b) + i sinh(a) sin(b)`. Character input is evaluated from numeric code points. NaN and infinity follow the floating-point operation. Sparse, table, timetable, tall, and distributed containers are not currently accepted by this builtin.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "GPU execution",
        paragraphs: &[
            "Supported real floating input can execute through the owning provider's unary operation and participate in elementwise fusion. Successful provider output is validated before it becomes the result.",
            "A typed unsupported response gathers through the concrete owner, computes on the host, and restores a new result to the same owner and device. Malformed output and other provider failures remain errors.",
            "Placement and fusion may keep eligible expressions resident without an explicit `gpuArray`. Explicit `gpuArray` and `gather` remain available when a program needs to control the boundary.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "scalar",
        title: "Compute a scalar hyperbolic cosine",
        program: "y = cosh(2)",
        display_output: Some("y = 3.7622"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(abs(y - 3.762195691083632) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "vector",
        title: "Apply cosh to a vector",
        program: "x = linspace(-2, 2, 5);\ny = cosh(x)",
        display_output: Some("y = [3.7622 1.5431 1 1.5431 3.7622]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "expected = [3.762195691083632 1.543080634815244 1 1.543080634815244 3.762195691083632];\nassert(max(abs(y - expected)) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "matrix",
        title: "Evaluate a matrix element by element",
        program: "A = [0 0.5; 1 1.5];\nB = cosh(A)",
        display_output: Some("B = [1 1.1276; 1.5431 2.3524]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "expected = [1 1.127625965206381; 1.543080634815244 2.352409615243247];\nassert(isequal(size(B), [2 2]));\nassert(max(abs(B - expected), [], \"all\") < 1e-12);",
        },
    },
    BuiltinExample {
        id: "gpu",
        title: "Evaluate provider-resident input",
        program: "G = gpuArray([0.25 0.75; 1.25 1.75]);\nresult_gpu = cosh(G);\nresult = gather(result_gpu)",
        display_output: Some("result = [1.0314 1.2947; 1.8884 2.9642]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        verification: BuiltinExampleVerification::Assertions {
            source: "expected = [1.031413099879573 1.294683284676844; 1.888423877161016 2.964188309728088];\nassert(max(abs(result - expected), [], \"all\") < 1e-12);",
        },
    },
    BuiltinExample {
        id: "complex",
        title: "Evaluate a complex value",
        program: "z = 1 + 2i;\nw = cosh(z)",
        display_output: Some("w = -0.6421 + 1.0686i"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "expected = -0.642148124715520 + 1.068607421382779i;\nassert(abs(w - expected) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "characters",
        title: "Evaluate character code points in RunMat mode",
        program: "chars = 'AZ';\ncodes = cosh(chars)",
        display_output: Some("codes contains cosh of the A and Z code points"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "expected = cosh([65 90]);\nassert(isequal(size(codes), [1 2]));\nassert(max(abs(codes - expected) ./ expected) < 1e-12);",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "When should I use cosh?", answer: "Use `cosh` in hyperbolic models, differential equations, and signal transformations where hyperbolic cosine appears." },
    BuiltinDocumentationFaq { question: "Does cosh accept complex input?", answer: "Yes. It evaluates the analytic continuation element by element while retaining supported floating precision." },
    BuiltinDocumentationFaq { question: "How are integers handled?", answer: "RunMat mode accepts all eight fixed-width classes when every value is exactly representable as double, and returns double. MATLAB compatibility mode rejects this extension." },
    BuiltinDocumentationFaq { question: "How are logical and character inputs handled?", answer: "They are separate RunMat extensions. Logical values and character code points enter the double computation boundary and preserve shape." },
    BuiltinDocumentationFaq { question: "What happens when a provider lacks unary cosh?", answer: "A typed unsupported response triggers an owner-preserving host fallback. Other provider failures are reported." },
    BuiltinDocumentationFaq { question: "Can cosh participate in fusion?", answer: "Yes, for supported real floating elementwise expressions. Complex and conversion paths remain runtime boundaries." },
    BuiltinDocumentationFaq { question: "How does cosh handle NaN and infinity?", answer: "Real NaN propagates and either real infinity produces positive infinity. Complex special values follow the complex floating-point formula." },
    BuiltinDocumentationFaq { question: "Is provider warmup required?", answer: "A provider may prepare pipelines during initialization. The typed fallback path remains available when its unary operation is unsupported." },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "MATLAB cosh documentation", target: BuiltinDocumentationLinkTarget::External("https://www.mathworks.com/help/matlab/ref/double.cosh.html") },
    BuiltinDocumentationLink { label: "sinh", target: BuiltinDocumentationLinkTarget::Builtin("sinh") },
    BuiltinDocumentationLink { label: "tanh", target: BuiltinDocumentationLinkTarget::Builtin("tanh") },
    BuiltinDocumentationLink { label: "acosh", target: BuiltinDocumentationLinkTarget::Builtin("acosh") },
    BuiltinDocumentationLink { label: "asinh", target: BuiltinDocumentationLinkTarget::Builtin("asinh") },
    BuiltinDocumentationLink { label: "atanh", target: BuiltinDocumentationLinkTarget::Builtin("atanh") },
    BuiltinDocumentationLink { label: "cos", target: BuiltinDocumentationLinkTarget::Builtin("cos") },
    BuiltinDocumentationLink { label: "sin", target: BuiltinDocumentationLinkTarget::Builtin("sin") },
    BuiltinDocumentationLink { label: "tan", target: BuiltinDocumentationLinkTarget::Builtin("tan") },
    BuiltinDocumentationLink { label: "acos", target: BuiltinDocumentationLinkTarget::Builtin("acos") },
    BuiltinDocumentationLink { label: "asin", target: BuiltinDocumentationLinkTarget::Builtin("asin") },
    BuiltinDocumentationLink { label: "atan", target: BuiltinDocumentationLinkTarget::Builtin("atan") },
    BuiltinDocumentationLink { label: "atan2", target: BuiltinDocumentationLinkTarget::Builtin("atan2") },
    BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") },
    BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Builtin("gather") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/trigonometry/cosh.rs") },
];

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Hyperbolic-cosine runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/trigonometry/cosh.rs") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Real, complex, typed, shape, extension, and error behavior", location: "crates/runmat-runtime/src/builtins/math/trigonometry/cosh.rs::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::ProviderTest, label: "Owner-preserving provider execution and malformed-output rejection", location: "crates/runmat-runtime/src/builtins/math/trigonometry/cosh.rs::tests::cosh_gpu_provider_roundtrip" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::WgpuTest, label: "Actual WGPU parity", location: "crates/runmat-runtime/src/builtins/math/trigonometry/cosh.rs::tests::cosh_wgpu_matches_cpu_elementwise" },
    ],
    notes: &[],
};

pub(crate) const COSH_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("cosh"),
    slug: Some("cosh"),
    summary: "Compute elementwise hyperbolic cosine for real and complex values.",
    description: "`cosh` preserves shape and floating precision, supports complex input, and retains supported provider ownership.",
    keywords: &["cosh", "hyperbolic cosine", "trigonometry", "elementwise", "complex", "gpu"],
    related: &["sinh", "tanh", "acosh", "asinh", "atanh", "cos", "sin", "tan", "acos", "asin", "atan", "atan2", "gpuArray", "gather"],
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
