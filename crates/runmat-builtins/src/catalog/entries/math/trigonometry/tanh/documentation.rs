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
            "`Y = tanh(X)` computes hyperbolic tangent element by element and preserves scalar, vector, matrix, empty, singleton, and N-D shapes.",
            "Real and complex single input retain single precision; double input retains double precision. Fixed-width integer, logical, and character input are RunMat extensions and produce double output. Integers must be exactly representable at the binary64 boundary.",
            "Complex values use the analytic extension `tanh(z) = sinh(z) / cosh(z)`. Large finite real magnitudes approach -1 or 1; complex poles and special values follow floating-point behavior. Character input is evaluated from Unicode code points. Strings, tables, and timetables are not currently supported.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "GPU execution",
        paragraphs: &[
            "Supported real floating input can execute through the owning provider's unary operation and participate in elementwise fusion. Successful provider output is validated before it becomes the result.",
            "A typed unsupported response gathers through the concrete owner, computes on the host, and restores a new result to the same owner and device. Complex and conversion forms use the same owner-preserving fallback. Other provider failures remain errors.",
            "Placement and fusion may keep neighboring operations resident without explicit transfers. `gpuArray` and `gather` remain available when a program needs to control residency.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "scalar",
        title: "Compute a scalar hyperbolic tangent",
        program: "y = tanh(1)",
        display_output: Some("y = 0.7616"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions { source: "assert(abs(y - 0.761594155955765) < 1e-12);" },
    },
    BuiltinExample {
        id: "vector",
        title: "Apply tanh to a symmetric vector",
        program: "x = linspace(-2, 2, 5);\ny = tanh(x)",
        display_output: Some("y = [-0.9640 -0.7616 0 0.7616 0.9640]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions { source: "expected = [-0.964027580075817 -0.761594155955765 0 0.761594155955765 0.964027580075817];\nassert(max(abs(y - expected)) < 1e-12);" },
    },
    BuiltinExample {
        id: "gpu",
        title: "Evaluate a provider-resident matrix",
        program: "G = gpuArray([0 0.5; 1 1.5]);\nresult_gpu = tanh(G);\nresult = gather(result_gpu)",
        display_output: Some("result = [0 0.4621; 0.7616 0.9051]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        verification: BuiltinExampleVerification::Assertions { source: "expected = [0 0.462117157260010; 0.761594155955765 0.905148253644866];\nassert(max(abs(result - expected), [], \"all\") < 1e-12);" },
    },
    BuiltinExample {
        id: "complex",
        title: "Evaluate a complex value",
        program: "z = 0.5 + 1i;\nw = tanh(z)",
        display_output: Some("w = 1.0428 + 0.8069i"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions { source: "expected = 1.042830728344361 + 0.806877412163085i;\nassert(abs(w - expected) < 1e-12);" },
    },
    BuiltinExample {
        id: "characters",
        title: "Evaluate character code points in RunMat mode",
        program: "c = tanh('ABC')",
        display_output: Some("c = [1 1 1] to displayed precision"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions { source: "expected = tanh([65 66 67]);\nassert(isequal(size(c), [1 3]));\nassert(max(abs(c - expected)) < 1e-12);" },
    },
    BuiltinExample {
        id: "empty-shape",
        title: "Preserve an empty matrix shape",
        program: "E = zeros(0, 3);\nout = tanh(E)",
        display_output: Some("out is a 0-by-3 double matrix"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(out), [0 3]));\nassert(isempty(out));" },
    },
    BuiltinExample {
        id: "activation",
        title: "Apply a bounded activation",
        program: "inputs = [-3 -1 0 1 3];\nactivations = tanh(inputs / 2)",
        display_output: Some("activations = [-0.9051 -0.4621 0 0.4621 0.9051]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions { source: "expected = [-0.905148253644866 -0.462117157260010 0 0.462117157260010 0.905148253644866];\nassert(max(abs(activations - expected)) < 1e-12);" },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "When should I use tanh?", answer: "Use `tanh` for hyperbolic-tangent evaluations in signal processing, numerical solvers, and bounded activation functions." },
    BuiltinDocumentationFaq { question: "Does tanh accept complex input?", answer: "Yes. It evaluates the analytic complex extension element by element and retains supported floating precision." },
    BuiltinDocumentationFaq { question: "How are integers and logicals handled?", answer: "RunMat mode accepts logical input and exactly representable values from all eight fixed-width integer classes, producing double output. MATLAB compatibility mode rejects these extensions." },
    BuiltinDocumentationFaq { question: "How are character arrays handled?", answer: "RunMat mode evaluates Unicode code points at the double computation boundary and preserves the input shape." },
    BuiltinDocumentationFaq { question: "How does provider fallback work?", answer: "A typed unsupported response triggers an owner-preserving host fallback. Malformed output and other provider failures are reported." },
    BuiltinDocumentationFaq { question: "Can tanh participate in fusion?", answer: "Yes, for supported real floating elementwise expressions. Complex and conversion paths remain runtime boundaries." },
    BuiltinDocumentationFaq { question: "What happens for large values and empty arrays?", answer: "Large finite real values approach -1 or 1. Empty arrays and singleton dimensions retain their shape." },
    BuiltinDocumentationFaq { question: "Does tanh provide automatic differentiation?", answer: "Not currently. This builtin exposes runtime and acceleration contracts but does not claim a public automatic-differentiation API." },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "sinh", target: BuiltinDocumentationLinkTarget::Builtin("sinh") },
    BuiltinDocumentationLink { label: "cosh", target: BuiltinDocumentationLinkTarget::Builtin("cosh") },
    BuiltinDocumentationLink { label: "atanh", target: BuiltinDocumentationLinkTarget::Builtin("atanh") },
    BuiltinDocumentationLink { label: "acosh", target: BuiltinDocumentationLinkTarget::Builtin("acosh") },
    BuiltinDocumentationLink { label: "asinh", target: BuiltinDocumentationLinkTarget::Builtin("asinh") },
    BuiltinDocumentationLink { label: "sin", target: BuiltinDocumentationLinkTarget::Builtin("sin") },
    BuiltinDocumentationLink { label: "cos", target: BuiltinDocumentationLinkTarget::Builtin("cos") },
    BuiltinDocumentationLink { label: "tan", target: BuiltinDocumentationLinkTarget::Builtin("tan") },
    BuiltinDocumentationLink { label: "asin", target: BuiltinDocumentationLinkTarget::Builtin("asin") },
    BuiltinDocumentationLink { label: "acos", target: BuiltinDocumentationLinkTarget::Builtin("acos") },
    BuiltinDocumentationLink { label: "atan", target: BuiltinDocumentationLinkTarget::Builtin("atan") },
    BuiltinDocumentationLink { label: "atan2", target: BuiltinDocumentationLinkTarget::Builtin("atan2") },
    BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") },
    BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Builtin("gather") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/trigonometry/tanh.rs") },
];

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Hyperbolic-tangent runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/trigonometry/tanh.rs") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Real, complex, typed, shape, extension, and error behavior", location: "crates/runmat-runtime/src/builtins/math/trigonometry/tanh.rs::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::ProviderTest, label: "Owner-preserving provider execution and fallback", location: "crates/runmat-runtime/src/builtins/math/trigonometry/tanh.rs::tests::tanh_gpu_provider_roundtrip" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::WgpuTest, label: "Actual WGPU parity", location: "crates/runmat-runtime/src/builtins/math/trigonometry/tanh.rs::tests::tanh_wgpu_matches_cpu_elementwise" },
    ],
    notes: &[],
};

pub(crate) const TANH_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("tanh"),
    slug: Some("tanh"),
    summary: "Compute elementwise hyperbolic tangent for real and complex values.",
    description: "`tanh` preserves shape and floating precision, supports complex input, and retains supported provider ownership.",
    keywords: &["tanh", "hyperbolic tangent", "trigonometry", "elementwise", "complex", "gpu"],
    related: &["sinh", "cosh", "atanh", "acosh", "asinh", "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "gpuArray", "gather"],
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
