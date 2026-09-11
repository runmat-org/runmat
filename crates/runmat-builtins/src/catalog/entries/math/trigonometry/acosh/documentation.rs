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
            "`Y = acosh(X)` computes the principal inverse hyperbolic cosine element by element and preserves the input shape. Real values greater than or equal to one remain real; real values below one produce complex output.",
            "Real and complex single input retain single precision. Real and complex double input retain double precision. Fixed-width integer, logical, and character input are RunMat extensions and enter an explicit double-precision computation boundary.",
            "Complex input follows the principal branch. `acosh(NaN)` is real NaN, positive infinity remains real infinity, and negative infinity produces positive real infinity with an imaginary component of pi.",
            "Sparse, table, and timetable input are not currently supported.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "GPU execution",
        paragraphs: &[
            "A provider can execute real floating input directly when domain inspection proves that every value is at least one. Provider output is validated before it becomes the result.",
            "Input that requires a complex result gathers through its owning provider, computes the principal result on the host, and restores the result to that same owner. A provider operation falls back only when it reports that the operation is unsupported; other provider failures remain errors.",
            "The value-dependent real-to-complex boundary prevents `acosh` from participating in ordinary real elementwise fusion. This avoids replacing required complex results with device NaNs for values below one.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "scalar",
        title: "Compute the inverse hyperbolic cosine of a scalar",
        program: "y = acosh(1.5)",
        display_output: Some("y = 0.9624"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(abs(y - 0.962423650119207) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "vector",
        title: "Apply acosh to a vector",
        program: "x = [1 1.5 2 4];\ny = acosh(x)",
        display_output: Some("y = [0 0.9624 1.3170 2.0634]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "expected = [0 0.962423650119207 1.316957896924817 2.063437068895561];\nassert(max(abs(y - expected)) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "complex-promotion",
        title: "Promote real values below one to complex results",
        program: "values = [0.5 1 2];\nz = acosh(values)",
        display_output: Some("z = [0.0000 + 1.0472i, 0.0000 + 0.0000i, 1.3170 + 0.0000i]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "expected = [1.047197551196598i 0 1.316957896924817];\nassert(max(abs(z - expected)) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "gpu-real-domain",
        title: "Keep real-domain input on its provider",
        program: "G = gpuArray([1 2 3 4 5]);\nresult_gpu = acosh(G);\nresult = gather(result_gpu)",
        display_output: Some("result = [0 1.3170 1.7627 2.0634 2.2924]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "expected = [0 1.316957896924817 1.762747174039086 2.063437068895561 2.292431669561177];\nassert(max(abs(result - expected)) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "complex-input",
        title: "Evaluate complex input on the principal branch",
        program: "z = [1 + 2i, -2 + 0.5i];\nw = acosh(z)",
        display_output: Some("w contains the principal complex inverse hyperbolic cosines"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(w), [1 2]));\nassert(max(abs(cosh(w) - z)) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "character-extension",
        title: "Evaluate character code points in RunMat mode",
        program: "C = char([0 65]);\nY = acosh(C)",
        display_output: Some("Y = [0.0000 + 1.5708i, 4.8675 + 0.0000i]"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "expected = [0 + 1.570796326794897i 4.867475273605342];\nassert(isequal(size(Y), [1 2]));\nassert(max(abs(Y - expected)) < 1e-12);",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Why does acosh sometimes return complex numbers?",
        answer: "The real inverse hyperbolic cosine is defined for values greater than or equal to one. RunMat returns the principal complex result for real values below one.",
    },
    BuiltinDocumentationFaq {
        question: "Can acosh run entirely on a GPU?",
        answer: "Yes, when domain inspection proves every real floating value is at least one and the provider implements the required operations. Values that need complex promotion use an owner-preserving host path.",
    },
    BuiltinDocumentationFaq {
        question: "How are NaN and infinity handled?",
        answer: "NaN returns real NaN, positive infinity returns real positive infinity, and negative infinity returns positive infinity plus pi times the imaginary unit.",
    },
    BuiltinDocumentationFaq {
        question: "Do logical and integer inputs work?",
        answer: "RunMat mode accepts logical input and all eight fixed-width integer classes as extensions. They produce double or complex-double output. MATLAB compatibility mode rejects these extension forms.",
    },
    BuiltinDocumentationFaq {
        question: "Can a complex result remain provider-resident?",
        answer: "Yes. The fallback gathers through the concrete owner and restores the complex result through that same provider rather than selecting an unrelated active provider.",
    },
    BuiltinDocumentationFaq {
        question: "Does acosh participate in elementwise fusion?",
        answer: "No. Real input can require complex promotion based on its values, so an ordinary real fused kernel cannot represent the complete contract safely.",
    },
    BuiltinDocumentationFaq {
        question: "Does domain inspection use a tolerance?",
        answer: "No. Every real value below one requires complex promotion; one and larger values remain real.",
    },
    BuiltinDocumentationFaq {
        question: "Does acosh preserve shape and precision?",
        answer: "Yes. It preserves shape and retains single or double precision for floating input. RunMat extension inputs use double precision.",
    },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "asinh", target: BuiltinDocumentationLinkTarget::Builtin("asinh") },
    BuiltinDocumentationLink { label: "atanh", target: BuiltinDocumentationLinkTarget::Builtin("atanh") },
    BuiltinDocumentationLink { label: "cosh", target: BuiltinDocumentationLinkTarget::Builtin("cosh") },
    BuiltinDocumentationLink { label: "sinh", target: BuiltinDocumentationLinkTarget::Builtin("sinh") },
    BuiltinDocumentationLink { label: "tanh", target: BuiltinDocumentationLinkTarget::Builtin("tanh") },
    BuiltinDocumentationLink { label: "acos", target: BuiltinDocumentationLinkTarget::Builtin("acos") },
    BuiltinDocumentationLink { label: "asin", target: BuiltinDocumentationLinkTarget::Builtin("asin") },
    BuiltinDocumentationLink { label: "atan", target: BuiltinDocumentationLinkTarget::Builtin("atan") },
    BuiltinDocumentationLink { label: "atan2", target: BuiltinDocumentationLinkTarget::Builtin("atan2") },
    BuiltinDocumentationLink { label: "cos", target: BuiltinDocumentationLinkTarget::Builtin("cos") },
    BuiltinDocumentationLink { label: "sin", target: BuiltinDocumentationLinkTarget::Builtin("sin") },
    BuiltinDocumentationLink { label: "tan", target: BuiltinDocumentationLinkTarget::Builtin("tan") },
    BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") },
    BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Builtin("gather") },
    BuiltinDocumentationLink {
        label: "Implementation",
        target: BuiltinDocumentationLinkTarget::Source(
            "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/trigonometry/acosh.rs",
        ),
    },
];

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink {
        label: "Inverse-hyperbolic-cosine runtime",
        target: BuiltinDocumentationLinkTarget::Source(
            "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/trigonometry/acosh.rs",
        ),
    }],
    verification: &[
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Real, complex, typed, domain, shape, extension, and error behavior",
            location: "crates/runmat-runtime/src/builtins/math/trigonometry/acosh.rs::tests",
        },
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::ProviderTest,
            label: "Owner-preserving provider execution and fallback",
            location: "crates/runmat-runtime/src/builtins/math/trigonometry/acosh.rs::tests::acosh_gpu_provider_roundtrip",
        },
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::WgpuTest,
            label: "Actual WGPU real-domain parity",
            location: "crates/runmat-runtime/src/builtins/math/trigonometry/acosh.rs::tests::acosh_wgpu_matches_cpu_when_real",
        },
    ],
    notes: &[],
};

pub(crate) const ACOSH_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("acosh"),
    slug: Some("acosh"),
    summary: "Compute elementwise principal inverse hyperbolic cosine with exact real-to-complex domain handling.",
    description: "`acosh` preserves shape and floating precision, promotes real values below one to principal complex results, and retains supported provider ownership.",
    keywords: &["acosh", "inverse hyperbolic cosine", "arccosh", "trigonometry", "complex", "gpu"],
    related: &["asinh", "atanh", "cosh", "sinh", "tanh", "acos", "asin", "atan", "atan2", "cos", "sin", "tan", "gpuArray", "gather"],
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
