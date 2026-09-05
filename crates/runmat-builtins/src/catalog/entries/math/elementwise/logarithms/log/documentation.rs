use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink {
        label: "Runtime implementation",
        target: BuiltinDocumentationLinkTarget::Source(
            "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/logarithms/log/mod.rs",
        ),
    }],
    verification: &[
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "CPU and representation tests",
            location: "builtins::math::elementwise::logarithms::log::tests",
        },
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::ProviderTest,
            label: "Provider round-trip test",
            location: "builtins::math::elementwise::logarithms::log::tests::provider::provider_result_remains_resident_and_matches_host",
        },
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::ProviderTest,
            label: "WGPU elementwise parity test",
            location: "builtins::math::elementwise::logarithms::log::tests::provider::log_wgpu_matches_cpu_elementwise",
        },
    ],
    notes: &[],
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "`Y = log(X)` computes the natural logarithm elementwise and preserves the input shape. Real and complex single inputs return single; real and complex double inputs return double.",
            "Zero maps to negative infinity. Negative real values promote to principal-branch complex results, and complex values use `log(z) = log(abs(z)) + i*angle(z)`.",
            "Tables and timetables are mapped variable by variable when each variable supports `log`.",
            "Fixed-width integer, logical, and character inputs are separate RunMat extensions. They return double or complex double, and MATLAB compatibility mode rejects each extension with its structured compatibility error.",
            "Integer extension inputs must lie in the inclusive exact binary64 interval `[-2^53, 2^53]`; RunMat rejects a wider value before conversion.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "GPU execution",
        paragraphs: &[
            "RunMat resolves the exact provider that owns a resident input. Supported real values can stay resident through provider minimum-reduction and unary-log operations. Every returned handle is validated for non-aliasing, shape, device, owner, storage, and precision.",
            "Complex input, real-to-complex promotion, and typed unsupported operations gather through that owner and restore the result when the provider can represent its class and precision. Other provider failures remain errors. Explicit real `gpuArray` input that requires complex promotion is available only in RunMat compatibility mode.",
            "`log` is not fused until fused execution can retain its complex-domain behavior.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "positive-scalar",
        title: "Take the natural logarithm of a positive scalar",
        program: "y = log(exp(3))",
        display_output: Some("y = 3"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(abs(y - 3) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "zero",
        title: "Evaluate the logarithm at zero",
        program: "value = log(0)",
        display_output: Some("value = -Inf"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isinf(value) && value < 0);",
        },
    },
    BuiltinExample {
        id: "negative-values",
        title: "Promote negative real values to complex results",
        program: "data = [-1 -2 -4];\nresult = log(data)",
        display_output: Some("result = [0.0000 + 3.1416i 0.6931 + 3.1416i 1.3863 + 3.1416i]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "expected = [pi*1i log(2)+pi*1i log(4)+pi*1i];\nassert(max(abs(result - expected)) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "complex-values",
        title: "Apply the logarithm to complex values",
        program: "z = [1+2i -1+pi*i];\nw = log(z)",
        display_output: Some("w = [0.8047 + 1.1071i 1.1920 + 1.8780i]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(max(abs(exp(w) - z)) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "gpu-array",
        title: "Apply the natural logarithm to provider-resident data",
        program: "G = gpuArray([1 2; 4 8]);\nout = log(G);\nresult = gather(out)",
        display_output: Some("result = [0.0000 0.6931; 1.3863 2.0794]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        verification: BuiltinExampleVerification::Assertions {
            source: "expected = [0 log(2); log(4) log(8)];\nassert(max(abs(result(:) - expected(:))) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "character-codes",
        title: "Use character code points in RunMat mode",
        program: "C = 'ABC';\nvalues = log(C)",
        display_output: Some("values = [4.1744 4.1897 4.2047]"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "expected = log([65 66 67]);\nassert(max(abs(values - expected)) < 1e-12);",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "When should I use `log`?",
        answer: "Use `log` for natural logarithms, including exponential-growth linearization, likelihood calculations, and transformations from multiplicative to additive relationships.",
    },
    BuiltinDocumentationFaq {
        question: "What happens when the input contains zero?",
        answer: "`log(0)` returns negative infinity. Arrays apply the same rule elementwise.",
    },
    BuiltinDocumentationFaq {
        question: "How are negative real values handled?",
        answer: "Negative real values promote to principal-branch complex results: `log(-x)` is `log(x) + i*pi` for positive `x`.",
    },
    BuiltinDocumentationFaq {
        question: "What if floating-point noise produces a small negative value?",
        answer: "Any negative real value, including a small rounding residual, promotes to a complex result. Clamp or transform the data explicitly when the application requires a real-only result.",
    },
    BuiltinDocumentationFaq {
        question: "How does GPU execution produce complex output?",
        answer: "A real provider hook cannot create a complex result. RunMat gathers through the exact owner, computes on the host, and restores the result when that provider represents the required complex class. MATLAB compatibility mode rejects complex promotion from explicit real `gpuArray` input.",
    },
    BuiltinDocumentationFaq {
        question: "Does `log` accept complex input directly?",
        answer: "Yes. Complex scalars and arrays use magnitude and phase on the principal branch.",
    },
];

pub(in super::super) const LOG_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("log"),
    slug: Some("log"),
    summary: "Compute elementwise natural logarithms for real and complex values.",
    description: "`Y = log(X)` computes the natural logarithm of each element, including principal-branch complex promotion and typed RunMat extensions for integer, logical, and character input.",
    keywords: &["log", "natural logarithm", "elementwise", "complex", "gpu"],
    related: &[
        "abs", "angle", "conj", "double", "exp", "expm1", "factorial", "gamma",
        "gather", "gpuArray", "hypot", "imag", "ldivide", "log1p", "log2", "log10",
        "minus", "plus", "pow2", "power", "rdivide", "real", "sign", "single", "sqrt",
        "times",
    ],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: &[],
    media: &[],
    evidence: EVIDENCE,
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
