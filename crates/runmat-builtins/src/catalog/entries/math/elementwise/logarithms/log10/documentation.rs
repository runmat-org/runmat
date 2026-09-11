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
            "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/logarithms/log10/mod.rs",
        ),
    }],
    verification: &[
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "CPU and representation tests",
            location: "builtins::math::elementwise::logarithms::log10::tests",
        },
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::ProviderTest,
            label: "Provider round-trip test",
            location: "builtins::math::elementwise::logarithms::log10::tests::provider::provider_result_remains_resident_and_matches_host",
        },
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::ProviderTest,
            label: "WGPU elementwise parity test",
            location: "builtins::math::elementwise::logarithms::log10::tests::provider::log10_wgpu_matches_cpu_elementwise",
        },
    ],
    notes: &[],
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "`Y = log10(X)` computes the base-10 logarithm elementwise and preserves the input shape. Real and complex single inputs return single; real and complex double inputs return double.",
            "Zero maps to negative infinity. Negative real values promote to principal-branch complex results, and complex values use `log10(z) = log(z) / log(10)`.",
            "Tables and timetables are mapped variable by variable when each variable supports `log10`.",
            "Fixed-width integer, logical, and character inputs are separate RunMat extensions. They return double or complex double, and MATLAB compatibility mode rejects each extension with its structured compatibility error.",
            "Integer extension inputs must lie in the inclusive exact binary64 interval `[-2^53, 2^53]`; RunMat rejects a wider value before conversion.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "GPU execution",
        paragraphs: &[
            "RunMat resolves the exact provider that owns a resident input. Supported real values can stay resident through provider minimum-reduction and unary-log10 operations, and returned handles are validated for shape, storage, precision, device, and ownership.",
            "Complex input, real-to-complex promotion, and typed unsupported operations gather through that owner and restore the result when representable. Other provider failures remain errors. Explicit real `gpuArray` input that requires complex promotion is available only in RunMat compatibility mode.",
            "`log10` is not fused until fused execution can retain its complex-domain behavior.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "order-of-magnitude",
        title: "Find the order of magnitude of a number",
        program: "value = log10(1000)",
        display_output: Some("value = 3"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(abs(value - 3) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "matrix",
        title: "Compute base-10 logarithms of a matrix",
        program: "A = [1 10 100; 0.1 0.01 0.001];\nB = log10(A)",
        display_output: Some("B = [0 1 2; -1 -2 -3]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(max(abs(B(:) - [0; -1; 1; -2; 2; -3])) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "zero",
        title: "Evaluate the base-10 logarithm at zero",
        program: "z = log10(0)",
        display_output: Some("z = -Inf"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isinf(z) && z < 0);",
        },
    },
    BuiltinExample {
        id: "negative-values",
        title: "Promote negative real values to complex results",
        program: "neg = [-10 -100];\nout = log10(neg)",
        display_output: Some("out = [1.0000 + 1.3644i 2.0000 + 1.3644i]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source:
                "expected = [1 2] + (pi/log(10))*1i;\nassert(max(abs(out - expected)) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "gpu-array",
        title: "Apply the base-10 logarithm to provider-resident data",
        program: "G = gpuArray([1 10 1000]);\nresult = log10(G);\nhost = gather(result)",
        display_output: Some("host = [0 1 3]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(max(abs(host - [0 1 3])) < 1e-12);",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "When should I use `log10` instead of `log`?",
        answer: "Use `log10` for decimal magnitudes, including scientific notation and decibel calculations. Use `log` for natural-logarithm calculations such as exponential growth and calculus.",
    },
    BuiltinDocumentationFaq {
        question: "What happens when an element is zero?",
        answer: "`log10(0)` returns negative infinity.",
    },
    BuiltinDocumentationFaq {
        question: "How does `log10` handle negative real values?",
        answer: "Negative real values promote to complex results with imaginary component `pi/log(10)` on the principal branch.",
    },
    BuiltinDocumentationFaq {
        question: "Can I pass complex input to `log10`?",
        answer: "Yes. Complex scalars and arrays are evaluated as `log(z) / log(10)`.",
    },
    BuiltinDocumentationFaq {
        question: "How does GPU execution produce complex output?",
        answer: "RunMat gathers through the exact owner, computes the complex result on the host, and restores it when that provider represents the required complex class. MATLAB compatibility mode rejects complex promotion from explicit real `gpuArray` input.",
    },
    BuiltinDocumentationFaq {
        question: "How is precision selected?",
        answer: "Double input returns double and single input returns single. Integer extensions enter the double domain only when every value is exactly representable in binary64. Tiny finite roundoff components may be normalized to zero.",
    },
];

pub(in super::super) const LOG10_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("log10"),
    slug: Some("log10"),
    summary: "Compute elementwise base-10 logarithms for real and complex values.",
    description: "`Y = log10(X)` computes the base-10 logarithm of each element, including principal-branch complex promotion and typed RunMat extensions for integer, logical, and character input.",
    keywords: &[
        "log10",
        "common logarithm",
        "base 10",
        "elementwise",
        "complex",
        "gpu",
    ],
    related: &[
        "abs", "angle", "conj", "double", "exp", "expm1", "factorial", "gamma",
        "gather", "gpuArray", "hypot", "imag", "ldivide", "log", "log1p", "log2",
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
