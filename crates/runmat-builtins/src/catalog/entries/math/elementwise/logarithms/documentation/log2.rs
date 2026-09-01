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
            "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/log2.rs",
        ),
    }],
    verification: &[
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "One-output and dissection tests",
            location: "builtins::math::elementwise::log2::tests",
        },
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::ProviderTest,
            label: "Provider round-trip test",
            location: "builtins::math::elementwise::log2::tests::log2_gpu_provider_roundtrip",
        },
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::ProviderTest,
            label: "WGPU elementwise parity test",
            location: "builtins::math::elementwise::log2::tests::log2_wgpu_matches_cpu_elementwise",
        },
    ],
    notes: &[],
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "One-output logarithm",
        paragraphs: &[
            "`Y = log2(X)` computes the base-2 logarithm elementwise and preserves shape. Real and complex single inputs return single; real and complex double inputs return double.",
            "Zero maps to negative infinity, positive infinity remains positive infinity, and NaN propagates. Negative real values promote to principal-branch complex results; complex values use `log(z) / log(2)`.",
            "Fixed-width integer, logical, and character inputs are separate RunMat extensions that enter the double domain. Integer values must lie in the inclusive exact binary64 interval `[-2^53, 2^53]`.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Floating-point dissection",
        paragraphs: &[
            "`[F,E] = log2(X)` decomposes real floating-point values so that `X = F .* 2.^E`. Finite nonzero values normally satisfy `0.5 <= abs(F) < 1`; zero returns `F = 0, E = 0`; infinities and NaN remain in `F` with `E = 0`.",
            "Both outputs preserve the input shape and floating-point class. The current compatibility contract rejects complex dissection. Tables and timetables are mapped variable by variable while retaining their organization.",
            "The two-output form is host-only. Its integer, logical, and character forms are RunMat extensions.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "GPU execution",
        paragraphs: &[
            "The one-output form keeps supported real values on their exact owner through the provider unary-log2 operation and validates the returned handle's shape, storage, precision, device, and ownership.",
            "Typed unsupported operations may use the owner-aware host path. Explicit real `gpuArray` input that would require complex promotion must be made explicitly complex; RunMat does not silently change its residency contract. The two-output form requires host input.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "powers-of-two",
        title: "Compute logarithms of powers of two",
        program: "values = [1 2 4 8];\npowers = log2(values)",
        display_output: Some("powers = [0 1 2 3]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(max(abs(powers - [0 1 2 3])) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "zero",
        title: "Evaluate the base-2 logarithm at zero",
        program: "z = log2(0)",
        display_output: Some("z = -Inf"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isinf(z) && z < 0);",
        },
    },
    BuiltinExample {
        id: "negative-values",
        title: "Promote negative real values to complex results",
        program: "neg = [-1 -2];\nout = log2(neg)",
        display_output: Some("out = [0.0000 + 4.5324i 1.0000 + 4.5324i]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "expected = [0 1] + (pi/log(2))*1i;\nassert(max(abs(out - expected)) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "matrix-exponents",
        title: "Find power-of-two exponents for a matrix",
        program: "A = [64 128; 256 512];\nexponents = log2(A)",
        display_output: Some("exponents = [6 7; 8 9]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "expected = [6 7; 8 9];\nassert(max(abs(exponents(:) - expected(:))) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "dissection",
        title: "Dissect values into binary fractions and exponents",
        program: "X = [1 -3 0 Inf];\n[F,E] = log2(X)",
        display_output: Some("F = [0.5 -0.75 0 Inf]\nE = [1 2 0 0]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(F, [0.5 -0.75 0 Inf]));\nassert(isequal(E, [1 2 0 0]));",
        },
    },
    BuiltinExample {
        id: "gpu-array",
        title: "Apply the base-2 logarithm to provider-resident data",
        program: "G = gpuArray([1 4 16 64]);\nresult = log2(G);\nhost = gather(result)",
        display_output: Some("host = [0 2 4 6]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(max(abs(host - [0 2 4 6])) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "character-codes",
        title: "Use character code points in RunMat mode",
        program: "C = 'ABC';\nvalues = log2(C)",
        display_output: Some("values = [6.0224 6.0444 6.0661]"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "expected = log2([65 66 67]);\nassert(max(abs(values - expected)) < 1e-12);",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "When should I use `log2` instead of `log` or `log10`?",
        answer: "Use `log2` for binary scaling, including bit widths, FFT sizes, and exponent analysis. Use `log` for natural logarithms and `log10` for decimal magnitudes.",
    },
    BuiltinDocumentationFaq {
        question: "What happens when an element is zero?",
        answer: "`log2(0)` returns negative infinity.",
    },
    BuiltinDocumentationFaq {
        question: "How does `log2` handle negative real values?",
        answer: "The one-output form promotes negative real values to principal-branch complex results with imaginary component `pi/log(2)`.",
    },
    BuiltinDocumentationFaq {
        question: "Can I pass complex input to `log2`?",
        answer: "The one-output form accepts complex values and computes `log(z) / log(2)`. The current compatibility contract rejects complex input in the two-output dissection form.",
    },
    BuiltinDocumentationFaq {
        question: "Does GPU execution support complex promotion?",
        answer: "The real provider hook cannot create complex output. Explicit real `gpuArray` input that would require promotion must be converted to an explicitly complex GPU value before calling `log2`.",
    },
    BuiltinDocumentationFaq {
        question: "How is precision selected?",
        answer: "Single input returns single and double input returns double. Integer, logical, and character extensions enter the double domain. Tiny finite imaginary roundoff may be normalized to zero.",
    },
];

pub(in super::super) const LOG2_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("log2"),
    slug: Some("log2"),
    summary: "Compute base-2 logarithms or dissect floating-point values.",
    description: "`Y = log2(X)` computes base-2 logarithms elementwise. `[F,E] = log2(X)` decomposes real values into binary fractions and exponents while preserving shape and floating-point class.",
    keywords: &[
        "log2",
        "base-2 logarithm",
        "floating-point dissection",
        "fraction",
        "exponent",
        "gpu",
        "complex",
    ],
    related: &[
        "abs", "angle", "conj", "double", "exp", "expm1", "factorial", "gamma",
        "gather", "gpuArray", "hypot", "imag", "ldivide", "log", "log1p", "log10",
        "minus", "nextpow2", "plus", "pow2", "power", "rdivide", "real", "sign",
        "single", "sqrt", "times",
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
