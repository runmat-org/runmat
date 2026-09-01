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
            "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/log1p.rs",
        ),
    }],
    verification: &[
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "CPU and representation tests",
            location: "builtins::math::elementwise::log1p::tests",
        },
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::ProviderTest,
            label: "Provider round-trip test",
            location: "builtins::math::elementwise::log1p::tests::log1p_gpu_provider_roundtrip",
        },
    ],
    notes: &[],
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "`Y = log1p(X)` evaluates `log(1 + X)` elementwise without the cancellation introduced by forming `1 + X` near zero. It preserves shape and the single or double precision of floating-point input.",
            "A real value equal to `-1` maps to negative infinity. Real values below `-1` promote to principal-branch complex output. Complex values compute the natural logarithm of `1 + z`.",
            "Fixed-width integer, logical, and character inputs are separate RunMat extensions. They produce double or complex double, and MATLAB compatibility mode rejects each extension with its structured compatibility error.",
            "Integer extension inputs must lie in the inclusive exact binary64 interval `[-2^53, 2^53]`; RunMat rejects a wider value before conversion.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "GPU execution",
        paragraphs: &[
            "Supported real tensors can remain on their exact provider through minimum-reduction and unary-log1p operations. Provider results are validated for non-aliasing, shape, storage, precision, device, and owner.",
            "Complex input, real-to-complex promotion, and typed unsupported operations gather through the owner and restore the result when representable. Other provider failures remain errors. Explicit real `gpuArray` input that requires complex promotion is available only in RunMat compatibility mode.",
            "`log1p` is not fused because replacing it with a raw `log(1 + X)` expression would violate its near-zero accuracy contract.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "near-zero-accuracy",
        title: "Retain accuracy for a tiny increment",
        program: "delta = 1e-12;\nvalue = log1p(delta)",
        display_output: Some("value = 9.999999999995e-13"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(abs(value - 9.999999999995e-13) < 1e-27);",
        },
    },
    BuiltinExample {
        id: "growth-rates",
        title: "Convert percentage changes into log-growth factors",
        program: "rates = [-0.25 -0.10 0 0.10 0.25];\ngrowth = log1p(rates)",
        display_output: Some("growth = [-0.2877 -0.1054 0 0.0953 0.2231]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "expected = [-0.287682072451781 -0.105360515657826 0 0.095310179804325 0.223143551314210];\nassert(max(abs(growth - expected)) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "branch-point",
        title: "Evaluate the branch point at negative one",
        program: "y = log1p(-1)",
        display_output: Some("y = -Inf"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isinf(y) && y < 0);",
        },
    },
    BuiltinExample {
        id: "complex-promotion",
        title: "Promote values below negative one to complex results",
        program: "data = [-2 -3 -5];\nresult = log1p(data)",
        display_output: Some("result = [0.0000 + 3.1416i 0.6931 + 3.1416i 1.3863 + 3.1416i]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "expected = [pi*1i log(2)+pi*1i log(4)+pi*1i];\nassert(max(abs(result - expected)) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "gpu-array",
        title: "Apply `log1p` to provider-resident data",
        program: "G = gpuArray(linspace(-0.5, 0.5, 5));\nout = log1p(G);\nrealResult = gather(out)",
        display_output: Some("realResult = [-0.6931 -0.2877 0 0.2231 0.4055]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        verification: BuiltinExampleVerification::Assertions {
            source: "expected = [-0.693147180559945 -0.287682072451781 0 0.223143551314210 0.405465108108164];\nassert(max(abs(realResult - expected)) < 1e-12);",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "When should I prefer `log1p` over `log(1 + X)`?",
        answer: "Use `log1p` whenever `X` can be close to zero. It avoids the significant-digit loss caused by rounding `1 + X` before taking the logarithm.",
    },
    BuiltinDocumentationFaq {
        question: "Does `log1p` preserve shape?",
        answer: "Yes. The result has the same shape as the input.",
    },
    BuiltinDocumentationFaq {
        question: "How are logical arrays handled?",
        answer: "Logical values convert to double only in RunMat compatibility mode; MATLAB compatibility mode rejects the extension.",
    },
    BuiltinDocumentationFaq {
        question: "What happens for values below `-1`?",
        answer: "Real values below `-1` promote to principal-branch complex results for `log(1 + X)`.",
    },
    BuiltinDocumentationFaq {
        question: "How does `log1p` handle complex values?",
        answer: "Complex scalars and arrays compute the principal natural logarithm of `1 + z`.",
    },
    BuiltinDocumentationFaq {
        question: "What happens when a provider lacks `unary_log1p`?",
        answer: "RunMat gathers through the exact owner, computes on the host in the required class, and restores the result when that provider can represent it. Genuine provider failures remain errors.",
    },
    BuiltinDocumentationFaq {
        question: "Is double precision always used?",
        answer: "No. Double input returns double and single input returns single. Integer, logical, and character extensions enter the double domain.",
    },
    BuiltinDocumentationFaq {
        question: "Can `log1p` participate in fusion?",
        answer: "Not currently. A fused `log(1 + X)` replacement would lose the near-zero accuracy that defines `log1p`.",
    },
    BuiltinDocumentationFaq {
        question: "What is the inverse of `log1p`?",
        answer: "Use `expm1(Y)`, which computes `exp(Y) - 1` accurately near zero. The two functions form a numerically stable pair up to floating-point rounding.",
    },
    BuiltinDocumentationFaq {
        question: "Where is `log1p` commonly useful?",
        answer: "Typical uses include small financial returns, log-probability calculations, perturbation terms, entropy and softplus gradients, and accurate evaluation of `log(1 - p)` as `log1p(-p)`.",
    },
    BuiltinDocumentationFaq {
        question: "How much accuracy can direct `log(1 + X)` lose?",
        answer: "When `X` approaches machine precision, forming `1 + X` can discard most or all of `X` before the logarithm runs. `log1p` avoids that intermediate rounding and remains accurate near zero.",
    },
];

pub(in super::super) const LOG1P_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("log1p"),
    slug: Some("log1p"),
    summary: "Compute `log(1 + X)` elementwise with near-zero accuracy.",
    description: "`Y = log1p(X)` computes the natural logarithm of one plus each element while retaining accuracy near zero and applying principal-branch complex promotion where required.",
    keywords: &[
        "log1p",
        "natural logarithm",
        "elementwise",
        "precision",
        "complex",
        "gpu",
    ],
    related: &[
        "abs", "angle", "conj", "double", "exp", "expm1", "factorial", "gamma",
        "gather", "gpuArray", "hypot", "imag", "ldivide", "log", "log2", "log10",
        "minus", "plus", "pow2", "power", "rdivide", "real", "sign", "sin", "single",
        "sqrt", "times",
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
