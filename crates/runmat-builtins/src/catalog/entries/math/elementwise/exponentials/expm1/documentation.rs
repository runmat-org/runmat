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
            "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/exponentials/expm1/mod.rs",
        ),
    }],
    verification: &[
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "CPU and representation tests",
            location: "builtins::math::elementwise::exponentials::expm1::tests::host",
        },
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::ProviderTest,
            label: "Provider round-trip test",
            location: "builtins::math::elementwise::exponentials::expm1::tests::provider",
        },
    ],
    notes: &[],
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "`Y = expm1(X)` evaluates `exp(X) - 1` elementwise without the cancellation that direct subtraction introduces near zero. It preserves the shape and the single or double precision of real and complex floating-point input.",
            "Complex input uses the principal complex exponential before subtracting one. Sparse floating-point input remains sparse because each implicit zero maps back to zero. Table and timetable input is mapped variable by variable only when each variable supports `expm1`.",
            "Fixed-width integer, logical, and character inputs are separate RunMat extensions. They produce double output and MATLAB compatibility mode rejects each extension with its corresponding structured compatibility error.",
            "Integer extension inputs must lie in the inclusive interval `[-2^53, 2^53]`. RunMat rejects a wider integer before conversion rather than silently rounding it at the binary64 boundary.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "GPU execution",
        paragraphs: &[
            "Real floating-point tensors can remain on the owning provider through its precise unary `expm1` operation. Complex values and providers that explicitly report an unsupported operation use an owner-aware host fallback; other provider failures remain errors.",
            "`expm1` is not fused until the fusion ABI can retain its near-zero accuracy. Replacing it with the expression `exp(X) - 1` would change the documented numerical contract.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "near-zero-accuracy",
        title: "Retain accuracy for a tiny growth rate",
        program: "x = 1e-12;\ny = expm1(x)",
        display_output: Some("y = 1.0000000000005e-12"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(abs(y - 1.0000000000005e-12) < 1e-27);",
        },
    },
    BuiltinExample {
        id: "growth-rates",
        title: "Convert growth rates into relative factors",
        program: "rates = [-0.10 -0.05 0 0.05 0.10];\nfactors = expm1(rates)",
        display_output: Some("factors = [-0.0952 -0.0488 0 0.0513 0.1052]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "expected = [-0.095162581964040 -0.048770575499286 0 0.051271096376024 0.105170918075648];\nassert(max(abs(factors - expected)) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "gpu-array",
        title: "Apply `expm1` to provider-resident data",
        program: "G = gpuArray(linspace(-1, 1, 5));\nresult = expm1(G);\nout = gather(result)",
        display_output: Some("out = [-0.6321 -0.3935 0 0.6487 1.7183]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "expected = [-0.632120558828558 -0.393469340287367 0 0.648721270700128 1.718281828459045];\nassert(max(abs(out - expected)) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "complex-values",
        title: "Use `expm1` with complex values",
        program: "z = [1+1i, -1+pi*1i];\nw = expm1(z)",
        display_output: Some("w = [0.4687 + 2.2874i, -1.3679 + 0.0000i]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "expected = [0.468693939915885 + 2.287355287178842i, -1.367879441171442];\nassert(max(abs(w - expected)) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "character-codes",
        title: "Evaluate character code points in RunMat mode",
        program: "C = 'ABC';\nY = expm1(C)",
        display_output: Some("Y = [1.6949e+28 4.6072e+28 1.2524e+29]"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "expected = [1.694889244410334e28 4.607186634331292e28 1.252363170842214e29];\nassert(max(abs((Y - expected) ./ expected)) < 1e-12);",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "When should I prefer `expm1` over `exp(X) - 1`?",
        answer: "Use `expm1` whenever values can be close to zero. It avoids the significant-digit loss caused by subtracting one from a result close to one.",
    },
    BuiltinDocumentationFaq {
        question: "Does `expm1` preserve shape?",
        answer: "Yes. The result has the same shape as the input.",
    },
    BuiltinDocumentationFaq {
        question: "How are logical arrays handled?",
        answer: "Logical arrays convert to double only in RunMat compatibility mode. MATLAB compatibility mode rejects that extension.",
    },
    BuiltinDocumentationFaq {
        question: "What happens with complex input?",
        answer: "Complex values use the principal complex exponential and subtract one from the real component of the result.",
    },
    BuiltinDocumentationFaq {
        question: "What happens when a provider lacks its `expm1` operation?",
        answer: "RunMat gathers through the owning provider and computes the same result on the host. It restores the result only when the provider supports the required precision.",
    },
    BuiltinDocumentationFaq {
        question: "Which precision does `expm1` return?",
        answer: "Single input returns single and double input returns double. Integer, logical, and character extension forms return double.",
    },
    BuiltinDocumentationFaq {
        question: "Why is `expm1` not fused?",
        answer: "The current fusion expression would evaluate `exp(X) - 1` and lose the accuracy that defines `expm1`. Direct provider execution uses the precise operation instead.",
    },
];

pub(super) const EXPM1_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("expm1"),
    slug: Some("expm1"),
    summary: "Compute `exp(X) - 1` elementwise with near-zero accuracy.",
    description: "`Y = expm1(X)` evaluates the natural exponential minus one for every element of a real or complex array. Its dedicated numerical path retains accuracy when `X` is close to zero.",
    keywords: &[
        "expm1",
        "exp(x)-1",
        "exponential",
        "elementwise",
        "gpu",
        "precision",
    ],
    related: &[
        "abs", "angle", "conj", "double", "exp", "factorial", "gamma", "gather",
        "gpuArray", "hypot", "imag", "ldivide", "log", "log1p", "log2", "log10",
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
