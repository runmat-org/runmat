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
            "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/exponentials/exp/mod.rs",
        ),
    }],
    verification: &[
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "CPU and representation tests",
            location: "builtins::math::elementwise::exponentials::exp::tests::host",
        },
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::ProviderTest,
            label: "Provider round-trip test",
            location: "builtins::math::elementwise::exponentials::exp::tests::provider",
        },
    ],
    notes: &[],
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "`Y = exp(X)` raises e to each element of `X` while preserving the shape. Real and complex single inputs return single; real and complex double inputs return double.",
            "Complex values use the principal exponential identity `exp(a + bi) = exp(a) * (cos(b) + i sin(b))`.",
            "Sparse floating-point input becomes dense because every implicit zero maps to one. Table and timetable input is mapped variable by variable only when each variable supports `exp`.",
            "Fixed-width integer, logical, and character inputs are separate RunMat extensions. They produce double output and MATLAB compatibility mode rejects each extension with its corresponding structured compatibility error.",
            "Integer extension inputs must lie in the inclusive interval `[-2^53, 2^53]`. RunMat rejects a wider integer before conversion rather than silently rounding it at the binary64 boundary.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "GPU execution",
        paragraphs: &[
            "Real floating-point tensors can remain on the owning provider through its unary exponential operation. Complex values and providers that explicitly report an unsupported operation use an owner-aware host fallback; other provider failures remain errors.",
            "A host fallback restores the result to the original provider only when that provider supports the result precision. Explicit `gpuArray` and `gather` calls remain available when placement is part of the program.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "scalar",
        title: "Calculate the exponential of a scalar",
        program: "y = exp(1)",
        display_output: Some("y = 2.7183"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(abs(y - 2.718281828459045) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "growth-rates",
        title: "Apply the exponential to a vector of growth rates",
        program: "rates = [-1 -0.5 0 0.5 1];\nfactor = exp(rates)",
        display_output: Some("factor = [0.3679 0.6065 1 1.6487 2.7183]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "expected = [0.367879441171442 0.606530659712633 1 1.648721270700128 2.718281828459045];\nassert(max(abs(factor - expected)) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "matrix",
        title: "Exponentiate every element of a matrix",
        program: "A = [0 1 2; 3 4 5];\nB = exp(A)",
        display_output: Some(
            "B = [1.0000 2.7183 7.3891; 20.0855 54.5982 148.4132]",
        ),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "expected = [1 2.718281828459045 7.389056098930650; 20.085536923187668 54.598150033144236 148.41315910257660];\nassert(max(abs(B(:) - expected(:))) < 1e-11);",
        },
    },
    BuiltinExample {
        id: "complex-values",
        title: "Compute exponentials of complex values",
        program: "z = [1+2i, -1+pi*i];\nw = exp(z)",
        display_output: Some("w = [-1.1312 + 2.4717i, -0.3679 + 0.0000i]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "expected = [-1.131204383756814 + 2.471726672004819i, -0.367879441171442];\nassert(max(abs(w - expected)) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "gpu-array",
        title: "Apply the exponential to provider-resident data",
        program: "G = gpuArray([0 1; 2 3]);\nout = exp(G);\nresult = gather(out)",
        display_output: Some("result = [1.0000 2.7183; 7.3891 20.0855]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        verification: BuiltinExampleVerification::Assertions {
            source: "expected = [1 2.718281828459045; 7.389056098930650 20.085536923187668];\nassert(max(abs(result(:) - expected(:))) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "character-codes",
        title: "Exponentiate character code points in RunMat mode",
        program: "C = 'ABC';\nY = exp(C)",
        display_output: Some("Y = [1.6949e+28 4.6072e+28 1.2524e+29]"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "expected = [1.694889244410334e28 4.607186634331292e28 1.252363170842214e29];\nassert(max(abs((Y - expected) ./ expected)) < 1e-12);",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "When should I use `exp`?",
        answer: "Use `exp` for the natural exponential of a scalar or array, including growth, continuous compounding, and exponential model terms.",
    },
    BuiltinDocumentationFaq {
        question: "Does `exp` preserve shape?",
        answer: "Yes. The result has the same shape as the input.",
    },
    BuiltinDocumentationFaq {
        question: "How are logical arrays handled?",
        answer: "Logical arrays convert to double only in RunMat compatibility mode. MATLAB compatibility mode rejects that extension.",
    },
    BuiltinDocumentationFaq {
        question: "What happens with complex input?",
        answer: "Complex scalars and arrays use the principal complex exponential and retain their floating-point precision.",
    },
    BuiltinDocumentationFaq {
        question: "What happens when a provider lacks its exponential operation?",
        answer: "RunMat gathers through the owning provider and computes the same result on the host. It restores the result only when the provider supports the required precision.",
    },
    BuiltinDocumentationFaq {
        question: "Which precision does `exp` return?",
        answer: "Single input returns single and double input returns double. Integer, logical, and character extension forms return double.",
    },
];

pub(super) const EXP_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("exp"),
    slug: Some("exp"),
    summary: "Compute elementwise natural exponentials for real and complex values.",
    description: "`Y = exp(X)` raises e to every element of a real or complex array. It preserves documented floating-point precision, handles complex values directly, and uses typed RunMat extensions for integer, logical, and character input.",
    keywords: &["exp", "exponential", "elementwise", "gpu", "complex"],
    related: &[
        "abs", "angle", "conj", "double", "expm1", "factorial", "gamma", "gather",
        "gpuArray", "hypot", "imag", "ldivide", "log", "log1p", "log2", "log10",
        "logspace", "minus", "plus", "pow2", "power", "rdivide", "real", "sign",
        "sin", "single", "sqrt", "times",
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
