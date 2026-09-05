use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/magnitude_phase_sign/sign/mod.rs") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "CPU and representation tests", location: "builtins::math::elementwise::magnitude_phase_sign::sign::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::ProviderTest, label: "Provider round-trip test", location: "builtins::math::elementwise::magnitude_phase_sign::sign::tests::provider::sign_gpu_provider_roundtrip" },
    ],
    notes: &[],
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`Y = sign(X)` maps real values to `-1`, `0`, or `1` and normalizes nonzero complex floating values as `X ./ abs(X)`. Complex zero remains complex zero; NaN propagates; positive and negative infinity map to `1` and `-1`.",
        "All eight fixed-width integer classes preserve class and shape and are transformed directly in native storage without overflow. Signed values map to `-1`, `0`, or `1`; unsigned values map to `0` or `1`.",
        "Logical and character input returns double results. Strings are unsupported. Complex values with infinite components normalize to their unit direction.",
    ] },
    BuiltinDocumentationSection { heading: "GPU execution", paragraphs: &[
        "Providers with unary sign support keep real and complex-interleaved tensors resident, including complex unit normalization.",
        "Typed unsupported operations gather through the exact owner, apply the same CPU contract, and restore representable output. Integer fallback preserves exact fixed-width class. Fusion can combine supported elementwise work.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "scalar",
        title: "Find the sign of a scalar",
        program: "result = sign(-42)",
        display_output: Some("result = -1"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(result == -1);",
        },
    },
    BuiltinExample {
        id: "mixed-vector",
        title: "Apply `sign` to mixed real values",
        program: "v = [-3 -0.0 0 2 5];\ns = sign(v)",
        display_output: Some("s = [-1 0 0 1 1]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(s, [-1 0 0 1 1]));",
        },
    },
    BuiltinExample {
        id: "complex-values",
        title: "Normalize complex values to unit magnitude",
        program: "z = [3+4i -1+1i 0+0i];\nu = sign(z)",
        display_output: Some("u = [0.6+0.8i -0.7071+0.7071i 0]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source:
                "expected = [0.6+0.8i (-1+1i)/sqrt(2) 0];\nassert(max(abs(u - expected)) < 1e-12);",
        },
    },
    BuiltinExample {
        id: "character-values",
        title: "Apply `sign` to character code points",
        program: "codes = sign('RunMat')",
        display_output: Some("codes = [1 1 1 1 1 1]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(codes, \"double\"));\nassert(isequal(codes, ones(1, 6)));",
        },
    },
    BuiltinExample {
        id: "logical-values",
        title: "Apply `sign` to a logical mask",
        program: "mask = [false true false; true false true];\nnumeric = sign(mask)",
        display_output: Some("numeric = [0 1 0; 1 0 1]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(numeric, \"double\"));\nassert(isequal(numeric, [0 1 0; 1 0 1]));",
        },
    },
    BuiltinExample {
        id: "gpu-array",
        title: "Apply `sign` to provider-resident data",
        program: "G = gpuArray([-3 0; 2 -1]);\nS = sign(G);\nhost = gather(S)",
        display_output: Some("host = [-1 0; 1 -1]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(host, [-1 0; 1 -1]));",
        },
    },
    BuiltinExample {
        id: "special-values",
        title: "Handle infinities and NaN",
        program: "values = [Inf -Inf NaN 0];\nout = sign(values)",
        display_output: Some("out = [1 -1 NaN 0]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(out(1) == 1 && out(2) == -1 && isnan(out(3)) && out(4) == 0);",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Does `sign` change NaN?", answer: "No. NaN propagates under IEEE rules." },
    BuiltinDocumentationFaq { question: "How does `sign` handle complex zero?", answer: "Complex zero remains complex zero; other complex values normalize to the unit circle." },
    BuiltinDocumentationFaq { question: "What happens for infinite complex components?", answer: "RunMat returns the corresponding unit direction, including diagonal directions when both components are infinite." },
    BuiltinDocumentationFaq { question: "Can I call `sign` on strings?", answer: "No. `sign` accepts numeric, logical, or character arrays." },
    BuiltinDocumentationFaq { question: "Does `sign` allocate?", answer: "It returns a new value; fusion can avoid materializing an intermediate." },
    BuiltinDocumentationFaq { question: "Does GPU execution match CPU behavior?", answer: "Yes within the provider precision, including NaN propagation and zero handling." },
    BuiltinDocumentationFaq { question: "Can `sign` participate in fusion?", answer: "Yes. Supported elementwise sign operations can be folded into neighboring kernels." },
    BuiltinDocumentationFaq { question: "How do I keep the result on the GPU?", answer: "Avoid `gather` until host data is required; supported outputs remain resident." },
];

pub(in super::super) const SIGN_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("sign"), slug: Some("sign"),
    summary: "Compute elementwise sign values for real and complex input.",
    description: "`sign(X)` returns real sign values or complex unit directions while preserving documented numeric classes and shape.",
    keywords: &["sign", "signum", "unit direction", "complex", "integer", "gpu"],
    related: &["abs", "angle", "conj", "double", "gather", "gpuArray", "imag", "real", "single"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None, faqs: FAQS,
    links: &[], media: &[], evidence: EVIDENCE, introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
