use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/complex_components/real") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "CPU and representation tests", location: "builtins::math::elementwise::complex_components::real::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::ProviderTest, label: "Provider round-trip test", location: "builtins::math::elementwise::complex_components::real::tests::provider::real_gpu_provider_roundtrip" },
    ],
    notes: &[],
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`Y = real(X)` extracts each real component and preserves shape. Complex input drops its imaginary component; real floating-point and integer input preserves its class.",
        "All eight real integer classes are exact identities. Paired complex-integer arrays project authoritative real storage without arithmetic. Logical and character input returns double values.",
        "String arrays are unsupported. Sparse input is currently densified.",
    ] },
    BuiltinDocumentationSection { heading: "GPU execution", paragraphs: &[
        "Supported providers extract the real lane on the owning device and preserve shape, class, owner, and residency. Exact integer input remains class-preserving.",
        "Typed unsupported operations use an owner-aware download and restoration path. Fusion may combine real projection with adjacent elementwise operations when the representation permits it.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "complex-scalar",
        title: "Extract the real part of a complex scalar",
        program: "z = 3 + 4i;\nr = real(z)",
        display_output: Some("r = 3"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(r == 3);",
        },
    },
    BuiltinExample {
        id: "complex-matrix",
        title: "Extract real components from a complex matrix",
        program: "Z = [1+2i 4-3i; -5+0i 7+8i];\nR = real(Z)",
        display_output: Some("R = [1 4; -5 7]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(R, [1 4; -5 7]));",
        },
    },
    BuiltinExample {
        id: "real-identity",
        title: "Leave real input unchanged",
        program: "data = [-2.5 0 9.75];\nresult = real(data)",
        display_output: Some("result = [-2.5 0 9.75]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(result, data));",
        },
    },
    BuiltinExample {
        id: "logical-values",
        title: "Convert a logical mask to double",
        program: "mask = logical([0 1 0; 1 1 0]);\nnumeric = real(mask)",
        display_output: Some("numeric = [0 1 0; 1 1 0]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(numeric, \"double\"));\nassert(isequal(numeric, [0 1 0; 1 1 0]));",
        },
    },
    BuiltinExample {
        id: "character-codes",
        title: "Convert characters to numeric code points",
        program: "chars = 'RunMat';\ncodes = real(chars)",
        display_output: Some("codes = [82 117 110 77 97 116]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source:
                "assert(isa(codes, \"double\"));\nassert(isequal(codes, [82 117 110 77 97 116]));",
        },
    },
    BuiltinExample {
        id: "gpu-array",
        title: "Extract real values from provider-resident data",
        program: "G = gpuArray([1 -2; 3 -4]);\nR = real(G);\nhost = gather(R)",
        display_output: Some("host = [1 -2; 3 -4]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(host, [1 -2; 3 -4]));",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Does `real` change real input?", answer: "Real floating-point and integer input retains its class and value. Logical and character values retain their numeric values but return double." },
    BuiltinDocumentationFaq { question: "How does `real` handle complex zero?", answer: "`real(0 + 0i)` returns exactly zero; the imaginary zero is discarded." },
    BuiltinDocumentationFaq { question: "Can I call `real` on string arrays?", answer: "No. `real` accepts numeric, logical, and character arrays." },
    BuiltinDocumentationFaq { question: "Does `real` allocate a new array?", answer: "It returns a distinct value when projection requires one. Identity and fusion paths can avoid materializing an intermediate." },
    BuiltinDocumentationFaq { question: "What happens when a provider lacks real projection?", answer: "RunMat gathers through the exact owner, applies host semantics, and restores the result when its class is representable." },
    BuiltinDocumentationFaq { question: "Does GPU execution match CPU behavior?", answer: "Yes. Results retain the provider precision and exact component values." },
    BuiltinDocumentationFaq { question: "Can `real` participate in fusion?", answer: "Yes. The planner can fold supported real projection into adjacent elementwise kernels." },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("real"),
    slug: Some("real"),
    summary: "Extract real components from numeric, logical, character, or complex values.",
    description: "`real(X)` returns each real component. Real-valued input passes through numerically, while complex input drops its imaginary component with class-preserving numeric rules.",
    keywords: &["real", "complex", "component", "elementwise", "gpu"],
    related: &["abs", "angle", "complex", "conj", "double", "gather", "gpuArray", "imag", "sign", "single"],
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
