use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/complex_components/imag") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "CPU and representation tests", location: "builtins::math::elementwise::complex_components::imag::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::ProviderTest, label: "Provider round-trip test", location: "builtins::math::elementwise::complex_components::imag::tests::provider::imag_gpu_provider_roundtrip" },
    ],
    notes: &[],
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`Y = imag(X)` extracts each imaginary component and preserves shape. Complex input returns its imaginary lane; real input returns zeros.",
        "Real double, single, and all eight integer classes produce zeros in the same numeric class. Paired complex integers project their exact imaginary storage. Logical and character input produces double zeros.",
        "String arrays are unsupported. Sparse input is currently densified.",
    ] },
    BuiltinDocumentationSection { heading: "GPU execution", paragraphs: &[
        "Supported providers extract a complex imaginary lane or materialize class-correct zeros on the owning device. Shape, class, owner, and residency are preserved.",
        "Typed unsupported operations gather through the exact owner and restore a representable result. Fusion may combine imaginary projection with adjacent elementwise work.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "complex-scalar",
        title: "Extract the imaginary part of a complex scalar",
        program: "z = 3 + 4i;\nb = imag(z)",
        display_output: Some("b = 4"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(b == 4);",
        },
    },
    BuiltinExample {
        id: "complex-matrix",
        title: "Extract imaginary components from a complex matrix",
        program: "Z = [1+2i 4-3i; -5+0i 7+8i];\nY = imag(Z)",
        display_output: Some("Y = [2 -3; 0 8]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(Y, [2 -3; 0 8]));",
        },
    },
    BuiltinExample {
        id: "real-zeros",
        title: "Return zeros for real input",
        program: "data = [-2.5 0 9.75];\nvalues = imag(data)",
        display_output: Some("values = [0 0 0]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(values, [0 0 0]));",
        },
    },
    BuiltinExample {
        id: "logical-zeros",
        title: "Return double zeros for a logical mask",
        program: "mask = logical([0 1 0; 1 0 1]);\nzerosOnly = imag(mask)",
        display_output: Some("zerosOnly = [0 0 0; 0 0 0]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(zerosOnly, \"double\"));\nassert(isequal(zerosOnly, zeros(2, 3)));",
        },
    },
    BuiltinExample {
        id: "gpu-array",
        title: "Extract imaginary values from provider-resident data",
        program: "G = gpuArray([1 -2; 3 -4]);\nres = imag(G);\nhost = gather(res)",
        display_output: Some("host = [0 0; 0 0]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(host, zeros(2, 2)));",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "What does `imag` return for real input?", answer: "It returns zeros with the same shape. Numeric input retains its class; logical and character input returns double zeros." },
    BuiltinDocumentationFaq { question: "How does `imag` handle complex zero?", answer: "`imag(0 + 0i)` returns exactly zero." },
    BuiltinDocumentationFaq { question: "Can I call `imag` on string arrays?", answer: "No. `imag` accepts numeric, logical, and character arrays." },
    BuiltinDocumentationFaq { question: "Does `imag` allocate a new array?", answer: "Projection or zero materialization may allocate; fusion can eliminate the intermediate when safe." },
    BuiltinDocumentationFaq { question: "What happens when a provider lacks imaginary projection?", answer: "RunMat gathers through the exact owner, applies host semantics, and restores the result when representable." },
    BuiltinDocumentationFaq { question: "Does GPU execution match CPU behavior?", answer: "Yes. Real values produce exact zeros and complex values return the same imaginary components in their native precision." },
    BuiltinDocumentationFaq { question: "Can `imag` participate in fusion?", answer: "Yes. Supported projection can be folded into adjacent elementwise kernels." },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("imag"),
    slug: Some("imag"),
    summary: "Extract imaginary components or produce class-correct zeros for real input.",
    description: "`imag(X)` returns each imaginary component. Real-valued input produces zeros with the same shape and documented class.",
    keywords: &["imag", "imaginary", "complex", "component", "elementwise", "gpu"],
    related: &["abs", "angle", "complex", "conj", "double", "gather", "gpuArray", "real", "sign", "single"],
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
