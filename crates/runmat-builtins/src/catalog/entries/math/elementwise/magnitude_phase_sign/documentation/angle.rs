use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/angle.rs") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "CPU and representation tests", location: "builtins::math::elementwise::angle::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::ProviderTest, label: "Provider round-trip test", location: "builtins::math::elementwise::angle::tests::angle_gpu_provider_roundtrip" },
    ],
    notes: &[],
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`theta = angle(X)` computes `atan2(imag(X), real(X))` elementwise in radians, producing values in `[-pi, pi]` with the input shape and floating-point class.",
        "Positive real values map to zero, negative real values to pi, and real or complex zero to zero. NaN propagates under IEEE rules.",
        "Only real or complex single and double input is supported. Every fixed-width integer class, including typed complex integers, rejects before floating-point materialization; logical, character, and string input is also rejected.",
    ] },
    BuiltinDocumentationSection { heading: "GPU execution", paragraphs: &[
        "Providers with unary angle support compute real or complex-interleaved phase directly on the owning device. Real storage uses `atan2(0, X)` and complex storage uses both lanes.",
        "Typed unsupported operations gather through the exact owner and apply the same host semantics. The output retains single or double precision, and fusion may eliminate safe intermediates.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "complex-scalar", title: "Compute the phase of a complex scalar", program: "z = 3 + 4i;\ntheta = angle(z)", display_output: Some("theta = 0.9273"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(abs(theta - atan2(4, 3)) < 1e-12);" } },
    BuiltinExample { id: "quadrants", title: "Extract phases from all four quadrants", program: "Z = [1+1i -1+1i -1-1i 1-1i];\nphases = angle(Z)", display_output: Some("phases = [0.7854 2.3562 -2.3562 -0.7854]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "expected = [pi/4 3*pi/4 -3*pi/4 -pi/4];\nassert(max(abs(phases - expected)) < 1e-12);" } },
    BuiltinExample { id: "real-values", title: "Compute phases of real values", program: "vals = [-2 -1 0 1 2];\nphi = angle(vals)", display_output: Some("phi = [3.1416 3.1416 0 0 0]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(max(abs(phi - [pi pi 0 0 0])) < 1e-12);" } },
    BuiltinExample { id: "gpu-array", title: "Compute phase on provider-resident data", program: "G = gpuArray([1 -1; -1 1]);\ntheta = gather(angle(G))", display_output: Some("theta = [0.0000 3.1416; 3.1416 0.0000]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Wgpu, verification: BuiltinExampleVerification::Assertions { source: "expected = [0 pi; pi 0];\nassert(max(abs(theta(:) - expected(:))) < 1e-12);" } },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Does `angle` return radians?", answer: "Yes. Results are in radians within `[-pi, pi]`." },
    BuiltinDocumentationFaq { question: "What happens at zero?", answer: "Real zero and complex zero both return zero." },
    BuiltinDocumentationFaq { question: "How does `angle` handle NaN?", answer: "NaN propagates under IEEE arithmetic." },
    BuiltinDocumentationFaq { question: "Can `angle` accept integers, logicals, characters, or strings?", answer: "No. Convert explicitly to single or double only when that conversion expresses the intended calculation." },
    BuiltinDocumentationFaq { question: "Does `angle` allocate?", answer: "It produces a dense floating array; fusion can eliminate an intermediate when safe." },
    BuiltinDocumentationFaq { question: "Can complex GPU values remain resident?", answer: "Yes when the owning provider supports complex-interleaved unary angle execution; otherwise the typed owner-aware fallback applies." },
    BuiltinDocumentationFaq { question: "Will GPU and CPU results match?", answer: "Double providers use double precision. Single providers may have small IEEE rounding differences." },
];

pub(in super::super) const ANGLE_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("angle"), slug: Some("angle"),
    summary: "Compute phase angles of real and complex floating-point values.",
    description: "`angle(X)` returns the elementwise phase in radians for real or complex single- or double-precision input.",
    keywords: &["angle", "phase", "argument", "atan2", "complex", "gpu"],
    related: &["abs", "complex", "conj", "double", "gather", "gpuArray", "imag", "real", "sign", "single"],
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None, faqs: FAQS,
    links: &[], media: &[], evidence: EVIDENCE, introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
