use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Construction and shapes",
        paragraphs: &[
            "`Z = complex(A, B)` constructs `A + 1i*B` from real numeric components. Two non-scalar components must have the same size; a scalar component expands across the other input. This constructor does not apply general implicit expansion.",
            "`Z = complex(A)` gives real input an explicit zero imaginary component. Existing complex scalars and arrays pass through unchanged. Empty arrays retain their shape.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Classes",
        paragraphs: &[
            "Two floating-point components produce complex single when either component is single and complex double otherwise. Logical components convert to complex double. Strings and character arrays are not numeric constructor inputs.",
            "All eight fixed-width integer classes use exact complex-integer storage. When either binary component is integer, the other must have the same integer class or be a full scalar double. A scalar double is rounded and saturated according to the integer class.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Accelerated execution",
        paragraphs: &[
            "Compatible real floating-point gpuArray inputs use the owning provider's complex-construction operation. RunMat validates the returned shape, storage, precision, owner, device, class metadata, and non-aliasing before accepting it.",
            "A typed unsupported result enters owner-aware host fallback. Other provider errors remain visible. Typed-integer values transfer without a floating-point intermediary and return to the selected owner when exact restoration is available.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "scalar",
        title: "Construct a complex scalar",
        program: "z = complex(3, 4)",
        display_output: Some("z = 3+4i"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(z == 3 + 4i);",
        },
    },
    BuiltinExample {
        id: "matching-arrays",
        title: "Combine matching arrays",
        program: "re = [1 2 3];\nim = [4 5 6];\nz = complex(re, im)",
        display_output: Some("z = [1+4i 2+5i 3+6i]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(z, [1+4i 2+5i 3+6i]));",
        },
    },
    BuiltinExample {
        id: "scalar-expansion",
        title: "Expand a scalar component",
        program: "re = [1 2 3];\nz = complex(re, -1)",
        display_output: Some("z = [1-1i 2-1i 3-1i]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(z, [1-1i 2-1i 3-1i]));",
        },
    },
    BuiltinExample {
        id: "explicit-complex-storage",
        title: "Add an explicit zero imaginary component",
        program: "z = complex(12);\ntf = isreal(z)",
        display_output: Some("z = 12+0i\n\ntf = 0"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(z == 12);\nassert(~tf);",
        },
    },
    BuiltinExample {
        id: "identity",
        title: "Preserve an existing complex value",
        program: "z = complex(1 + 2i)",
        display_output: Some("z = 1+2i"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(z == 1 + 2i);",
        },
    },
    BuiltinExample {
        id: "gpu-residency",
        title: "Construct a resident complex array",
        program:
            "re = gpuArray([1 2 3]);\nim = gpuArray([4 5 6]);\nG = complex(re, im);\nz = gather(G)",
        display_output: Some("z = [1+4i 2+5i 3+6i]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(G, 'gpuArray'));\nassert(isequal(z, [1+4i 2+5i 3+6i]));",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Does complex use general implicit expansion?", answer: "No. A scalar component can expand, but two non-scalar components must have identical sizes." },
    BuiltinDocumentationFaq { question: "Can binary complex accept complex components?", answer: "No. Both components in `complex(A, B)` must be real numeric values." },
    BuiltinDocumentationFaq { question: "What does unary complex do?", answer: "It adds explicit zero imaginary storage to real input and preserves input that is already complex." },
    BuiltinDocumentationFaq { question: "Why does isreal(complex(5)) return false?", answer: "The value has complex storage with an explicit zero imaginary component, and `isreal` reports the storage domain." },
    BuiltinDocumentationFaq { question: "Does complex accept logical values?", answer: "Yes. Logical input produces complex double output." },
    BuiltinDocumentationFaq { question: "How are integer components represented?", answer: "RunMat retains their fixed-width class in exact paired real and imaginary storage." },
    BuiltinDocumentationFaq { question: "Can the result remain on a provider?", answer: "Yes. Supported floating-point construction stays resident; exact fallback restores typed results to the selected owner when possible." },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "real",
        target: BuiltinDocumentationLinkTarget::Builtin("real"),
    },
    BuiltinDocumentationLink {
        label: "imag",
        target: BuiltinDocumentationLinkTarget::Builtin("imag"),
    },
    BuiltinDocumentationLink {
        label: "isreal",
        target: BuiltinDocumentationLinkTarget::Builtin("isreal"),
    },
    BuiltinDocumentationLink {
        label: "conj",
        target: BuiltinDocumentationLinkTarget::Builtin("conj"),
    },
];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/complex_construction/complex") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Host classes, shapes, identity, and diagnostics", location: "builtins::math::elementwise::complex_construction::complex::tests::host" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::ProviderTest, label: "Provider ownership, contracts, and fallback", location: "builtins::math::elementwise::complex_construction::complex::tests::provider" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::WgpuTest, label: "Actual WGPU construction parity", location: "builtins::math::elementwise::complex_construction::complex::tests::wgpu" },
    ],
    notes: &[],
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("complex"),
    slug: Some("complex"),
    summary: "Construct complex values from real and imaginary components.",
    description: "`complex` constructs class-aware complex scalars and arrays, applies scalar component expansion, and preserves supported provider residency.",
    keywords: &["complex", "constructor", "real", "imaginary", "integer", "gpu"],
    related: &["abs", "angle", "conj", "double", "gather", "gpuArray", "imag", "isreal", "real", "single"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: EVIDENCE,
    introduced: Some("Before R2006a"),
    status: Some(BuiltinDocumentationStatus::Stable),
};
