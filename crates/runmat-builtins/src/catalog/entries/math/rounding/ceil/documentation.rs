use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "`Y = ceil(X)` rounds each element toward positive infinity. It returns the smallest integer-valued result greater than or equal to the input and preserves scalar, vector, matrix, empty, singleton, and N-D shapes.",
            "Double and single inputs retain their class. All eight real fixed-width integer classes are exact identity operations and retain class, shape, and bits. Logical and character arrays produce double values; character elements use their Unicode code points. Complex floating values are rounded component by component.",
            "`NaN`, positive infinity, and negative infinity propagate unchanged. Tables and timetables are reconstructed with `ceil` applied independently to every supported variable while retaining the tabular class and metadata.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "GPU execution and fusion",
        paragraphs: &[
            "Resident real floating input uses the owning provider's `unary_ceil` operation when available. Provider output must be distinct, preserve shape, precision, storage, owner, and device, and contain ordinary floating data before RunMat accepts it.",
            "A typed unsupported response gathers once, computes on the host at the input precision, and restores the result to the same owner and device. Other provider errors remain visible. Logical input follows the conversion-to-double fallback. Resident integer input is returned as the same exact handle because every value is already integral.",
            "`ceil` is eligible for elementwise fusion, allowing compatible neighboring operations to remain resident. `gpuArray` and `gather` remain available for explicit placement control.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "vector",
        title: "Round a vector toward positive infinity",
        program: "x = [-2.7, -0.3, 0, 0.8, 3.2];\ny = ceil(x)",
        display_output: Some("y = [-2 0 0 1 4]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(y, [-2 0 0 1 4]));",
        },
    },
    BuiltinExample {
        id: "matrix",
        title: "Round every matrix element upward",
        program: "A = [1.2 4.7; -3.4 5.0];\nB = ceil(A)",
        display_output: Some("B = [2 5; -3 5]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(B, [2 5; -3 5]));",
        },
    },
    BuiltinExample {
        id: "tensor",
        title: "Preserve an N-D input's shape",
        program: "t = reshape([-1.8, -0.2, 0.4, 1.1, 2.1, 3.6], [3, 2]);\nup = ceil(t)",
        display_output: Some("up = [-1 2; 0 3; 1 4]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(up), [3 2])); assert(isequal(up, [-1 2; 0 3; 1 4]));",
        },
    },
    BuiltinExample {
        id: "complex",
        title: "Round real and imaginary components independently",
        program: "z = [1.2 + 2.1i, -0.2 - 3.9i];\nresult = ceil(z)",
        display_output: Some("result = [2 + 3i, 0 - 3i]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(result, [2 + 3i, 0 - 3i]));",
        },
    },
    BuiltinExample {
        id: "typed-integer",
        title: "Retain exact typed integers",
        program: "x = uint64([0x0000000000000000u64, 0x0020000000000001u64, 0xFFFFFFFFFFFFFFFFu64]);\ny = ceil(x)",
        display_output: Some("y retains class uint64 and every input bit"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(y, 'uint64')); assert(isequal(y, x));",
        },
    },
    BuiltinExample {
        id: "gpu-residency",
        title: "Round a resident matrix and gather the result",
        program: "A = gpuArray([1.8 -0.2; 2.7 3.4]);\nG = ceil(A);\nresult = gather(G)",
        display_output: Some("result = [2 0; 3 4]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(result, [2 0; 3 4]));",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Which direction does `ceil` use?", answer: "It rounds toward positive infinity. Positive fractions move away from zero, while negative fractions move toward zero; for example, `ceil(-0.1)` is `0`." },
    BuiltinDocumentationFaq { question: "How does `ceil` handle complex values?", answer: "It rounds the real and imaginary components independently." },
    BuiltinDocumentationFaq { question: "What happens to logical arrays?", answer: "Logical values become double values `0` and `1`; rounding does not change those values." },
    BuiltinDocumentationFaq { question: "Can `ceil` accept character arrays?", answer: "Yes. Each character becomes its Unicode code point in a double array of the same shape." },
    BuiltinDocumentationFaq { question: "What happens to `NaN` and infinities?", answer: "They propagate unchanged." },
    BuiltinDocumentationFaq { question: "Does provider execution change floating results?", answer: "No. Native output is validated. A provider that reports the operation as unsupported uses the same-precision host implementation and restores the result to the owner." },
    BuiltinDocumentationFaq { question: "Can fusion keep `ceil` on the GPU?", answer: "Yes. Compatible elementwise graphs can fuse `ceil` with neighboring operations." },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "floor", target: BuiltinDocumentationLinkTarget::Builtin("floor") },
    BuiltinDocumentationLink { label: "fix", target: BuiltinDocumentationLinkTarget::Builtin("fix") },
    BuiltinDocumentationLink { label: "round", target: BuiltinDocumentationLinkTarget::Builtin("round") },
    BuiltinDocumentationLink { label: "mod", target: BuiltinDocumentationLinkTarget::Builtin("mod") },
    BuiltinDocumentationLink { label: "rem", target: BuiltinDocumentationLinkTarget::Builtin("rem") },
    BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") },
    BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Builtin("gather") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/rounding/ceil.rs") },
];

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Ceiling runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/rounding/ceil.rs") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Real, complex, typed, logical, character, tabular, shape, and error behavior", location: "crates/runmat-runtime/src/builtins/math/rounding/ceil.rs::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::ProviderTest, label: "Owner-preserving provider execution and fallback", location: "crates/runmat-runtime/src/builtins/math/rounding/ceil.rs::tests::ceil_gpu_provider_roundtrip" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::WgpuTest, label: "Actual WGPU parity", location: "crates/runmat-runtime/src/builtins/math/rounding/ceil.rs::tests::ceil_wgpu_matches_cpu_elementwise" },
    ],
    notes: &[],
};

pub(crate) const CEIL_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("ceil"),
    slug: Some("ceil"),
    summary: "Round values toward positive infinity.",
    description: "`ceil` rounds numeric, logical, character, and supported tabular values element by element while preserving shape and supported residency.",
    keywords: &["ceil", "rounding", "positive infinity", "integers", "complex", "gpu"],
    related: &["floor", "fix", "round", "mod", "rem", "gpuArray", "gather"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: EVIDENCE,
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
