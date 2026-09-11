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
            "`Y = fix(X)` removes each fractional part by rounding toward zero. Positive values behave like `floor`; negative values behave like `ceil`. Scalar, vector, matrix, empty, singleton, and N-D shapes are preserved.",
            "Double and single inputs retain their class. All eight real fixed-width integer classes are exact identity operations and retain class, shape, and bits. Logical and character arrays produce double values; character elements use their Unicode code points. Complex floating values are rounded component by component.",
            "`NaN`, positive infinity, and negative infinity propagate unchanged. A zero result is normalized to positive zero. Tables and timetables are reconstructed with `fix` applied independently to every supported variable while retaining the tabular class and metadata.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "GPU execution and fusion",
        paragraphs: &[
            "Resident real floating input uses the owning provider's `unary_fix` operation when available. Provider output must be distinct, preserve shape, precision, storage, owner, and device, and contain ordinary floating data before RunMat accepts it.",
            "A typed unsupported response gathers once, computes on the host at the input precision, and restores the result to the same owner and device. Other provider errors remain visible. Logical input follows the conversion-to-double fallback. Resident integer input is returned as the same exact handle.",
            "`fix` is eligible for elementwise fusion, allowing compatible neighboring operations to remain resident. `gpuArray` and `gather` remain available for explicit placement control.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "vector",
        title: "Remove positive and negative fractional parts",
        program: "values = [-3.7 -2.4 -0.6 0 0.6 2.4 3.7];\ntruncated = fix(values)",
        display_output: Some("truncated = [-3 -2 0 0 0 2 3]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(truncated, [-3 -2 0 0 0 2 3]));" },
    },
    BuiltinExample {
        id: "matrix",
        title: "Remove fractional parts from a matrix",
        program: "A = [1.9 4.1; -2.8 0.5];\nB = fix(A)",
        display_output: Some("B = [1 4; -2 0]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(B, [1 4; -2 0]));" },
    },
    BuiltinExample {
        id: "complex",
        title: "Truncate real and imaginary components independently",
        program: "z = [1.9 + 2.6i, -3.4 - 0.2i];\nfixed = fix(z)",
        display_output: Some("fixed = [1 + 2i, -3 + 0i]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(fixed, [1 + 2i, -3 + 0i]));" },
    },
    BuiltinExample {
        id: "character-codes",
        title: "Convert character code points to doubles",
        program: "letters = ['A' 'B' 'C'];\ncodes = fix(letters)",
        display_output: Some("codes = [65 66 67]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isa(codes, 'double')); assert(isequal(codes, [65 66 67]));" },
    },
    BuiltinExample {
        id: "typed-integer",
        title: "Retain exact typed integers",
        program: "x = int64([-7, 0, 7]);\ny = fix(x)",
        display_output: Some("y retains class int64 and every input bit"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isa(y, 'int64')); assert(isequal(y, x));" },
    },
    BuiltinExample {
        id: "gpu-residency",
        title: "Truncate a resident vector and gather the result",
        program: "G = gpuArray(linspace(-2.4, 2.4, 6));\ntruncGpu = fix(G);\nhostValues = gather(truncGpu)",
        display_output: Some("hostValues = [-2 -1 0 0 1 2]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(hostValues, [-2 -1 0 0 1 2]));" },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "How is `fix` different from `floor` and `ceil`?", answer: "`fix` always rounds toward zero. Positive values behave like `floor`, and negative values behave like `ceil`." },
    BuiltinDocumentationFaq { question: "What happens to existing integers?", answer: "Every real typed integer retains its exact class, shape, and value. Logical input is different: it produces double values `0` and `1`." },
    BuiltinDocumentationFaq { question: "Does `fix` change `NaN` or infinities?", answer: "No. `NaN`, positive infinity, and negative infinity propagate unchanged." },
    BuiltinDocumentationFaq { question: "How does `fix` handle complex values?", answer: "It rounds the real and imaginary components independently toward zero." },
    BuiltinDocumentationFaq { question: "Will `fix` stay on the GPU?", answer: "Supported floating input uses the owner directly. Typed unsupported fallback restores the result to that owner, and resident integers retain their existing handle." },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "floor", target: BuiltinDocumentationLinkTarget::Builtin("floor") },
    BuiltinDocumentationLink { label: "ceil", target: BuiltinDocumentationLinkTarget::Builtin("ceil") },
    BuiltinDocumentationLink { label: "round", target: BuiltinDocumentationLinkTarget::Builtin("round") },
    BuiltinDocumentationLink { label: "mod", target: BuiltinDocumentationLinkTarget::Builtin("mod") },
    BuiltinDocumentationLink { label: "rem", target: BuiltinDocumentationLinkTarget::Builtin("rem") },
    BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") },
    BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Builtin("gather") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/rounding/fix.rs") },
];

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Toward-zero runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/rounding/fix.rs") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Real, complex, typed, logical, character, tabular, shape, and error behavior", location: "crates/runmat-runtime/src/builtins/math/rounding/fix.rs::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::ProviderTest, label: "Owner-preserving provider execution and fallback", location: "crates/runmat-runtime/src/builtins/math/rounding/fix.rs::tests::fix_gpu_provider_roundtrip" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::WgpuTest, label: "Actual WGPU parity", location: "crates/runmat-runtime/src/builtins/math/rounding/fix.rs::tests::fix_wgpu_matches_cpu_elementwise" },
    ],
    notes: &[],
};

pub(crate) const FIX_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("fix"),
    slug: Some("fix"),
    summary: "Round values toward zero.",
    description: "`fix` removes fractional parts from numeric, logical, character, and supported tabular values while preserving shape and supported residency.",
    keywords: &["fix", "truncate", "rounding", "toward zero", "integers", "complex", "gpu"],
    related: &["floor", "ceil", "round", "mod", "rem", "gpuArray", "gather"],
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
