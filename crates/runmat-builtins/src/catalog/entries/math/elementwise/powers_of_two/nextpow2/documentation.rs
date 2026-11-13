use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Exponent values",
        paragraphs: &[
            "`nextpow2(X)` returns the smallest exponent `p` for which `2^p` is greater than or equal to `abs(X)`. It applies the calculation element by element and preserves the input shape.",
            "Zero maps to zero. Negative values use their magnitude. Floating-point infinity remains infinite and NaN remains NaN.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Classes and exact integers",
        paragraphs: &[
            "Double and single inputs retain their class. Logical inputs produce double output. Each fixed-width integer class produces the same integer class; RunMat reads native integer storage directly, including signed minima and `uint64` values above `flintmax`.",
            "Complex, character, string, sparse, cell, object, and other nonnumeric representations are rejected with a structured input error.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Accelerated execution",
        paragraphs: &[
            "A compatible floating-point provider can evaluate `nextpow2` through its unary operation. RunMat validates output shape, storage, precision, owner, device, and aliasing before accepting the result.",
            "Unsupported provider operations and typed integer inputs use exact-owner gather, the same host calculation, and validated restoration to the original provider. Provider failures or malformed outputs remain errors.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "scalar", title: "Find the exponent above a scalar", program: "p = nextpow2(9)", display_output: Some("p = 4"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(p == 4);" } },
    BuiltinExample { id: "array", title: "Transform an array element by element", program: "p = nextpow2([0 1 3 9])", display_output: Some("p = [0 0 2 4]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(p, [0 0 2 4]));" } },
    BuiltinExample { id: "fft-length", title: "Choose a power-of-two transform length", program: "x = 1:1000;\nN = 2^nextpow2(length(x));", display_output: Some("N = 1024"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(N == 1024);" } },
    BuiltinExample { id: "typed-integer", title: "Preserve a fixed-width integer class", program: "X = uint16([0 1 3 65535]);\np = nextpow2(X)", display_output: Some("p is uint16([0 0 2 16])"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(p, 'uint16'));\nassert(isequal(p, uint16([0 0 2 16])));" } },
    BuiltinExample { id: "gpu", title: "Keep a provider result resident", program: "G = gpuArray(single([0 1 3 9]));\nGp = nextpow2(G);\np = gather(Gp)", display_output: Some("p = single([0 0 2 4])"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Wgpu, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(Gp, 'gpuArray'));\nassert(isa(p, 'single'));\nassert(isequal(p, single([0 0 2 4])));" } },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Why does nextpow2 return an exponent?", answer: "The result is the exponent itself. Use `2^nextpow2(x)` for a scalar power of two or `2.^nextpow2(X)` for element-wise array results." },
    BuiltinDocumentationFaq { question: "Does nextpow2 preserve array shape?", answer: "Yes. Scalar, vector, matrix, N-D, and empty-array shapes are preserved." },
    BuiltinDocumentationFaq { question: "How are fixed-width integers handled?", answer: "RunMat computes magnitudes and exponents from native integer storage and returns the same fixed-width class." },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "pow2", target: BuiltinDocumentationLinkTarget::Builtin("pow2") },
    BuiltinDocumentationLink { label: "fft", target: BuiltinDocumentationLinkTarget::Builtin("fft") },
    BuiltinDocumentationLink { label: "length", target: BuiltinDocumentationLinkTarget::Builtin("length") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/powers_of_two/nextpow2") },
];

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Power-of-two exponent runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/powers_of_two/nextpow2") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Floating, logical, special-value, shape, and rejected-input semantics", location: "crates/runmat-runtime/src/builtins/math/elementwise/powers_of_two/nextpow2/tests.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::ConformanceTest, label: "All fixed-width classes, signed minima, and wide unsigned values", location: "crates/runmat-runtime/src/builtins/math/elementwise/powers_of_two/nextpow2/tests.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::ProviderTest, label: "Direct execution, fallback restoration, ownership, and output validation", location: "crates/runmat-runtime/src/builtins/math/elementwise/powers_of_two/nextpow2/tests.rs" },
    ],
    notes: &[],
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("nextpow2"),
    slug: Some("nextpow2"),
    summary: "Return the next-power-of-two exponent for each input value.",
    description: "`nextpow2` evaluates the magnitude of each real numeric or logical value, preserves its shape, and keeps supported numeric classes and provider residency explicit.",
    keywords: &["nextpow2", "power of two", "exponent", "FFT", "zero padding", "GPU"],
    related: &["pow2", "fft", "length", "gpuArray", "gather"],
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
