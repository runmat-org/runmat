use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Powers and binary scaling",
        paragraphs: &[
            "`pow2(E)` computes `2.^E` element by element. `pow2(F, E)` computes `F .* 2.^E` and applies implicit expansion to compatible shapes.",
            "Real and complex floating inputs are supported. Complex exponents use `exp(E * log(2))`; a complex significand is multiplied by the resulting power of two.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Classes and compatibility",
        paragraphs: &[
            "Single input retains single precision in the unary form. The two-input form uses single precision when either materialized operand is single; otherwise it uses double. Logical and character inputs enter the double domain, with characters interpreted by Unicode code point.",
            "RunMat mode accepts real fixed-width integers when every value is exactly representable at the floating calculation boundary. Unary integer input returns double. Integer significands and binary exponents are checked independently. MATLAB compatibility mode rejects those extension forms.",
            "Typed complex-integer values and sparse inputs are not supported. Large positive exponents may overflow to infinity, and large negative exponents may underflow to zero according to the active floating precision.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Accelerated and fused execution",
        paragraphs: &[
            "A compatible provider can evaluate the unary form directly. Shape-matched binary floating operands on the same provider and device can use its binary-scaling operation. RunMat validates shape, storage, precision, ownership, device, and aliasing before accepting either result.",
            "Typed integer and logical values, implicit expansion, mixed residency, and unsupported provider operations use the canonical host calculation. Unary fallback can restore the result to the exact input owner; binary fallback currently returns a host value. Floating unary expressions can participate in element-wise fusion.",
            "Distributed arguments use the declared materialization path before evaluation; the catalog does not claim a partition-local binary operation.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "scalar", title: "Compute a scalar power of two", program: "Y = pow2(3)", display_output: Some("Y = 8"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(Y == 8);" } },
    BuiltinExample { id: "vector", title: "Transform a vector of exponents", program: "E = [-1 0 1 2];\nY = pow2(E)", display_output: Some("Y = [0.5 1 2 4]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(Y, [0.5 1 2 4]));" } },
    BuiltinExample { id: "scaling", title: "Scale significands by binary exponents", program: "F = [0.75 1.5];\nE = [4 5];\nY = pow2(F, E)", display_output: Some("Y = [12 48]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(Y, [12 48]));" } },
    BuiltinExample { id: "complex", title: "Evaluate a complex exponent", program: "Y = pow2(1 + 2i)", display_output: Some("Y = -0.3667 + 0.8894i"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "expected = exp((1 + 2i) * log(2));\nassert(abs(Y - expected) < 1e-12);" } },
    BuiltinExample { id: "character", title: "Use character code points", program: "Y = pow2('AB')", display_output: Some("Y = [2^65 2^66]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(Y, [2^65 2^66]));" } },
    BuiltinExample { id: "integer-extension", title: "Use exact integer exponents in RunMat mode", program: "E = int16([-1 0 3]);\nY = pow2(E)", display_output: Some("Y = [0.5 1 8]"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(Y, 'double'));\nassert(isequal(Y, [0.5 1 8]));" } },
    BuiltinExample { id: "gpu", title: "Keep a unary result resident", program: "G = gpuArray(single([-1 0 3]));\nGy = pow2(G);\nY = gather(Gy)", display_output: Some("Y = single([0.5 1 8])"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Wgpu, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(Gy, 'gpuArray'), 'pow2 must preserve explicit gpuArray identity');\nassert(isa(Y, 'single'), 'gathered pow2 output must retain single precision');\nassert(all(abs(Y - single([0.5 1 8])) < single(1e-5)), 'gathered pow2 values must match within single precision');" } },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "How do the two forms differ?", answer: "`pow2(E)` returns `2.^E`. `pow2(F, E)` multiplies each significand in `F` by the corresponding power of two." },
    BuiltinDocumentationFaq { question: "Can the two-input form expand scalars or singleton dimensions?", answer: "Yes. Compatible singleton dimensions expand; incompatible dimensions produce a size-mismatch error." },
    BuiltinDocumentationFaq { question: "What happens for very large exponents?", answer: "The active IEEE floating precision determines overflow and underflow. Results can become infinity or zero." },
    BuiltinDocumentationFaq { question: "Can fixed-width integers be used?", answer: "RunMat mode accepts real fixed-width integers after exact floating-boundary validation. MATLAB compatibility mode rejects these extension forms." },
    BuiltinDocumentationFaq { question: "When does a result remain provider-resident?", answer: "Validated direct unary and shape-matched binary provider results remain resident. Unary host fallback can restore the result; binary fallback currently returns host data." },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "nextpow2", target: BuiltinDocumentationLinkTarget::Builtin("nextpow2") },
    BuiltinDocumentationLink { label: "power", target: BuiltinDocumentationLinkTarget::Builtin("power") },
    BuiltinDocumentationLink { label: "exp", target: BuiltinDocumentationLinkTarget::Builtin("exp") },
    BuiltinDocumentationLink { label: "log2", target: BuiltinDocumentationLinkTarget::Builtin("log2") },
    BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") },
    BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Builtin("gather") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/powers_of_two/pow2") },
];

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Power and binary-scaling runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/elementwise/powers_of_two/pow2") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Unary, binary, complex, character, class, shape, and error behavior", location: "crates/runmat-runtime/src/builtins/math/elementwise/powers_of_two/pow2/tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::ConformanceTest, label: "Compatibility-gated exact integer boundaries", location: "crates/runmat-runtime/src/builtins/math/elementwise/powers_of_two/pow2/tests/integer.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::ProviderTest, label: "Direct operations, fallback, ownership, and output validation", location: "crates/runmat-runtime/src/builtins/math/elementwise/powers_of_two/pow2/tests/provider.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::WgpuTest, label: "Actual WGPU unary and binary parity", location: "crates/runmat-runtime/src/builtins/math/elementwise/powers_of_two/pow2/tests/wgpu.rs" },
    ],
    notes: &[],
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("pow2"),
    slug: Some("pow2"),
    summary: "Compute powers of two or scale significands by binary exponents.",
    description: "`pow2` evaluates `2.^E` or `F .* 2.^E` with explicit class, compatibility, broadcasting, and provider behavior.",
    keywords: &["pow2", "power of two", "binary scaling", "ldexp", "gpu"],
    related: &["nextpow2", "power", "exp", "log2", "gpuArray", "gather"],
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
