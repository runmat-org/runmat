use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Stable element-wise norms",
        paragraphs: &[
            "`hypot(A, B)` computes `sqrt(abs(A).^2 + abs(B).^2)` element by element with a scaled algorithm that avoids avoidable overflow and underflow. Results are nonnegative. A NaN operand produces NaN even when the other operand is infinite, matching MATLAB's documented behavior.",
            "Complex floating operands contribute their magnitudes. This is equivalent to combining the real Euclidean magnitudes without first forming potentially overflowing squares.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Classes and shapes",
        paragraphs: &[
            "Documented inputs are single or double arrays, including complex values. Inputs use MATLAB-style implicit expansion. Two single operands return single; any other documented combination returns double. The result is always real and has the broadcasted shape. Sparse input is not currently supported.",
            "RunMat mode also accepts all eight real fixed-width integer classes, logical arrays, and character arrays. Integer values must be exactly representable at the binary64 calculation boundary; logical values contribute zero or one and characters contribute their Unicode code points. These forms are rejected in MATLAB compatibility mode.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Accelerated, fused, and distributed execution",
        paragraphs: &[
            "Shape-matched real floating operands owned by one provider can use its direct `elem_hypot` operation. RunMat validates the returned shape, real storage, precision, owner, device, and non-aliasing before accepting it. Only a typed unsupported result enters host fallback; provider and contract failures remain visible.",
            "Broadcasting, complex values, mixed physical types, and admitted integer or logical storage gather through the exact owner, use the canonical host calculation, and restore the result when the owner supports its physical precision. Real floating expressions may also participate in element-wise fusion.",
            "Distributed inputs currently use the declared materialization path before the same operation is applied. This avoids claiming partition-local behavior that is not yet provided by the execution service.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "triangle", title: "Find a triangle hypotenuse", program: "C = hypot(3, 4)", display_output: Some("C = 5"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(C == 5);" } },
    BuiltinExample { id: "implicit-expansion", title: "Expand a scalar across a vector", program: "A = [-3 0 3];\nC = hypot(A, 4)", display_output: Some("C = [5 4 5]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(C, [5 4 5]));" } },
    BuiltinExample { id: "complex", title: "Combine complex magnitudes", program: "A = [1+2i 3-4i];\nB = [2-1i -1+1i];\nC = hypot(A, B)", display_output: Some("C = [3.1623 5.1962]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(max(abs(C - [sqrt(10) sqrt(27)])) < 1e-12);" } },
    BuiltinExample { id: "overflow", title: "Avoid intermediate overflow", program: "C = hypot(1e308, 1e308)", display_output: Some("C = 1.4142e+308"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isfinite(C));\nassert(abs(C/1e308 - sqrt(2)) < 1e-12);" } },
    BuiltinExample { id: "single", title: "Retain all-single precision", program: "C = hypot(single([3 5]), single([4 12]))", display_output: Some("C = single([5 13])"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(C, 'single'));\nassert(isequal(C, single([5 13])));" } },
    BuiltinExample { id: "character-extension", title: "Use a character code point in RunMat mode", program: "C = hypot('A', 0)", display_output: Some("C = 65"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(C == 65);" } },
    BuiltinExample { id: "gpu-residency", title: "Keep matching operands resident", program: "Ga = gpuArray([3 5; 8 7]);\nGb = gpuArray([4 12; 15 24]);\nGc = hypot(Ga, Gb);\nC = gather(Gc)", display_output: Some("C = [5 13; 17 25]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Wgpu, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(Gc, 'gpuArray'));\nassert(max(abs(C(:) - [5; 17; 13; 25])) < 1e-6);" } },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Why use hypot instead of writing the square-root formula?", answer: "The scaled implementation avoids intermediate overflow and underflow that can occur while squaring the operands." },
    BuiltinDocumentationFaq { question: "Does hypot accept complex input?", answer: "Yes. Each complex operand contributes its magnitude, and the result is real and nonnegative." },
    BuiltinDocumentationFaq { question: "What happens for NaN and Inf?", answer: "A NaN operand produces NaN, including a NaN paired with positive or negative infinity. Otherwise an infinite magnitude produces infinity." },
    BuiltinDocumentationFaq { question: "How does implicit expansion work?", answer: "Singleton dimensions expand to match the other operand. Incompatible non-singleton dimensions produce a size-mismatch error." },
    BuiltinDocumentationFaq { question: "What class does hypot return?", answer: "Two single operands return single. Other supported combinations return double; RunMat integer, logical, and character extensions also return double." },
    BuiltinDocumentationFaq { question: "Can the result remain provider-resident?", answer: "Yes. A valid direct result remains resident, while unsupported forms gather through the exact owner and restore when the result precision is supported." },
    BuiltinDocumentationFaq { question: "How can I combine three components?", answer: "Nest the operation, for example `hypot(x, hypot(y, z))`, to retain the stable pairwise calculation." },
];

const RELATED: &[&str] = &[
    "sqrt",
    "abs",
    "norm",
    "atan2",
    "sin",
    "gpuArray",
    "gather",
    "angle",
    "conj",
    "double",
    "exp",
    "expm1",
    "factorial",
    "gamma",
    "imag",
    "ldivide",
    "log",
    "log10",
    "log1p",
    "log2",
    "minus",
    "plus",
    "pow2",
    "power",
    "rdivide",
    "real",
    "sign",
    "single",
    "times",
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "sqrt", target: BuiltinDocumentationLinkTarget::Builtin("sqrt") },
    BuiltinDocumentationLink { label: "abs", target: BuiltinDocumentationLinkTarget::Builtin("abs") },
    BuiltinDocumentationLink { label: "norm", target: BuiltinDocumentationLinkTarget::Builtin("norm") },
    BuiltinDocumentationLink { label: "GPU arrays", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") },
    BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Builtin("gather") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/tree/main/crates/runmat-runtime/src/builtins/math/elementwise/hypot") },
];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Stable norm runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/tree/main/crates/runmat-runtime/src/builtins/math/elementwise/hypot") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Shapes, classes, complex magnitudes, extensions, and errors", location: "crates/runmat-runtime/src/builtins/math/elementwise/hypot/tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::ProviderTest, label: "Direct execution, fallback, ownership, and output validation", location: "crates/runmat-runtime/src/builtins/math/elementwise/hypot/tests/provider.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::WgpuTest, label: "Actual WGPU stable-norm parity", location: "crates/runmat-runtime/src/builtins/math/elementwise/hypot/tests/wgpu.rs::hypot_wgpu_matches_cpu_elementwise" },
    ],
    notes: &[],
};

pub(super) const HYPOT_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("hypot"), slug: Some("hypot"),
    summary: "Compute a stable element-wise Euclidean norm.",
    description: "`hypot` combines two real or complex floating magnitudes without avoidable intermediate overflow or underflow, applies implicit expansion, and preserves supported execution residency.",
    keywords: &["hypot", "euclidean norm", "distance", "overflow", "underflow", "complex", "gpu"],
    related: RELATED,
    sections: SECTIONS, examples: EXAMPLES, example_exemption: None, faqs: FAQS, links: LINKS,
    media: &[], evidence: EVIDENCE,
    introduced: Some("Before R2006a"), status: Some(BuiltinDocumentationStatus::Stable),
};
