use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior and supported arrays",
        paragraphs: &["`issymmetric(A)` returns true when matrix `A` is symmetric about its main diagonal. Optional skew-symmetric mode and tolerance behavior follow MATLAB semantics.", "Works with scalars, vectors (treated as matrices), and higher-rank arrays whose trailing dimensions are singleton (MATLAB-compatible matrix semantics). An error is raised when the input has more than two non-singleton dimensions.", "Non-square matrices immediately return `false`.", "Logical inputs are promoted to double precision (`true → 1.0`, `false → 0.0`) before testing.", "Complex inputs are compared without conjugation; use `ishermitian` for conjugate symmetry.", "Floating-point tolerance can be supplied to account for numerical noise.", "Pass `'skew'` to test skew-symmetry (`A == -A.'`). Diagonal entries must be zero (within the tolerance) when `skew` mode is active."],
    },
    BuiltinDocumentationSection {
        heading: "GPU residency and provider behavior",
        paragraphs: &["Manual `gpuArray` / `gather` calls are optional. When a provider implements the symmetry hook, RunMat executes the predicate in-place on the GPU and only transfers the scalar result. Otherwise the runtime gathers the tensor transparently and reuses the CPU path, preserving correctness with a minor residency cost.", "RunMat keeps GPU tensors resident whenever feasible. When the active acceleration provider exposes a symmetry predicate hook (`issymmetric`), the test runs entirely on the device and returns a host logical scalar. Providers without that hook gracefully fall back to downloading the matrix, so results remain correct even without GPU specialisation.", "Falls back to gathering GPU operands when providers lack a device-side symmetry predicate."],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "Checking whether a matrix is symmetric",
        program: "A = [2 1 1; 1 3 4; 1 4 5];\ntf = issymmetric(A)",
        display_output: Some("tf = logical\n   1"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(tf);",
        },
    },
    BuiltinExample {
        id: "example-2",
        title: "Allowing numerical noise with a tolerance",
        program: "A = [1 1+1e-12; 1-1e-12 1];\ntf = issymmetric(A, 1e-9)",
        display_output: Some("tf = logical\n   1"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(tf);",
        },
    },
    BuiltinExample {
        id: "example-3",
        title: "Detecting a skew-symmetric matrix",
        program: "B = [0 -2 4; 2 0 -3; -4 3 0];\ntf = issymmetric(B, 'skew')",
        display_output: Some("tf = logical\n   1"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(tf);",
        },
    },
    BuiltinExample {
        id: "example-4",
        title: "Handling non-square matrices",
        program: "C = [1 2 3; 4 5 6];\ntf = issymmetric(C)",
        display_output: Some("tf = logical\n   0"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(~tf);",
        },
    },
    BuiltinExample {
        id: "example-5",
        title: "Working with complex-valued matrices",
        program: "Z = [1+2i 3-4i; 3-4i 5+6i];\ntf = issymmetric(Z)",
        display_output: Some("tf = logical\n   1"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(tf);",
        },
    },
    BuiltinExample {
        id: "example-6",
        title: "Inspecting a GPU-resident matrix",
        program: "G = gpuArray([0 5; 5 9]);\ntf = issymmetric(G)",
        display_output: Some("tf = logical\n   1"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(tf);",
        },
    },
];
const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Does `issymmetric` accept tolerance arguments?", answer: "Yes. Pass a non-negative scalar tolerance as the second or third argument. The comparison uses an absolute tolerance on the element-wise difference (or sum for skew tests)." },
    BuiltinDocumentationFaq { question: "What strings are accepted for the skew flag?", answer: "Use `'skew'` to test skew-symmetry and `'nonskew'` (or omit the flag) for the default symmetry test." },
    BuiltinDocumentationFaq { question: "How are diagonal elements handled in skew mode?", answer: "Diagonal elements must be zero (within the tolerance) because a skew-symmetric matrix satisfies `A(i,i) = -A(i,i)`." },
    BuiltinDocumentationFaq { question: "Are NaN values considered symmetric?", answer: "No. Any NaN encountered off or on the diagonal causes the test to return `false`, matching MATLAB behaviour." },
    BuiltinDocumentationFaq { question: "Do logical matrices work?", answer: "Yes. Logical inputs are promoted to double precision and checked using the same rules as numeric matrices." },
    BuiltinDocumentationFaq { question: "Does `issymmetric` conjugate complex inputs?", answer: "No. The builtin compares complex entries without conjugation. Use `ishermitian` if you need conjugate symmetry." },
    BuiltinDocumentationFaq { question: "What happens with higher-dimensional arrays?", answer: "`issymmetric` raises an error when the input has more than two non-singleton dimensions. Reshape the data to a 2-D matrix before calling it." },
    BuiltinDocumentationFaq { question: "Can I combine the skew flag and tolerance?", answer: "Yes. You can call `issymmetric(A, 'skew', tol)` or `issymmetric(A, tol, 'skew')`. The order of the optional arguments does not matter." },
    BuiltinDocumentationFaq { question: "Is an empty matrix symmetric?", answer: "Yes. Empty square matrices (`0x0`) return logical `true`, while non-square empty matrices return `false`." },
    BuiltinDocumentationFaq { question: "Does the result depend on GPU availability?", answer: "No. You receive the same logical answer regardless of whether a GPU provider is registered. Only the execution strategy changes (device-side predicate vs. host fallback)." },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "bandwidth", target: BuiltinDocumentationLinkTarget::Builtin("bandwidth") },
    BuiltinDocumentationLink { label: "chol", target: BuiltinDocumentationLinkTarget::Builtin("chol") },
    BuiltinDocumentationLink { label: "eig", target: BuiltinDocumentationLinkTarget::Builtin("eig") },
    BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuarray") },
    BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Builtin("gather") },
    BuiltinDocumentationLink { label: "ishermitian", target: BuiltinDocumentationLinkTarget::Builtin("ishermitian") },
    BuiltinDocumentationLink { label: "symrcm", target: BuiltinDocumentationLinkTarget::Builtin("symrcm") },
    BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/structure/issymmetric.rs") },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("issymmetric"),
    slug: Some("issymmetric"),
    summary: "Test matrix symmetry in MATLAB and RunMat.",
    description: "`issymmetric(A)` returns true when matrix `A` is symmetric about its main diagonal. Optional skew-symmetric mode and tolerance behavior follow MATLAB semantics.",
    keywords: &["issymmetric", "symmetric matrix", "skew-symmetric", "matrix structure", "gpu"],
    related: &["bandwidth", "chol", "eig", "gpuarray", "gather", "ishermitian", "symrcm"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/structure/issymmetric.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Existing runtime unit tests", location: "builtins::math::linalg::structure::issymmetric::tests" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Existing runtime integration tests", location: "builtins::math::linalg::structure::issymmetric::tests::issymmetric_gpu_roundtrip" }],
        notes: &["Source-only catalog migration. Compiled conservation and executable example verification remain pending."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
