use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior and supported arrays",
        paragraphs: &["`ishermitian(A)` returns true when `A` equals its conjugate transpose. Optional skew-Hermitian mode and tolerance behavior follow MATLAB semantics.", "`ishermitian(A)` tests for Hermitian structure (`A == conj(A').`).", "Pass `'skew'` (or `'skewhermitian'`) to test for skew-Hermitian structure (`A == -conj(A')`).", "Non-square matrices immediately return `false`.", "Diagonal entries must be real for Hermitian matrices and purely imaginary (within tolerance) for skew-Hermitian matrices.", "Logical inputs are promoted to double precision before testing.", "NaN values anywhere in the matrix cause the predicate to return `false`.", "Additional singleton trailing dimensions are allowed; higher-dimensional inputs with more than two non-singleton axes raise an error."],
    },
    BuiltinDocumentationSection {
        heading: "GPU residency and provider behavior",
        paragraphs: &["Manual `gpuArray` / `gather` calls are optional. When a provider implements the Hermitian predicate the computation runs entirely on the GPU and only the boolean result is read back. Otherwise the runtime gathers the tensor automatically and evaluates the predicate on the host.", "RunMat keeps GPU tensors resident whenever possible. When the active acceleration provider exposes an `ishermitian` predicate hook the comparison runs entirely on the device and only a logical scalar is transferred back. Providers without a dedicated hook fall back to gathering the matrix and running the CPU implementation transparently, so correctness is preserved even without GPU specialisation.", "Falls back to gathering GPU operands when providers lack a device-side Hermitian predicate."],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "How to check if a complex matrix is Hermitian",
        program: "A = [2   1-3i; 1+3i   5];\ntf = ishermitian(A)",
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
        title: "How to determine that a real matrix is Hermitian",
        program: "B = [4 2 2; 2 7 3; 2 3 9];\ntf = ishermitian(B)",
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
        id: "example-3",
        title: "How to spot a matrix that is not Hermitian",
        program: "C = [1 2+i; 2-i 1+0.02i];\ntf = ishermitian(C)",
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
        id: "example-4",
        title: "How to allow numerical noise with a tolerance",
        program: "D = [1 1+1e-12; 1-1e-12 1];\ntf = ishermitian(D, 1e-9)",
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
        id: "example-5",
        title: "How to test for skew-Hermitian structure",
        program: "S = [0 2+3i; -2+3i 0];\ntf = ishermitian(S, 'skew')",
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
        title: "How to inspect a GPU-resident matrix",
        program: "G = gpuArray([3 4-2i; 4+2i 6]);\ntf = ishermitian(G)",
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
    BuiltinDocumentationFaq { question: "Does `ishermitian` work with complex matrices?", answer: "Yes. Off-diagonal elements are compared against the conjugate transpose, and diagonal elements must be real (for Hermitian) or purely imaginary (for skew-Hermitian) within the supplied tolerance." },
    BuiltinDocumentationFaq { question: "What tolerance values are supported?", answer: "Pass a finite, non-negative scalar tolerance. The comparison uses the tolerance on the magnitude of the element-wise difference between the matrix and its (signed) conjugate transpose." },
    BuiltinDocumentationFaq { question: "Can I test for skew-Hermitian matrices?", answer: "Yes. Provide `'skew'` or `'skewhermitian'` (optionally along with a tolerance) to require `A == -conj(A')`." },
    BuiltinDocumentationFaq { question: "How are diagonal elements handled?", answer: "For Hermitian matrices the diagonal entries must be real. For skew-Hermitian matrices the diagonal entries must have zero real part. Imaginary parts are unconstrained in skew mode." },
    BuiltinDocumentationFaq { question: "How does the builtin treat NaN values?", answer: "If any element of the matrix is NaN the predicate returns `false`, matching MATLAB behaviour." },
    BuiltinDocumentationFaq { question: "Do I need to reshape vectors before calling `ishermitian`?", answer: "Vectors are treated as matrices; column vectors are `n×1` matrices and row vectors are `1×n` matrices. Non-square shapes always return `false`." },
    BuiltinDocumentationFaq { question: "What happens with empty matrices?", answer: "Empty square matrices (`0×0`) return `true`, whereas empty rectangular matrices return `false`." },
    BuiltinDocumentationFaq { question: "Do I need to call `gpuArray` to use `ishermitian` on the GPU?", answer: "No. RunMat automatically keeps tensors resident on the GPU when it is profitable. Explicit `gpuArray` calls remain available for MATLAB compatibility." },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "issymmetric", target: BuiltinDocumentationLinkTarget::Builtin("issymmetric") },
    BuiltinDocumentationLink { label: "eig", target: BuiltinDocumentationLinkTarget::Builtin("eig") },
    BuiltinDocumentationLink { label: "chol", target: BuiltinDocumentationLinkTarget::Builtin("chol") },
    BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuarray") },
    BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Builtin("gather") },
    BuiltinDocumentationLink { label: "bandwidth", target: BuiltinDocumentationLinkTarget::Builtin("bandwidth") },
    BuiltinDocumentationLink { label: "symrcm", target: BuiltinDocumentationLinkTarget::Builtin("symrcm") },
    BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/structure/ishermitian.rs") },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("ishermitian"),
    slug: Some("ishermitian"),
    summary: "Test Hermitian structure of matrices in MATLAB and RunMat.",
    description: "`ishermitian(A)` returns true when `A` equals its conjugate transpose. Optional skew-Hermitian mode and tolerance behavior follow MATLAB semantics.",
    keywords: &["ishermitian", "hermitian", "skew-hermitian", "matrix structure", "gpu"],
    related: &["issymmetric", "eig", "chol", "gpuarray", "gather", "bandwidth", "symrcm"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/structure/ishermitian.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Existing runtime unit tests", location: "builtins::math::linalg::structure::ishermitian::tests" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Existing runtime integration tests", location: "builtins::math::linalg::structure::ishermitian::tests::ishermitian_gpu_roundtrip" }],
        notes: &["Source-only catalog migration. Compiled conservation and executable example verification remain pending."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
