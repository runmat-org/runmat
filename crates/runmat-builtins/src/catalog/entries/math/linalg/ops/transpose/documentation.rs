use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior and supported arrays",
        paragraphs: &["`B = transpose(A)` swaps the rows and columns of a vector or matrix without conjugating complex values. It is equivalent to MATLAB `A.'` and preserves the input class.", "Works for scalars, vectors, and matrices. Use `permute` to reorder N-D arrays.", "All eight integer classes preserve exact values and class across dense, sparse, empty, and supported gpuArray inputs.", "Complex values are **not** conjugated. Use `ctranspose`/`A'` for conjugate transpose.", "Logical, string, character, and cell arrays preserve their types and metadata.", "Vectors become column or row matrices as needed (e.g., `size(transpose(1:3)) == [3 1]`).", "Empty and singleton dimensions follow MATLAB's column-major semantics."],
    },
    BuiltinDocumentationSection {
        heading: "GPU residency and provider behavior",
        paragraphs: &["No additional residency management is required. If the planner keeps your data on the GPU, `transpose` will honour that residency and either invoke a provider kernel or, in the worst case, perform a gather/transpose/upload round-trip automatically.", "When a gpuArray is provided, RunMat first asks the active Accel provider for a dedicated transpose kernel. The WGPU backend ships such a kernel today; other providers may supply their own implementation. If the hook is missing, RunMat gathers the data to the host, performs the transpose once, and re-uploads it so downstream GPU work continues without residency churn.", "Prefers the provider transpose hook; falls back to gather→transpose→upload when unavailable."],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "Transposing a numeric matrix to swap rows and columns",
        program: "A = [1 2 3; 4 5 6];\nB = transpose(A)",
        display_output: Some("B =\n     1     4\n     2     5\n     3     6"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(B, [1 4; 2 5; 3 6]));",
        },
    },
    BuiltinExample {
        id: "example-2",
        title: "Turning a row vector into a column vector",
        program: "row = 1:4;\ncol = transpose(row);\nsize(col)",
        display_output: Some("ans = [4 1]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(col), [4 1]));",
        },
    },
    BuiltinExample {
        id: "example-3",
        title: "Preserving imaginary parts when transposing complex matrices",
        program: "Z = [1+2i 3-4i];\nZT = transpose(Z)",
        display_output: Some("ZT =\n     1.0000 + 2.0000i\n     3.0000 - 4.0000i"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(ZT, [1+2i; 3-4i]));",
        },
    },
    BuiltinExample {
        id: "example-4",
        title: "Transposing logical masks while keeping logical type",
        program: "mask = logical([1 0 1; 0 1 0]);\nmaskT = transpose(mask);\nclass(maskT)",
        display_output: Some("ans = 'logical'"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(maskT, logical([1 0; 0 1; 1 0])));",
        },
    },
    BuiltinExample {
        id: "example-5",
        title: "Transposing character arrays to flip rows and columns of text",
        program: "C = ['r' 'u' 'n'; 'm' 'a' 't'];\nCT = transpose(C)",
        display_output: Some("CT =\n    'rm'\n    'ua'\n    'nt'"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(CT), [3 2]));",
        },
    },
    BuiltinExample {
        id: "example-6",
        title: "Transposing gpuArray data without leaving the device",
        program: "G = gpuArray(rand(1024, 32));\nGT = transpose(G);\nisgpuarray(GT)",
        display_output: Some("ans = logical 1"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isgpuarray(GT)); assert(isequal(size(GT), [32 1024]));",
        },
    },
];
const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Does `transpose` conjugate complex numbers?", answer: "No. Use `ctranspose` or the `'` operator for conjugate transpose." },
    BuiltinDocumentationFaq { question: "What happens for tensors with more than two dimensions?", answer: "The public `transpose` contract is for vectors and matrices. Use `permute(A, [2 1 3 ...])` for N-D arrays." },
    BuiltinDocumentationFaq { question: "Do empty matrices stay empty after transposition?", answer: "Yes. MATLAB's empty-dimension rules are preserved exactly." },
    BuiltinDocumentationFaq { question: "Is the result a copy or a view?", answer: "It is a new array. Neither the input nor the output share storage." },
    BuiltinDocumentationFaq { question: "Can I transpose cell arrays?", answer: "Yes—RunMat mirrors MATLAB by rearranging each cell handle into the new layout." },
    BuiltinDocumentationFaq { question: "Are logical arrays still logical after transpose?", answer: "Absolutely. The data stay in compact logical storage." },
    BuiltinDocumentationFaq { question: "How does `transpose` interact with the fusion planner?", answer: "Fusion treats transposes as pipeline boundaries, so kernels before and after the transpose can still fuse independently." },
    BuiltinDocumentationFaq { question: "What if my provider lacks a transpose kernel?", answer: "RunMat transparently gathers, transposes on the host, and re-uploads while logging a warning." },
    BuiltinDocumentationFaq { question: "Does `transpose` change sparse matrices?", answer: "Sparse matrices are supported and retain their sparse storage and numeric class." },
    BuiltinDocumentationFaq { question: "Can I compose `transpose` with `permute`?", answer: "Yes—`transpose` is equivalent to `permute(A, [2 1 3 ...])`." },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "ctranspose", target: BuiltinDocumentationLinkTarget::Builtin("ctranspose") },
    BuiltinDocumentationLink { label: "permute", target: BuiltinDocumentationLinkTarget::Builtin("permute") },
    BuiltinDocumentationLink { label: "mtimes", target: BuiltinDocumentationLinkTarget::Builtin("mtimes") },
    BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuarray") },
    BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Builtin("gather") },
    BuiltinDocumentationLink { label: "dot", target: BuiltinDocumentationLinkTarget::Builtin("dot") },
    BuiltinDocumentationLink { label: "mldivide", target: BuiltinDocumentationLinkTarget::Builtin("mldivide") },
    BuiltinDocumentationLink { label: "mpower", target: BuiltinDocumentationLinkTarget::Builtin("mpower") },
    BuiltinDocumentationLink { label: "mrdivide", target: BuiltinDocumentationLinkTarget::Builtin("mrdivide") },
    BuiltinDocumentationLink { label: "trace", target: BuiltinDocumentationLinkTarget::Builtin("trace") },
    BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/ops/transpose.rs") },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("transpose"),
    slug: Some("transpose"),
    summary: "Transpose arrays by swapping the first two dimensions without complex conjugation.",
    description: "`B = transpose(A)` swaps the rows and columns of a vector or matrix without conjugating complex values. It is equivalent to MATLAB `A.'` and preserves the input class.",
    keywords: &["transpose", "swap rows and columns", "non-conjugate transpose", "gpu"],
    related: &["ctranspose", "permute", "mtimes", "gpuarray", "gather", "dot", "mldivide", "mpower", "mrdivide", "trace"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/ops/transpose.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Existing runtime unit tests", location: "builtins::math::linalg::ops::transpose::tests" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Existing runtime integration tests", location: "builtins::math::linalg::ops::transpose::tests::transpose_gpu_roundtrip" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::WgpuTest, label: "Existing GPU parity tests", location: "builtins::math::linalg::ops::transpose::tests::transpose_wgpu_matches_cpu" }],
        notes: &["Source-only catalog migration. Compiled conservation and executable example verification remain pending."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
