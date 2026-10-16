use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior and supported arrays",
        paragraphs: &["`Y = pagetranspose(X)` swaps the first two dimensions of `X` on every page without conjugating complex values. For N-D arrays, each output page `Y(:,:,i,j,...)` is the nonconjugate transpose of `X(:,:,i,j,...)`.",
            "Equivalent to `permute(X,[2 1 3:ndims(X)])` for dense arrays.",
            "Works for scalars, vectors, matrices, and N-D arrays; only the first two axes are swapped.",
            "Complex values are not conjugated. Compose with `conj` when Hermitian page transpose semantics are needed.",
            "All eight integer classes, logical, character, string, and cell arrays preserve their element type while their first two dimensions are swapped. Integer payloads, including values above `flintmax`, are permuted without a floating conversion.",
            "Sparse matrices transpose through RunMat's sparse CSC transpose path.",
            "Scalar structs are treated as 1-by-1 values and round-trip unchanged; true struct-array permutation will follow RunMat's struct-array representation when that lands.",
            "gpuArray inputs remain device-resident when a provider is available; unsupported integer layouts use transparent exact-owner gather and class-preserving restoration."],
    },
    BuiltinDocumentationSection {
        heading: "GPU residency and provider behavior",
        paragraphs: &["Yes for explicit `gpuArray` inputs when an acceleration provider is available.", "`pagetranspose` uses the same provider `permute` hook as `permute(X,[2 1 3:ndims(X)])`.", "If the provider lacks a native hook, RunMat gathers, performs the page-wise transpose on the host, and uploads the result back to the provider.", "Equivalent to `permute(X,[2 1 3:ndims(X)])`; providers may keep gpuArray results resident through the permute hook, otherwise RunMat gathers once, transposes on the host, and uploads the result."],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "Transpose each page of a 3-D array",
        program: "A = reshape(1:12, [2 3 2]);\nB = pagetranspose(A);\nsize(B)",
        display_output: Some("ans = [3 2 2]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(B), [3 2 2]));",
        },
    },
    BuiltinExample {
        id: "example-2",
        title: "Preserve complex values without conjugation",
        program: "Z = reshape([1+2i 3-4i], [1 2 1]);\nY = pagetranspose(Z)",
        display_output: Some("Y =\n   1.0000 + 2.0000i\n   3.0000 - 4.0000i"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(Y, [1+2i; 3-4i]));",
        },
    },
    BuiltinExample {
        id: "example-3",
        title: "Use `pagetranspose` as the page-wise form of `transpose`",
        program: "A = rand(5, 3, 10);\nB = pagetranspose(A);\nsize(B)",
        display_output: Some("ans = [3 5 10]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(B), [3 5 10]));",
        },
    },
    BuiltinExample {
        id: "example-4",
        title: "Transpose gpuArray pages while preserving residency",
        program: "G = gpuArray(rand(4, 2, 8));\nH = pagetranspose(G);\nisgpuarray(H)",
        display_output: Some("ans = logical 1"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isgpuarray(H)); assert(isequal(size(H), [2 4 8]));",
        },
    },
];
const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "How does `pagetranspose` differ from `transpose`?", answer: "For dense arrays they use the same first-two-dimension swap. `pagetranspose` names the page-wise intent explicitly for N-D arrays and mirrors MATLAB's page-wise API." },
    BuiltinDocumentationFaq { question: "Does `pagetranspose` conjugate complex numbers?", answer: "No. It is the page-wise nonconjugate transpose." },
    BuiltinDocumentationFaq { question: "What happens to dimensions three and higher?", answer: "They stay in the same order and size. Only dimensions one and two are swapped." },
    BuiltinDocumentationFaq { question: "Is the result a view?", answer: "No. RunMat materializes a new value, although GPU providers may optimize the data movement internally." },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "transpose", target: BuiltinDocumentationLinkTarget::Builtin("transpose") },
    BuiltinDocumentationLink { label: "ctranspose", target: BuiltinDocumentationLinkTarget::Builtin("ctranspose") },
    BuiltinDocumentationLink { label: "pagemtimes", target: BuiltinDocumentationLinkTarget::Builtin("pagemtimes") },
    BuiltinDocumentationLink { label: "permute", target: BuiltinDocumentationLinkTarget::Builtin("permute") },
    BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuarray") },
    BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Builtin("gather") },
    BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/ops/pagetranspose.rs") },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("pagetranspose"),
    slug: Some("pagetranspose"),
    summary: "Apply the nonconjugate transpose to every matrix page of an array.",
    description: "`Y = pagetranspose(X)` swaps the first two dimensions of `X` on every page without conjugating complex values. For N-D arrays, each output page `Y(:,:,i,j,...)` is the nonconjugate transpose of `X(:,:,i,j,...)`.",
    keywords: &["pagetranspose", "page-wise transpose", "non-conjugate transpose", "permute", "gpuArray"],
    related: &["transpose", "ctranspose", "pagemtimes", "permute", "gpuarray", "gather"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/ops/pagetranspose.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Existing runtime unit tests", location: "builtins::math::linalg::ops::pagetranspose::tests" }],
        notes: &["Source-only catalog migration. Compiled conservation and executable example verification remain pending."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
