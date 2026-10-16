use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior and supported arrays",
        paragraphs: &["`r = symrcm(A)` returns a permutation vector for reordering square matrices to reduce bandwidth. Use `A(r, r)` to obtain a tighter band structure. More than one valid RCM ordering can exist, so exact permutation values need not match another implementation.",
            "Treats the input as an undirected graph based on the structural pattern of `A` and `A'`. Any value that is not exactly zero (including `NaN` or `Inf`) creates an edge; diagonal entries are ignored.",
            "Accepts full or sparse numeric, logical, and complex matrices. Host sparse CSC indices and native typed values build the graph without densification. Explicit numeric zeros are ignored; logical sparse pattern entries are true. Integer entries are tested for zero exactly, and complex entries are nonzero when either component is nonzero.",
            "Requires a square matrix (scalars and the empty matrix are allowed). Inputs with more than two non-singleton dimensions raise an error, matching MATLAB matrix semantics.",
            "Processes disconnected components independently. A degree/index-stable pseudoperipheral search chooses each component start, then Cuthill-McKee breadth-first search and reversal yield a deterministic symmetric ordering.",
            "Returns a row vector of 1-based indices. Fully diagonal or zero matrices produce the identity permutation."],
    },
    BuiltinDocumentationSection {
        heading: "GPU residency and provider behavior",
        paragraphs: &["No. The runtime gathers GPU matrices automatically when computing the permutation today. When native GPU implementations land, the same builtin will run entirely on the device without code changes. You can still call `gpuArray` explicitly to mirror MATLAB workflows, but it is optional in RunMat.", "RunMat first asks the active acceleration provider whether it implements the `sym_rcm` hook. The WGPU backend exposes this hook today and downloads the matrix once to reuse the optimised CPU algorithm before returning the permutation. Providers without support (or those that report an error) trigger an automatic gather and reuse the host implementation, so results stay correct regardless of GPU capabilities.", "Acceleration providers may implement the `sym_rcm` hook. The WGPU backend currently downloads the matrix, runs the host algorithm, and returns the permutation."],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "example-1", title: "Reducing bandwidth of a near-band matrix", program: "A = [4 1 0 0 2;\n     1 4 1 0 0;\n     0 1 4 1 0;\n     0 0 1 4 1;\n     2 0 0 1 4];\nr = symrcm(A);\nB = A(r, r);\nbandwidth(A)\nbandwidth(B)", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(sort(r(:))', 1:numel(r))); assert(isequal(size(B), size(A)));" } },
    BuiltinExample { id: "example-2", title: "Handling disconnected components with `symrcm`", program: "A = [1 1 0 0 0;\n     1 1 0 0 0;\n     0 0 1 0 1;\n     0 0 0 1 1;\n     0 0 1 1 1];\nr = symrcm(A);\nB = A(r, r)", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(sort(r(:))', 1:numel(r))); assert(isequal(size(B), size(A)));" } },
    BuiltinExample { id: "example-3", title: "Using `symrcm` with logical adjacency matrices", program: "adj = logical([0 1 0 0;\n               1 0 1 0;\n               0 1 0 1;\n               0 0 1 0]);\nr = symrcm(adj)", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(sort(r(:))', 1:numel(r)));" } },
    BuiltinExample { id: "example-4", title: "Applying `symrcm` to GPU matrices", program: "G = gpuArray([0 1 0 0 2;\n              1 0 1 0 0;\n              0 1 0 1 0;\n              0 0 1 0 1;\n              2 0 0 1 0]);\nr = symrcm(G);\nH = gather(G(r, r))", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Wgpu, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(sort(r(:))', 1:numel(r))); assert(isequal(size(H), size(G)));" } },
];
const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "What class of matrices benefit from `symrcm`?", answer: "Sparse or full square matrices whose graph bandwidth dominates solver cost, such as discretised PDE operators or graph Laplacians. Host sparse CSC input builds the symmetric graph directly without densifying." },
    BuiltinDocumentationFaq { question: "Does `symrcm` modify the matrix?", answer: "No. It returns a permutation vector. Apply it as `A(r, r)` or reorder vectors with `x(r)`." },
    BuiltinDocumentationFaq { question: "What happens if the matrix is not symmetric?", answer: "RunMat mirrors MATLAB and forms the pattern of `A + A'` internally. As long as the matrix has symmetric nonzero structure, `symrcm` works. Strongly asymmetric inputs may lead to less effective orderings but still produce a valid permutation." },
    BuiltinDocumentationFaq { question: "Are diagonal entries considered?", answer: "Diagonal entries are ignored when building the graph. Only off-diagonal nonzeros contribute edges, which aligns with MATLAB semantics." },
    BuiltinDocumentationFaq { question: "How are NaNs or Infs handled?", answer: "Any value that is not numerically equal to zero is treated as nonzero, including `NaN` or `Inf`. This matches MATLAB's structural interpretation." },
    BuiltinDocumentationFaq { question: "Can I use `symrcm` on dense matrices?", answer: "Yes. Dense inputs are supported. For dense matrices, the result is typically the identity permutation because all vertices have high degree." },
    BuiltinDocumentationFaq { question: "How do I verify the permutation improves bandwidth?", answer: "Compute `bandwidth(A)` and `bandwidth(A(r, r))` and compare the results. Lower lower/upper bandwidth values indicate a tighter band and more efficient factorizations." },
    BuiltinDocumentationFaq { question: "What output shape should I expect?", answer: "The permutation is returned as a row vector (`1 × n`). The empty matrix returns the empty row vector `[]`." },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "bandwidth", target: BuiltinDocumentationLinkTarget::Builtin("bandwidth") },
    BuiltinDocumentationLink { label: "issymmetric", target: BuiltinDocumentationLinkTarget::Builtin("issymmetric") },
    BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuarray") },
    BuiltinDocumentationLink { label: "ishermitian", target: BuiltinDocumentationLinkTarget::Builtin("ishermitian") },
    BuiltinDocumentationLink { label: "Source", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/structure/symrcm.rs") },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("symrcm"),
    slug: Some("symrcm"),
    summary: "Compute symmetric reverse Cuthill-McKee permutations that reduce matrix bandwidth.",
    description: "`r = symrcm(A)` returns a permutation vector for reordering symmetric or structurally symmetric matrices to reduce bandwidth. Use `A(r, r)` to obtain a tighter band structure, matching MATLAB-compatible ordering behavior.",
    keywords: &["symrcm", "reverse cuthill-mckee", "bandwidth reduction", "symmetric ordering", "permutation", "gpu"],
    related: &["bandwidth", "issymmetric", "gpuarray", "ishermitian"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/structure/symrcm.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Existing runtime unit tests", location: "builtins::math::linalg::structure::symrcm::tests" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Existing runtime integration tests", location: "builtins::math::linalg::structure::symrcm::tests::symrcm_gpu_roundtrip" }],
        notes: &["Source-only catalog migration. Compiled conservation and executable example verification remain pending."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
