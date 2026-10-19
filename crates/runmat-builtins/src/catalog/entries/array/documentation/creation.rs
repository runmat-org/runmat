use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};

const FULL_EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink {
        label: "Runtime implementation",
        target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/creation/full.rs"),
    }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Sparse, typed, compatibility, and residency tests", location: "builtins::array::creation::full::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "VM sparse materialization test", location: "crates/runmat-vm/tests/logic.rs::full_densifies_sparse_storage_through_vm_dispatch" },
    ],
    notes: &[],
};

const FULL_SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`A = full(S)` materializes sparse double, single, logical, or supported complex storage as a dense matrix with the same class, values, row count, and column count. Stored values follow column-major order; unstored positions become explicit zeros.",
        "Input that is already full passes through unchanged. This includes real and complex numeric values, logical and character arrays, and resident gpuArray handles. Cells, structs, strings, objects, function handles, symbolic values, classes, and exceptions return `RunMat:full:InvalidInput`.",
        "RunMat compatibility mode also densifies sparse matrices with any fixed-width integer value class. That exact sparse integer representation is separately gated because the compatible sparse value surface is limited to double, single, and logical storage.",
    ] },
    BuiltinDocumentationSection { heading: "GPU execution", paragraphs: &[
        "RunMat currently materializes sparse matrices on the host. An already-dense gpuArray is already full, so `full` returns its existing handle and owning-provider placement without gathering or consulting a newly registered provider.",
    ] },
];

const FULL_EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "sparse-matrix", title: "Convert sparse storage to a dense matrix", program: "S = sparse([1; 3; 2], [1; 1; 2], [10; 30; 20], 3, 2);\nA = full(S)", display_output: Some("A = [10 0; 0 20; 30 0]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(~issparse(A));\nassert(isequal(A, [10 0; 0 20; 30 0]));" } },
    BuiltinExample { id: "empty-sparse", title: "Materialize an empty sparse matrix", program: "S = sparse(2, 3);\nA = full(S)", display_output: Some("A = zeros(2, 3)"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(~issparse(A));\nassert(isequal(size(A), [2 3]));\nassert(isequal(A, zeros(2, 3)));" } },
    BuiltinExample { id: "dense-identity", title: "Keep an already-full array unchanged", program: "source = uint16([1 0; 0 2]);\nA = full(source)", display_output: Some("A remains a uint16 matrix"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(A, \"uint16\"));\nassert(isequal(A, source));" } },
    BuiltinExample { id: "logical-sparse", title: "Preserve logical class while densifying", program: "S = sparse(logical([1 0; 0 1]));\nA = full(S)", display_output: Some("A = logical(eye(2))"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(A, \"logical\"));\nassert(~issparse(A));\nassert(isequal(A, logical(eye(2))));" } },
    BuiltinExample { id: "gpu-identity", title: "Keep an already-full gpuArray resident", program: "G = gpuArray(uint16([1 2; 3 4]));\nH = full(G);\nhost = gather(H)", display_output: Some("host remains a uint16 matrix"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Wgpu, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(host, \"uint16\"));\nassert(isequal(host, uint16([1 2; 3 4])));" } },
];

const FULL_FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Does `full` change matrix values?", answer: "No. It changes sparse storage to dense storage while preserving class, shape, and stored values; unstored positions become explicit zeros." },
    BuiltinDocumentationFaq { question: "What happens to an already-full value?", answer: "Supported dense values pass through unchanged." },
    BuiltinDocumentationFaq { question: "Does `full` gather a dense gpuArray?", answer: "No. It returns the resident handle with the same owner and placement." },
    BuiltinDocumentationFaq { question: "Does `full` preserve integer classes?", answer: "Yes for dense integers. RunMat mode also supports exact sparse integer storage and densifies it without binary64 conversion." },
    BuiltinDocumentationFaq { question: "Is sparse integer input portable?", answer: "No. Exact fixed-width sparse value storage is a RunMat extension; compatible sparse values are double, single, or logical." },
    BuiltinDocumentationFaq { question: "Can `full` create sparse GPU output?", answer: "No. `full` produces dense host storage from sparse host input; already-dense gpuArray input stays resident." },
];

const FULL_LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "issparse",
        target: BuiltinDocumentationLinkTarget::Builtin("issparse"),
    },
    BuiltinDocumentationLink {
        label: "nnz",
        target: BuiltinDocumentationLinkTarget::Builtin("nnz"),
    },
    BuiltinDocumentationLink {
        label: "find",
        target: BuiltinDocumentationLinkTarget::Builtin("find"),
    },
    BuiltinDocumentationLink {
        label: "gather",
        target: BuiltinDocumentationLinkTarget::Builtin("gather"),
    },
    BuiltinDocumentationLink {
        label: "sparse",
        target: BuiltinDocumentationLinkTarget::Builtin("sparse"),
    },
];

pub(in super::super) const FULL_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("full"), slug: Some("full"), summary: "Convert sparse matrix storage to dense full storage.",
    description: "`full(S)` materializes sparse values as a dense matrix while preserving class, shape, and values. Already-full supported values pass through unchanged.",
    keywords: &["full", "sparse", "dense", "matrix", "storage"],
    related: &["find", "gather", "issparse", "nnz", "sparse"],
    sections: FULL_SECTIONS, examples: FULL_EXAMPLES, example_exemption: None, faqs: FULL_FAQS,
    links: FULL_LINKS, media: &[], evidence: FULL_EVIDENCE, introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};

const ZEROS_EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink {
        label: "Runtime implementation",
        target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/creation/zeros.rs"),
    }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Shape, class, prototype, compatibility, and provider tests", location: "builtins::array::creation::zeros::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::ProviderTest, label: "Resident prototype allocation tests", location: "builtins::array::creation::zeros::tests::zeros_gpu_like_alloc" },
    ],
    notes: &[],
};

const ZEROS_SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`zeros()` returns scalar double zero. `zeros(n)` creates an `n`-by-`n` matrix; separate dimensions or a row size vector create the requested dense N-D shape. Size controls accept double, single, and all eight fixed-width integer classes.",
        "A trailing class name selects double, single, logical, or any fixed-width integer output. `zeros(..., \"like\", prototype)` preserves the prototype's class, sparsity, complexity, and supported residency. Integer output uses native fixed-width storage.",
        "RunMat mode additionally accepts a column size vector, a resident numeric size control, or a prototype without the `like` keyword. Use the explicit compatible forms when source must run under a MATLAB compatibility pin.",
    ] },
    BuiltinDocumentationSection { heading: "GPU execution", paragraphs: &[
        "A resident `like` prototype selects its owning provider and physical element type. RunMat asks that provider for a zero-filled allocation and can fall back to a typed host upload when the provider lacks a zero-allocation hook.",
        "The planner may also place eligible constructor results on an accelerator when doing so benefits a surrounding expression. Explicit resident prototypes retain user-selected placement and ownership.",
    ] },
];

const ZEROS_EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "scalar", title: "Create scalar zero", program: "z = zeros()", display_output: Some("z = 0"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(z, \"double\"));\nassert(isequal(size(z), [1 1]));\nassert(z == 0);" } },
    BuiltinExample { id: "matrix", title: "Create a rectangular matrix", program: "A = zeros(2, 3)", display_output: Some("A = [0 0 0; 0 0 0]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(A, \"double\"));\nassert(isequal(size(A), [2 3]));\nassert(all(A(:) == 0));" } },
    BuiltinExample { id: "logical", title: "Create a logical column", program: "mask = zeros(4, 1, \"logical\")", display_output: Some("mask is a 4-by-1 logical array of false values"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(mask, \"logical\"));\nassert(isequal(size(mask), [4 1]));\nassert(~any(mask(:)));" } },
    BuiltinExample { id: "integer-class", title: "Create native fixed-width integer storage", program: "counts = zeros(2, 3, \"uint16\")", display_output: Some("counts is a 2-by-3 uint16 matrix"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(counts, \"uint16\"));\nassert(isequal(counts, uint16([0 0 0; 0 0 0])));" } },
    BuiltinExample { id: "size-vector", title: "Create an N-D array from a size vector", program: "T = zeros([2 3 4]);\nshape = size(T)", display_output: Some("shape = [2 3 4]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(shape, [2 3 4]));\nassert(all(T(:) == 0));" } },
    BuiltinExample { id: "host-like", title: "Match a host prototype's class", program: "prototype = single([1 2; 3 4]);\nA = zeros(3, 2, \"like\", prototype)", display_output: Some("A is a 3-by-2 single matrix"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(A, \"single\"));\nassert(isequal(size(A), [3 2]));\nassert(isequal(A, single([0 0; 0 0; 0 0])));" } },
    BuiltinExample { id: "gpu-like", title: "Allocate zeros with a resident prototype", program: "prototype = gpuArray(single([1 2; 3 4]));\nG = zeros(3, 2, \"like\", prototype);\nhost = gather(G)", display_output: Some("host is a 3-by-2 single matrix"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Wgpu, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(host, \"single\"));\nassert(isequal(size(host), [3 2]));\nassert(isequal(host, single([0 0; 0 0; 0 0])));" } },
    BuiltinExample { id: "implicit-prototype-extension", title: "Infer shape and class from a prototype in RunMat mode", program: "prototype = int32([1 2; 3 4]);\nA = zeros(prototype)", display_output: Some("A is a 2-by-2 int32 zero matrix"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(A, \"int32\"));\nassert(isequal(size(A), [2 2]));\nassert(isequal(A, int32([0 0; 0 0])));" } },
];

const ZEROS_FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "What does `zeros()` return?", answer: "It returns scalar double zero." },
    BuiltinDocumentationFaq { question: "What does `zeros(n)` return?", answer: "It returns an `n`-by-`n` dense double matrix." },
    BuiltinDocumentationFaq { question: "How do I create an N-D array?", answer: "Pass each dimension separately or provide one row size vector, such as `zeros([2 3 4])`." },
    BuiltinDocumentationFaq { question: "Which output classes can I request?", answer: "Double, single, logical, and all eight fixed-width integer classes are supported." },
    BuiltinDocumentationFaq { question: "How do I match another value?", answer: "Use `zeros(..., \"like\", prototype)` to preserve its class, sparsity, complexity, and supported residency." },
    BuiltinDocumentationFaq { question: "Can `zeros` allocate on a GPU?", answer: "Yes. A resident `like` prototype selects the owning provider; the planner can also place eligible results automatically." },
    BuiltinDocumentationFaq { question: "Does a provider need a native zero-allocation hook?", answer: "No. RunMat can upload a correctly typed host zero value when that hook is unavailable." },
    BuiltinDocumentationFaq { question: "Is `zeros(A)` portable prototype syntax?", answer: "No. The implicit prototype form is a RunMat extension; use `zeros(..., \"like\", A)` for compatible source." },
    BuiltinDocumentationFaq { question: "Is output always dense?", answer: "Ordinary constructor forms are dense. A sparse `like` prototype preserves sparse storage." },
    BuiltinDocumentationFaq { question: "Why preallocate with `zeros`?", answer: "Preallocation fixes the intended class and shape before indexed assignment and can avoid repeated growth." },
];

const ZEROS_LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "ones",
        target: BuiltinDocumentationLinkTarget::Builtin("ones"),
    },
    BuiltinDocumentationLink {
        label: "eye",
        target: BuiltinDocumentationLinkTarget::Builtin("eye"),
    },
    BuiltinDocumentationLink {
        label: "gpuArray",
        target: BuiltinDocumentationLinkTarget::Builtin("gpuArray"),
    },
    BuiltinDocumentationLink {
        label: "gather",
        target: BuiltinDocumentationLinkTarget::Builtin("gather"),
    },
    BuiltinDocumentationLink {
        label: "false",
        target: BuiltinDocumentationLinkTarget::Builtin("false"),
    },
    BuiltinDocumentationLink {
        label: "sparse",
        target: BuiltinDocumentationLinkTarget::Builtin("sparse"),
    },
];

pub(in super::super) const ZEROS_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("zeros"), slug: Some("zeros"), summary: "Create zero-filled arrays with a selected shape, class, storage, and residency.",
    description: "`zeros` creates scalar, matrix, or N-D zero values from dimensions or a size vector. Class names and `like` prototypes select typed storage and supported placement.",
    keywords: &["zeros", "array", "preallocation", "logical", "integer", "gpu", "like", "size"],
    related: &["eye", "false", "gather", "gpuArray", "ones", "sparse"],
    sections: ZEROS_SECTIONS, examples: ZEROS_EXAMPLES, example_exemption: None, faqs: ZEROS_FAQS,
    links: ZEROS_LINKS, media: &[], evidence: ZEROS_EVIDENCE, introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
