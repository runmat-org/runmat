use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Matrix geometry", paragraphs: &["`ismatrix(A)` is true when the MATLAB-visible rank is at most two. Scalars, row and column vectors, and ordinary two-dimensional arrays are matrices. The rule applies across numeric, logical, character, string, cell, struct, object, sparse, and complex arrays.", "Empty arrays with at most two visible dimensions are matrices. An empty or populated array with a nonsingleton third or later dimension is not."] },
    BuiltinDocumentationSection { heading: "Trailing dimensions and residency", paragraphs: &["Trailing singleton dimensions do not increase MATLAB-visible rank, so 2-by-3-by-1 and 1-by-1-by-1 remain matrices. A nonsingleton later dimension makes the result false.", "gpuArray and distributed values are classified from validated shape metadata without launching kernels or transferring payload data. The result is a host logical scalar."] },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "matrix", title: "Check a two-dimensional matrix", program: "tf = ismatrix([1 2 3; 4 5 6])", display_output: Some("tf = true"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(tf);" } },
    BuiltinExample { id: "scalar-vector", title: "Check scalars and vectors", program: "scalar = ismatrix(42);\nrow = ismatrix(1:5);\ncolumn = ismatrix((1:5)')", display_output: Some("All results are true"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(scalar);\nassert(row);\nassert(column);" } },
    BuiltinExample { id: "higher-rank", title: "Reject a three-dimensional array", program: "tf = ismatrix(ones(2, 2, 3))", display_output: Some("tf = false"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(~tf);" } },
    BuiltinExample { id: "empty", title: "Compare empty shapes", program: "plain = ismatrix([]);\nrow = ismatrix(zeros(1, 0));\ncolumn = ismatrix(zeros(0, 1));\nhigher = ismatrix(zeros(0, 0, 3))", display_output: Some("The first three are true; higher is false"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(plain);\nassert(row);\nassert(column);\nassert(~higher);" } },
    BuiltinExample { id: "trailing-singletons", title: "Ignore trailing singleton dimensions", program: "tf = ismatrix(ones(2, 3, 1, 1))", display_output: Some("tf = true"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(tf);" } },
    BuiltinExample { id: "containers", title: "Use cell and string arrays", program: "cells = ismatrix({1, 2; 3, 4});\nstrings = ismatrix([\"a\" \"b\" \"c\"])", display_output: Some("Both results are true"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(cells);\nassert(strings);" } },
    BuiltinExample { id: "gpu-shape", title: "Inspect a resident matrix", program: "G = gpuArray(ones(4, 4));\ntf = ismatrix(G)", display_output: Some("tf = true"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Wgpu, verification: BuiltinExampleVerification::Assertions { source: "assert(tf);\nassert(~isgpuarray(tf));" } },
];
const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Are scalars and vectors matrices?", answer: "Yes. They all have at most two visible dimensions." },
    BuiltinDocumentationFaq { question: "Do trailing singleton dimensions matter?", answer: "No. They do not increase MATLAB-visible rank. A nonsingleton third or later dimension does." },
    BuiltinDocumentationFaq { question: "Are empty arrays matrices?", answer: "Two-dimensional empty shapes are matrices; a nonsingleton later dimension makes an empty array nonmatrix." },
    BuiltinDocumentationFaq { question: "Does gpuArray input gather?", answer: "No. RunMat reads the required dimensions from the handle and returns a host logical scalar." },
];
const RELATED: &[&str] = &[
    "isscalar", "isvector", "isrow", "iscolumn", "isempty", "size", "ndims", "gpuArray",
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "isscalar", target: BuiltinDocumentationLinkTarget::Builtin("isscalar") },
    BuiltinDocumentationLink { label: "isvector", target: BuiltinDocumentationLinkTarget::Builtin("isvector") },
    BuiltinDocumentationLink { label: "isrow", target: BuiltinDocumentationLinkTarget::Builtin("isrow") },
    BuiltinDocumentationLink { label: "iscolumn", target: BuiltinDocumentationLinkTarget::Builtin("iscolumn") },
    BuiltinDocumentationLink { label: "isempty", target: BuiltinDocumentationLinkTarget::Builtin("isempty") },
    BuiltinDocumentationLink { label: "length", target: BuiltinDocumentationLinkTarget::Builtin("length") },
    BuiltinDocumentationLink { label: "numel", target: BuiltinDocumentationLinkTarget::Builtin("numel") },
    BuiltinDocumentationLink { label: "size", target: BuiltinDocumentationLinkTarget::Builtin("size") },
    BuiltinDocumentationLink { label: "ndims", target: BuiltinDocumentationLinkTarget::Builtin("ndims") },
    BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") },
    BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Builtin("gather") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/introspection/ismatrix.rs") },
];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Shape-predicate runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/introspection/shape_predicate.rs") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Scalar, vector, matrix, empty, trailing-singleton, container, and resident shapes", location: "crates/runmat-runtime/src/builtins/array/introspection/ismatrix/tests.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Validated distributed global-shape behavior", location: "crates/runmat-runtime/src/builtins/array/introspection/shape_predicate.rs::tests::distributed_values_use_validated_global_shape_without_materialization" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::WgpuTest, label: "Actual WGPU matrix-shape metadata", location: "crates/runmat-runtime/src/builtins/array/introspection/ismatrix/tests.rs::wgpu_reads_shape_without_payload_transfer" },
    ],
    notes: &[],
};
pub(super) const ISMATRIX_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("ismatrix"),
    slug: Some("ismatrix"),
    summary: "Determine whether an array has matrix geometry.",
    description:
        "`ismatrix` returns one host logical scalar when the MATLAB-visible rank is at most two.",
    keywords: &[
        "ismatrix",
        "matrix",
        "matrix detection",
        "rank",
        "shape",
        "metadata",
        "metadata query",
        "logical",
        "gpu",
        "gpuArray",
    ],
    related: RELATED,
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
