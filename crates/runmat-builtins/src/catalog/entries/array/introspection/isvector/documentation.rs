use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Vector geometry", paragraphs: &["`isvector(A)` is true for 1-by-N and N-by-1 visible shapes, including scalars. It applies the same rule to numeric, logical, character, string, cell, struct, object, sparse, and complex arrays.", "Empty 1-by-0 and 0-by-1 arrays are vectors; an empty 0-by-3 array is not. A matrix with both visible dimensions greater than one is not a vector."] },
    BuiltinDocumentationSection { heading: "Higher dimensions", paragraphs: &["MATLAB-visible rank ignores trailing singleton dimensions. Shapes such as 3-by-1-by-1 and 1-by-3-by-1 remain vectors. A nonsingleton third or later dimension makes the result false.", "gpuArray and distributed inputs are classified from validated shape metadata without kernels or payload transfers. The result is one host logical scalar."] },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "row", title: "Check a row vector", program: "tf = isvector([1 2 3])", display_output: Some("tf = true"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(tf);" } },
    BuiltinExample { id: "matrix", title: "Reject a matrix", program: "tf = isvector([1 2 3; 4 5 6])", display_output: Some("tf = false"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(~tf);" } },
    BuiltinExample { id: "scalar", title: "Treat a scalar as a vector", program: "tf = isvector(42)", display_output: Some("tf = true"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(tf);" } },
    BuiltinExample { id: "empty-shapes", title: "Compare empty vector shapes", program: "row = isvector(zeros(1, 0));\ncolumn = isvector(zeros(0, 1));\nwide = isvector(zeros(0, 3))", display_output: Some("row and column are true; wide is false"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(row);\nassert(column);\nassert(~wide);" } },
    BuiltinExample { id: "trailing-singletons", title: "Ignore trailing singleton dimensions", program: "tf = isvector(ones(3, 1, 1))", display_output: Some("tf = true"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(tf);" } },
    BuiltinExample { id: "higher-rank", title: "Reject a nonsingleton third dimension", program: "tf = isvector(ones(1, 1, 4))", display_output: Some("tf = false"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(~tf);" } },
    BuiltinExample { id: "text", title: "Check character and string rows", program: "tf_char = isvector('RunMat');\ntf_strings = isvector([\"a\" \"b\" \"c\"])", display_output: Some("Both results are true"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(tf_char);\nassert(tf_strings);" } },
    BuiltinExample { id: "gpu-shape", title: "Inspect a resident column vector", program: "G = gpuArray((1:5)');\ntf = isvector(G)", display_output: Some("tf = true"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Wgpu, verification: BuiltinExampleVerification::Assertions { source: "assert(tf);\nassert(~isgpuarray(tf));" } },
];
const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Are scalars vectors?", answer: "Yes. A scalar has visible shape 1-by-1." },
    BuiltinDocumentationFaq { question: "Which empty arrays are vectors?", answer: "1-by-0 and 0-by-1 are vectors; shapes such as 0-by-3 are not." },
    BuiltinDocumentationFaq { question: "Do trailing singleton dimensions matter?", answer: "No. They do not increase MATLAB-visible rank. A nonsingleton third or later dimension does." },
    BuiltinDocumentationFaq { question: "Do cells, structs, strings, sparse arrays, and gpuArrays use different rules?", answer: "No. Their visible dimensions enter the same shape predicate; resident payloads are not transferred." },
];
const RELATED: &[&str] = &[
    "isscalar", "ismatrix", "isrow", "iscolumn", "isempty", "size", "ndims", "gpuArray",
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "isscalar", target: BuiltinDocumentationLinkTarget::Builtin("isscalar") },
    BuiltinDocumentationLink { label: "isempty", target: BuiltinDocumentationLinkTarget::Builtin("isempty") },
    BuiltinDocumentationLink { label: "ismatrix", target: BuiltinDocumentationLinkTarget::Builtin("ismatrix") },
    BuiltinDocumentationLink { label: "isrow", target: BuiltinDocumentationLinkTarget::Builtin("isrow") },
    BuiltinDocumentationLink { label: "iscolumn", target: BuiltinDocumentationLinkTarget::Builtin("iscolumn") },
    BuiltinDocumentationLink { label: "length", target: BuiltinDocumentationLinkTarget::Builtin("length") },
    BuiltinDocumentationLink { label: "numel", target: BuiltinDocumentationLinkTarget::Builtin("numel") },
    BuiltinDocumentationLink { label: "size", target: BuiltinDocumentationLinkTarget::Builtin("size") },
    BuiltinDocumentationLink { label: "ndims", target: BuiltinDocumentationLinkTarget::Builtin("ndims") },
    BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") },
    BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Builtin("gather") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/introspection/isvector.rs") },
];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Shape-predicate runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/introspection/shape_predicate.rs") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Row, column, scalar, empty, trailing-singleton, container, and resident shapes", location: "crates/runmat-runtime/src/builtins/array/introspection/isvector/tests.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Validated distributed global-shape behavior", location: "crates/runmat-runtime/src/builtins/array/introspection/shape_predicate.rs::tests::distributed_values_use_validated_global_shape_without_materialization" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::WgpuTest, label: "Actual WGPU vector-shape metadata", location: "crates/runmat-runtime/src/builtins/array/introspection/isvector/tests.rs::wgpu_reads_shape_without_payload_transfer" },
    ],
    notes: &[],
};
pub(super) const ISVECTOR_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("isvector"),
    slug: Some("isvector"),
    summary: "Determine whether an array has vector geometry.",
    description: "`isvector` returns one host logical scalar for a visible 1-by-N or N-by-1 shape.",
    keywords: &[
        "isvector",
        "vector",
        "vector detection",
        "row",
        "column",
        "shape",
        "metadata",
        "metadata query",
        "gpu",
        "gpuArray",
        "logical",
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
