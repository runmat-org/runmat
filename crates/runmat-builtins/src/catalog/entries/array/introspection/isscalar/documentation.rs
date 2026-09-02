use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Scalar shape", paragraphs: &["`isscalar(A)` is true when `A` contains exactly one element and every MATLAB-visible dimension is one. Numeric, logical, complex, string, cell, struct, and object values use the same shape rule.", "Empty arrays and vectors with more than one element return false. A character value must contain exactly one character; a string scalar remains scalar even when its text is empty."] },
    BuiltinDocumentationSection { heading: "Trailing dimensions and residency", paragraphs: &["Trailing singleton dimensions do not change MATLAB-visible scalar geometry. A shape such as 1-by-1-by-1 is scalar, while any nonsingleton dimension makes the result false.", "gpuArray and distributed inputs are classified from validated shape metadata. The predicate does not launch a kernel or transfer payload data, and returns a host logical scalar."] },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "numeric", title: "Check a numeric scalar", program: "tf = isscalar(42)", display_output: Some("tf = true"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(tf);" } },
    BuiltinExample { id: "vector", title: "Reject a row vector", program: "tf = isscalar([1 2 3])", display_output: Some("tf = false"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(~tf);" } },
    BuiltinExample { id: "text", title: "Compare string and character shapes", program: "tf_string = isscalar(\"hello\");\ntf_char = isscalar('h');\ntf_char_row = isscalar('runmat')", display_output: Some("The string and one character are scalar; the character row is not"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(tf_string);\nassert(tf_char);\nassert(~tf_char_row);" } },
    BuiltinExample { id: "empty", title: "Reject an empty array", program: "tf = isscalar([])", display_output: Some("tf = false"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(~tf);" } },
    BuiltinExample { id: "cell", title: "Check a one-element cell", program: "C = {pi};\ntf = isscalar(C)", display_output: Some("tf = true"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(tf);" } },
    BuiltinExample { id: "gpu-shape", title: "Inspect a resident scalar shape", program: "G = gpuArray(ones(1, 1));\ntf = isscalar(G)", display_output: Some("tf = true"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Wgpu, verification: BuiltinExampleVerification::Assertions { source: "assert(tf);\nassert(~isgpuarray(tf));" } },
];
const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Does `isscalar([])` return true?",
        answer: "No. An empty array contains zero elements.",
    },
    BuiltinDocumentationFaq {
        question: "Is an empty string scalar?",
        answer: "Yes. Its container shape is 1-by-1 regardless of its text content.",
    },
    BuiltinDocumentationFaq {
        question: "Are 1-by-1-by-1 arrays scalar?",
        answer: "Yes. Trailing singleton dimensions do not change MATLAB-visible scalar geometry.",
    },
    BuiltinDocumentationFaq {
        question: "Does gpuArray input launch a kernel?",
        answer: "No. RunMat reads the handle shape and returns a host logical scalar.",
    },
];
const RELATED: &[&str] = &[
    "isempty", "isvector", "ismatrix", "numel", "size", "ndims", "gpuArray",
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "isempty", target: BuiltinDocumentationLinkTarget::Builtin("isempty") },
    BuiltinDocumentationLink { label: "isvector", target: BuiltinDocumentationLinkTarget::Builtin("isvector") },
    BuiltinDocumentationLink { label: "ismatrix", target: BuiltinDocumentationLinkTarget::Builtin("ismatrix") },
    BuiltinDocumentationLink { label: "numel", target: BuiltinDocumentationLinkTarget::Builtin("numel") },
    BuiltinDocumentationLink { label: "size", target: BuiltinDocumentationLinkTarget::Builtin("size") },
    BuiltinDocumentationLink { label: "length", target: BuiltinDocumentationLinkTarget::Builtin("length") },
    BuiltinDocumentationLink { label: "ndims", target: BuiltinDocumentationLinkTarget::Builtin("ndims") },
    BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") },
    BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Builtin("gather") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/introspection/isscalar.rs") },
];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Shape-predicate runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/introspection/shape_predicate.rs") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Scalar, empty, text, container, and resident geometry", location: "crates/runmat-runtime/src/builtins/array/introspection/isscalar/tests.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Validated distributed global-shape behavior", location: "crates/runmat-runtime/src/builtins/array/introspection/shape_predicate.rs::tests::distributed_values_use_validated_global_shape_without_materialization" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::WgpuTest, label: "Actual WGPU scalar-shape metadata", location: "crates/runmat-runtime/src/builtins/array/introspection/isscalar/tests.rs::wgpu_reads_shape_without_payload_transfer" },
    ],
    notes: &[],
};
pub(super) const ISSCALAR_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("isscalar"),
    slug: Some("isscalar"),
    summary: "Determine whether a value has scalar geometry.",
    description:
        "`isscalar` returns one host logical scalar from the input's MATLAB-visible shape.",
    keywords: &[
        "isscalar",
        "scalar",
        "shape",
        "metadata",
        "metadata query",
        "gpu",
        "gpuArray",
        "logical",
        "distributed",
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
