use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Column geometry", paragraphs: &["`iscolumn(A)` is true when `A` has at most two MATLAB-visible dimensions and its second visible dimension is one. Scalars and N-by-1 arrays return true; 1-by-N arrays with N greater than one return false.", "Empty 0-by-1 arrays are columns. Empty 1-by-0 arrays are not. Trailing singleton dimensions do not change the result, while a nonsingleton third or later dimension makes it false."] },
    BuiltinDocumentationSection { heading: "Metadata-only execution", paragraphs: &["All supported value classes use the same dimensions as `size`. gpuArray and distributed inputs are classified from validated shape metadata without launching kernels or reading payload data. The result is a host logical scalar."] },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "vectors",
        title: "Compare row and column vectors",
        program: "row = iscolumn([1 2 3]);\ncolumn = iscolumn([1; 2; 3])",
        display_output: Some("row = false and column = true"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(~row);\nassert(column);",
        },
    },
    BuiltinExample {
        id: "scalar",
        title: "Treat a scalar as a column",
        program: "tf = iscolumn(42)",
        display_output: Some("tf = true"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(tf);",
        },
    },
    BuiltinExample {
        id: "empty",
        title: "Compare empty column geometry",
        program: "column = iscolumn(zeros(0, 1));\nnot_column = iscolumn(zeros(1, 0))",
        display_output: Some("column = true and not_column = false"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(column);\nassert(~not_column);",
        },
    },
    BuiltinExample {
        id: "gpu-shape",
        title: "Inspect a resident column",
        program: "G = gpuArray((1:5)');\ntf = iscolumn(G)",
        display_output: Some("tf = true"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(tf);\nassert(~isgpuarray(tf));",
        },
    },
];
const FAQS: &[BuiltinDocumentationFaq] = &[BuiltinDocumentationFaq { question: "Is a scalar a column?", answer: "Yes. A scalar has visible shape 1-by-1." }, BuiltinDocumentationFaq { question: "Does `iscolumn` require a vector?", answer: "No. Any two-dimensional N-by-1 array is a column, including a 1-by-1 scalar and a 0-by-1 empty array." }, BuiltinDocumentationFaq { question: "Does gpuArray input launch a kernel?", answer: "No. RunMat reads shape metadata and returns a host logical scalar." }];
const RELATED: &[&str] = &["isrow", "isvector", "ismatrix", "size", "gpuArray"];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "isrow", target: BuiltinDocumentationLinkTarget::Builtin("isrow") },
    BuiltinDocumentationLink { label: "isvector", target: BuiltinDocumentationLinkTarget::Builtin("isvector") },
    BuiltinDocumentationLink { label: "ismatrix", target: BuiltinDocumentationLinkTarget::Builtin("ismatrix") },
    BuiltinDocumentationLink { label: "size", target: BuiltinDocumentationLinkTarget::Builtin("size") },
    BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/introspection/iscolumn.rs") },
];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Shape-predicate runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/introspection/shape_predicate.rs") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Scalar, vector, empty, trailing-dimension, and resident column geometry", location: "crates/runmat-runtime/src/builtins/array/introspection/iscolumn/tests.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Validated distributed global-shape behavior", location: "crates/runmat-runtime/src/builtins/array/introspection/shape_predicate.rs::tests::distributed_values_use_validated_global_shape_without_materialization" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::WgpuTest, label: "Actual WGPU column-shape metadata", location: "crates/runmat-runtime/src/builtins/array/introspection/iscolumn/tests.rs::wgpu_reads_shape_without_payload_transfer" },
    ],
    notes: &[],
};
pub(super) const ISCOLUMN_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("iscolumn"),
    slug: Some("iscolumn"),
    summary: "Determine whether an array has one visible column.",
    description: "`iscolumn` returns one host logical scalar from MATLAB-visible shape metadata.",
    keywords: &[
        "iscolumn",
        "column",
        "column vector",
        "shape",
        "metadata",
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
