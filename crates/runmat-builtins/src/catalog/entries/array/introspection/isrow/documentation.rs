use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Row geometry", paragraphs: &["`isrow(A)` is true when `A` has at most two MATLAB-visible dimensions and its first visible dimension is one. Scalars and 1-by-N arrays return true; N-by-1 arrays with N greater than one return false.", "Empty 1-by-0 arrays are rows. Empty 0-by-N arrays are not. Trailing singleton dimensions do not change the result, while a nonsingleton third or later dimension makes it false."] },
    BuiltinDocumentationSection { heading: "Metadata-only execution", paragraphs: &["All supported value classes use the same dimensions as `size`. gpuArray and distributed inputs are classified from validated shape metadata without launching kernels or reading payload data. The result is a host logical scalar."] },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "vectors",
        title: "Compare row and column vectors",
        program: "row = isrow([1 2 3]);\ncolumn = isrow([1; 2; 3])",
        display_output: Some("row = true and column = false"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(row);\nassert(~column);",
        },
    },
    BuiltinExample {
        id: "scalar",
        title: "Treat a scalar as a row",
        program: "tf = isrow(42)",
        display_output: Some("tf = true"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(tf);",
        },
    },
    BuiltinExample {
        id: "empty",
        title: "Compare empty row geometry",
        program: "row = isrow(zeros(1, 0));\nnot_row = isrow(zeros(0, 1))",
        display_output: Some("row = true and not_row = false"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(row);\nassert(~not_row);",
        },
    },
    BuiltinExample {
        id: "gpu-shape",
        title: "Inspect a resident row",
        program: "G = gpuArray(1:5);\ntf = isrow(G)",
        display_output: Some("tf = true"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(tf);\nassert(~isgpuarray(tf));",
        },
    },
];
const FAQS: &[BuiltinDocumentationFaq] = &[BuiltinDocumentationFaq { question: "Is a scalar a row?", answer: "Yes. A scalar has visible shape 1-by-1." }, BuiltinDocumentationFaq { question: "Does `isrow` require a vector?", answer: "No. Any two-dimensional 1-by-N array is a row, including a 1-by-1 scalar and a 1-by-0 empty array." }, BuiltinDocumentationFaq { question: "Does gpuArray input launch a kernel?", answer: "No. RunMat reads shape metadata and returns a host logical scalar." }];
const RELATED: &[&str] = &["iscolumn", "isvector", "ismatrix", "size", "gpuArray"];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "iscolumn", target: BuiltinDocumentationLinkTarget::Builtin("iscolumn") },
    BuiltinDocumentationLink { label: "isvector", target: BuiltinDocumentationLinkTarget::Builtin("isvector") },
    BuiltinDocumentationLink { label: "ismatrix", target: BuiltinDocumentationLinkTarget::Builtin("ismatrix") },
    BuiltinDocumentationLink { label: "size", target: BuiltinDocumentationLinkTarget::Builtin("size") },
    BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/introspection/isrow.rs") },
];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Shape-predicate runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/introspection/shape_predicate.rs") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Scalar, vector, empty, trailing-dimension, and resident row geometry", location: "crates/runmat-runtime/src/builtins/array/introspection/isrow/tests.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Validated distributed global-shape behavior", location: "crates/runmat-runtime/src/builtins/array/introspection/shape_predicate.rs::tests::distributed_values_use_validated_global_shape_without_materialization" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::WgpuTest, label: "Actual WGPU row-shape metadata", location: "crates/runmat-runtime/src/builtins/array/introspection/isrow/tests.rs::wgpu_reads_shape_without_payload_transfer" },
    ],
    notes: &[],
};
pub(super) const ISROW_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("isrow"),
    slug: Some("isrow"),
    summary: "Determine whether an array has one visible row.",
    description: "`isrow` returns one host logical scalar from MATLAB-visible shape metadata.",
    keywords: &[
        "isrow",
        "row",
        "row vector",
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
