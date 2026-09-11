use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Element count", paragraphs: &["`isempty(A)` returns one logical scalar and is true when any visible dimension is zero. It checks the container shape and does not inspect cell contents or stored sparse values.", "Numeric, logical, character, string, cell, struct, object, sparse, and complex arrays use their MATLAB-visible dimensions. Ordinary scalar values and function handles occupy one element and return false."] },
    BuiltinDocumentationSection { heading: "Text and empty containers", paragraphs: &["An empty character array has zero character elements and returns true. A string scalar is a 1-by-1 container even when its text is empty, so `isempty(\"\")` is false. A zero-sized string array returns true.", "A scalar struct or object is not empty; a zero-sized struct or object array is empty. Cell arrays are classified from their outer dimensions, regardless of the values stored in their cells."] },
    BuiltinDocumentationSection { heading: "Resident and distributed values", paragraphs: &["gpuArray and distributed inputs are classified from validated shape metadata without launching a kernel, allocating device memory, or reading payload data. The result is always a host logical scalar and forms a fusion boundary."] },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "numeric",
        title: "Check an empty numeric matrix",
        program: "A = zeros(0, 3);\ntf = isempty(A)",
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
        id: "cell",
        title: "Check an empty cell array",
        program: "C = cell(0, 4);\ntf = isempty(C)",
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
        id: "text",
        title: "Distinguish character and string containers",
        program: "tf_chars = isempty('');\ntf_string = isempty(\"\")",
        display_output: Some("tf_chars = true and tf_string = false"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(tf_chars);\nassert(~tf_string);",
        },
    },
    BuiltinExample {
        id: "scalar",
        title: "Check a scalar",
        program: "tf = isempty(42)",
        display_output: Some("tf = false"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(~tf);",
        },
    },
    BuiltinExample {
        id: "string-array",
        title: "Check an empty string array",
        program: "S = strings(0, 2);\ntf = isempty(S)",
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
        id: "gpu-shape",
        title: "Inspect a resident empty shape",
        program: "G = gpuArray(zeros(5, 0));\ntf = isempty(G)",
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
const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Does `isempty` gather gpuArray data?", answer: "No. A gpuArray handle always carries its shape, so the predicate reads metadata only." },
    BuiltinDocumentationFaq { question: "Why is `isempty(\"\")` false but `isempty('')` true?", answer: "The string value is a 1-by-1 string container; the character literal contains zero character elements." },
    BuiltinDocumentationFaq { question: "Does `isempty` look inside cells?", answer: "No. It checks only the cell array's outer dimensions." },
    BuiltinDocumentationFaq { question: "How are structs and objects handled?", answer: "Their array dimensions determine the result. Scalar instances are not empty; zero-sized arrays are empty." },
];
const RELATED: &[&str] = &[
    "numel", "size", "length", "ndims", "isscalar", "isvector", "ismatrix", "gpuArray",
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "numel", target: BuiltinDocumentationLinkTarget::Builtin("numel") },
    BuiltinDocumentationLink { label: "size", target: BuiltinDocumentationLinkTarget::Builtin("size") },
    BuiltinDocumentationLink { label: "length", target: BuiltinDocumentationLinkTarget::Builtin("length") },
    BuiltinDocumentationLink { label: "ndims", target: BuiltinDocumentationLinkTarget::Builtin("ndims") },
    BuiltinDocumentationLink { label: "isscalar", target: BuiltinDocumentationLinkTarget::Builtin("isscalar") },
    BuiltinDocumentationLink { label: "isvector", target: BuiltinDocumentationLinkTarget::Builtin("isvector") },
    BuiltinDocumentationLink { label: "ismatrix", target: BuiltinDocumentationLinkTarget::Builtin("ismatrix") },
    BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") },
    BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Builtin("gather") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/introspection/isempty.rs") },
];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Shape-predicate runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/introspection/shape_predicate.rs") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Numeric, text, cell, object, resident, and empty-shape behavior", location: "crates/runmat-runtime/src/builtins/array/introspection/isempty/tests.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Validated distributed global-shape behavior", location: "crates/runmat-runtime/src/builtins/array/introspection/shape_predicate.rs::tests::distributed_values_use_validated_global_shape_without_materialization" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::WgpuTest, label: "Actual WGPU shape-only classification", location: "crates/runmat-runtime/src/builtins/array/introspection/isempty/tests.rs::wgpu_reads_shape_without_payload_transfer" },
    ],
    notes: &[],
};
pub(super) const ISEMPTY_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation { authority: BuiltinDocumentationAuthority::Catalog, title: Some("isempty"), slug: Some("isempty"), summary: "Determine whether an array contains zero elements.", description: "`isempty` classifies a value from its MATLAB-visible dimensions and returns one host logical scalar.", keywords: &["isempty", "empty array", "shape", "metadata", "metadata query", "gpu", "gpuArray", "logical", "distributed"], related: RELATED, sections: SECTIONS, examples: EXAMPLES, example_exemption: None, faqs: FAQS, links: LINKS, media: &[], evidence: EVIDENCE, introduced: None, status: Some(BuiltinDocumentationStatus::Stable) };
