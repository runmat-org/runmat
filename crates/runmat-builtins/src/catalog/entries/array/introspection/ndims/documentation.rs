use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Visible rank",
        paragraphs: &[
            "`ndims(A)` returns the MATLAB-visible rank of `A` as a double scalar. The result is at least 2, so scalars, row and column vectors, and matrices all return 2.",
            "Trailing singleton dimensions do not increase visible rank. A 2-by-3-by-1-by-1 array returns 2; a 2-by-3-by-1-by-4 array returns 4 because the fourth dimension is nonsingleton.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Value classes and residency",
        paragraphs: &[
            "Numeric, logical, character, string, cell, struct, object, sparse, and complex arrays all use their outer array dimensions. The stored element class and values do not affect the result.",
            "gpuArray and distributed inputs use validated shape metadata. `ndims` does not launch a kernel, call a provider, allocate device storage, or transfer payload data. It returns one host double scalar and forms a fusion boundary.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "scalar",
        title: "Inspect a scalar",
        program: "n = ndims(42)",
        display_output: Some("n = 2"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(n == 2);",
        },
    },
    BuiltinExample {
        id: "matrix",
        title: "Inspect a matrix",
        program: "A = rand(5, 3);\nn = ndims(A)",
        display_output: Some("n = 2"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(n == 2);",
        },
    },
    BuiltinExample {
        id: "three-dimensional",
        title: "Inspect a three-dimensional array",
        program: "A = rand(4, 5, 6);\nn = ndims(A)",
        display_output: Some("n = 3"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(n == 3);",
        },
    },
    BuiltinExample {
        id: "trailing-singletons",
        title: "Ignore trailing singleton dimensions",
        program: "A = ones(2, 3, 1, 1);\nn = ndims(A)",
        display_output: Some("n = 2"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(n == 2);",
        },
    },
    BuiltinExample {
        id: "cell-array",
        title: "Inspect a cell array",
        program: "C = {1, 2, 3; 4, 5, 6};\nn = ndims(C)",
        display_output: Some("n = 2"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(n == 2);",
        },
    },
    BuiltinExample {
        id: "string-array",
        title: "Inspect a string array",
        program: "S = [\"alpha\"; \"beta\"; \"gamma\"];\nn = ndims(S)",
        display_output: Some("n = 2"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(n == 2);",
        },
    },
    BuiltinExample {
        id: "gpu-shape",
        title: "Inspect a resident N-D shape",
        program: "G = gpuArray(ones(16, 32, 2));\nn = ndims(G)",
        display_output: Some("n = 3"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(n == 3);\nassert(~isgpuarray(n));",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Why do scalars and vectors return 2?", answer: "Their MATLAB-visible shapes are 1-by-1, 1-by-N, or N-by-1, and visible rank has a minimum of two." },
    BuiltinDocumentationFaq { question: "Are trailing singleton dimensions counted?", answer: "No. They are ignored. A later nonsingleton dimension remains visible and determines the rank." },
    BuiltinDocumentationFaq { question: "Does `ndims` gather gpuArray data?", answer: "No. The resident handle carries the required shape metadata." },
    BuiltinDocumentationFaq { question: "Can `ndims` inspect cells, structs, and objects?", answer: "Yes. It uses their outer MATLAB-visible array dimensions and does not inspect contents." },
    BuiltinDocumentationFaq { question: "Do element classes change the result?", answer: "No. Floating, integer, logical, complex, character, and string arrays all use the same shape rule." },
    BuiltinDocumentationFaq { question: "Can `ndims` appear beside fused GPU work?", answer: "Yes. It reads metadata and returns a host scalar at a fusion boundary without moving the input payload." },
];

const RELATED: &[&str] = &[
    "size", "length", "numel", "isempty", "ismatrix", "isvector", "gpuArray",
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "size", target: BuiltinDocumentationLinkTarget::Builtin("size") },
    BuiltinDocumentationLink { label: "length", target: BuiltinDocumentationLinkTarget::Builtin("length") },
    BuiltinDocumentationLink { label: "numel", target: BuiltinDocumentationLinkTarget::Builtin("numel") },
    BuiltinDocumentationLink { label: "isempty", target: BuiltinDocumentationLinkTarget::Builtin("isempty") },
    BuiltinDocumentationLink { label: "ismatrix", target: BuiltinDocumentationLinkTarget::Builtin("ismatrix") },
    BuiltinDocumentationLink { label: "isscalar", target: BuiltinDocumentationLinkTarget::Builtin("isscalar") },
    BuiltinDocumentationLink { label: "isvector", target: BuiltinDocumentationLinkTarget::Builtin("isvector") },
    BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/introspection/ndims.rs") },
];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Shape scalar-query runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/introspection/shape_scalar_query.rs") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Scalar, N-D, trailing-singleton, integer, text, container, and resident behavior", location: "crates/runmat-runtime/src/builtins/array/introspection/ndims/tests.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Validated distributed global-shape behavior", location: "crates/runmat-runtime/src/builtins/array/introspection/shape_scalar_query.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::WgpuTest, label: "Actual WGPU metadata-only rank", location: "crates/runmat-runtime/src/builtins/array/introspection/ndims/tests.rs::wgpu_reads_shape_without_payload_transfer" },
    ],
    notes: &[],
};

pub(super) const NDIMS_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("ndims"),
    slug: Some("ndims"),
    summary: "Return the MATLAB-visible rank of an array.",
    description: "`ndims` reads shape metadata, ignores trailing singleton dimensions, and returns a host double scalar of at least two.",
    keywords: &["ndims", "number of dimensions", "array rank", "gpu metadata", "MATLAB compatibility", "distributed"],
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
