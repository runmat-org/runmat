use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Largest dimension",
        paragraphs: &[
            "`length(A)` returns the largest extent in the MATLAB-visible shape of `A` as a double scalar. Scalars have length 1; row and column vectors return their populated dimension; matrices and N-D arrays return the largest extent across all dimensions.",
            "For empty arrays, zero extents participate like any other dimension. `zeros(0, 0)` has length 0, while `zeros(0, 7)` and `zeros(0, 0, 5)` have lengths 7 and 5.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Containers and text",
        paragraphs: &[
            "Character, string, cell, struct, object, sparse, complex, and logical arrays use their outer array dimensions. A string scalar such as `\"abc\"` has length 1; use `strlength` to count characters in string elements.",
            "A `containers.Map` handle reports its entry count. Tables and timetables are rejected because their conventional dimensions have dedicated `height`, `width`, and `size` queries.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Resident and distributed values",
        paragraphs: &[
            "gpuArray and distributed inputs are classified from validated shape metadata. `length` does not launch a kernel, call an acceleration provider, or transfer payload data. The result is a host double scalar and forms a fusion boundary.",
            "RunMat returns an error if a distributed extent is too large to be represented exactly by the documented double result instead of silently rounding the structural value.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "row",
        title: "Measure a row vector",
        program: "row = [1 2 3 4];\nn = length(row)",
        display_output: Some("n = 4"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(n == 4);",
        },
    },
    BuiltinExample {
        id: "matrix",
        title: "Find the longer matrix dimension",
        program: "A = randn(5, 12);\nn = length(A)",
        display_output: Some("n = 12"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(n == 12);",
        },
    },
    BuiltinExample {
        id: "empty",
        title: "Inspect an empty rectangular array",
        program: "E = zeros(0, 7);\nn = length(E)",
        display_output: Some("n = 7"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(n == 7);",
        },
    },
    BuiltinExample {
        id: "characters",
        title: "Measure a character row",
        program: "name = 'RunMat';\nn = length(name)",
        display_output: Some("n = 6"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(n == 6);",
        },
    },
    BuiltinExample {
        id: "string-scalar",
        title: "Distinguish string shape from text length",
        program: "n = length(\"RunMat\")",
        display_output: Some("n = 1"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(n == 1);",
        },
    },
    BuiltinExample {
        id: "n-dimensional",
        title: "Inspect every dimension of an N-D array",
        program: "A = zeros(2, 9, 4);\nn = length(A)",
        display_output: Some("n = 9"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(n == 9);",
        },
    },
    BuiltinExample {
        id: "gpu-shape",
        title: "Inspect a resident shape",
        program: "G = gpuArray(ones(256, 4));\nn = length(G)",
        display_output: Some("n = 256"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(n == 256);\nassert(~isgpuarray(n));",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "How is `length` different from `size`?", answer: "`length(A)` returns only the largest dimension as a scalar. `size(A)` exposes individual dimensions or the complete dimension vector." },
    BuiltinDocumentationFaq { question: "What does `length` return for a scalar?", answer: "A scalar has visible shape 1-by-1, so the result is 1." },
    BuiltinDocumentationFaq { question: "How are empty arrays handled?", answer: "The largest extent is returned. `zeros(0, 0)` has length 0, while `zeros(0, 5)` has length 5." },
    BuiltinDocumentationFaq { question: "Does `length` gather gpuArray data?", answer: "No. The resident handle carries the complete shape metadata needed by the query." },
    BuiltinDocumentationFaq { question: "Can `length` inspect cells, structs, and objects?", answer: "Yes. Their outer MATLAB-visible array dimensions determine the result; contents do not." },
    BuiltinDocumentationFaq { question: "Does `length` count characters or encoded bytes?", answer: "It reports character-array dimensions. For string scalars it reports the 1-by-1 container shape; use `strlength` to count text." },
    BuiltinDocumentationFaq { question: "Why are tables rejected?", answer: "Use `height`, `width`, or `size` to state which tabular dimension the program needs." },
    BuiltinDocumentationFaq { question: "Can `length` appear beside fused GPU work?", answer: "Yes. It reads metadata and returns a host scalar at a fusion boundary without moving the input payload." },
];

const RELATED: &[&str] = &[
    "size",
    "numel",
    "ndims",
    "isempty",
    "isvector",
    "strlength",
    "gpuArray",
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "size", target: BuiltinDocumentationLinkTarget::Builtin("size") },
    BuiltinDocumentationLink { label: "numel", target: BuiltinDocumentationLinkTarget::Builtin("numel") },
    BuiltinDocumentationLink { label: "ndims", target: BuiltinDocumentationLinkTarget::Builtin("ndims") },
    BuiltinDocumentationLink { label: "isempty", target: BuiltinDocumentationLinkTarget::Builtin("isempty") },
    BuiltinDocumentationLink { label: "ismatrix", target: BuiltinDocumentationLinkTarget::Builtin("ismatrix") },
    BuiltinDocumentationLink { label: "isscalar", target: BuiltinDocumentationLinkTarget::Builtin("isscalar") },
    BuiltinDocumentationLink { label: "isvector", target: BuiltinDocumentationLinkTarget::Builtin("isvector") },
    BuiltinDocumentationLink { label: "strlength", target: BuiltinDocumentationLinkTarget::Builtin("strlength") },
    BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/introspection/length.rs") },
];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Shape scalar-query runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/introspection/shape_scalar_query.rs") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Scalar, array, empty, text, container, table, and resident behavior", location: "crates/runmat-runtime/src/builtins/array/introspection/length/tests.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "containers.Map entry-count semantics", location: "crates/runmat-runtime/src/builtins/containers/map/containers.map.rs::tests::length_delegates_to_map_count" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Validated distributed global-shape and exact-result behavior", location: "crates/runmat-runtime/src/builtins/array/introspection/shape_scalar_query.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::WgpuTest, label: "Actual WGPU metadata-only length", location: "crates/runmat-runtime/src/builtins/array/introspection/length/tests.rs::wgpu_reads_shape_without_payload_transfer" },
    ],
    notes: &[],
};

pub(super) const LENGTH_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("length"),
    slug: Some("length"),
    summary: "Return the largest dimension of an array.",
    description: "`length` reads MATLAB-visible shape metadata and returns the largest extent as an exact host double scalar.",
    keywords: &["length", "largest dimension", "vector length", "gpu metadata", "array size", "distributed"],
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
