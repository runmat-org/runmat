use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Rows in arrays and tabular values",
        paragraphs: &[
            "`height(A)` returns the first MATLAB-visible dimension as a host double scalar. For a table or timetable, that dimension is the number of rows. For an ordinary array, `height(A)` is equivalent to `size(A, 1)`.",
            "Numeric, complex, logical, character, string, cell, datetime, duration, categorical, sparse, and object arrays use their outer shape. The element class and values do not affect the result.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Structural metadata",
        paragraphs: &[
            "All eight fixed-width integer classes use the same structural path. Integer payloads are not converted or inspected. Table variables may contain several columns; their row counts still determine table height.",
            "gpuArray and distributed inputs use validated handle metadata. `height` launches no kernel, calls no acceleration provider, and transfers no payload. The result is returned on the host at a fusion boundary.",
            "RunMat reports an error if the row count cannot be represented exactly by the documented double result instead of rounding the structural value.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "matrix",
        title: "Count matrix rows",
        program: "A = zeros(5, 3, 'uint64');\nn = height(A)",
        display_output: Some("n = 5"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(n == 5);\nassert(isa(n, 'double'));",
        },
    },
    BuiltinExample {
        id: "table",
        title: "Count table rows",
        program: "id = [101; 102; 103];\nscore = [8; 5; 9];\nT = table(id, score);\nn = height(T)",
        display_output: Some("n = 3"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(n == 3);",
        },
    },
    BuiltinExample {
        id: "empty",
        title: "Inspect an empty rectangular array",
        program: "A = zeros(0, 7);\nn = height(A)",
        display_output: Some("n = 0"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(n == 0);",
        },
    },
    BuiltinExample {
        id: "gpu-shape",
        title: "Read a resident row count",
        program: "G = gpuArray(ones(128, 4));\nn = height(G)",
        display_output: Some("n = 128"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(n == 128);\nassert(~isgpuarray(n));",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "How is `height` related to `size`?", answer: "For arrays, tables, and timetables, `height(A)` returns the same count as `size(A, 1)`." },
    BuiltinDocumentationFaq { question: "Does `height` inspect table variables?", answer: "It validates tabular metadata and returns the shared row count. It does not inspect the values stored in each row." },
    BuiltinDocumentationFaq { question: "Does `height` gather a gpuArray?", answer: "No. The resident handle carries the first-dimension extent." },
    BuiltinDocumentationFaq { question: "Where does the result live?", answer: "The result is a host double scalar because it describes shape metadata rather than array payload." },
];

const RELATED: &[&str] = &["width", "size", "length", "numel", "head", "gpuArray"];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "width", target: BuiltinDocumentationLinkTarget::Builtin("width") },
    BuiltinDocumentationLink { label: "size", target: BuiltinDocumentationLinkTarget::Builtin("size") },
    BuiltinDocumentationLink { label: "head", target: BuiltinDocumentationLinkTarget::Builtin("head") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/introspection/height.rs") },
];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Shared dimension metadata", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/introspection/dimension_metadata.rs") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Array, integer, table, empty, resident, and output-count behavior", location: "crates/runmat-runtime/src/builtins/array/introspection/height/tests.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Validated distributed global-shape behavior", location: "crates/runmat-runtime/src/builtins/array/introspection/shape_scalar_query.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::WgpuTest, label: "Assertion-backed resident metadata example", location: "height::gpu-shape" },
    ],
    notes: &[],
};

pub(super) const HEIGHT_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("height"),
    slug: Some("height"),
    summary: "Return the number of table rows or array rows.",
    description: "`height` reads the first visible dimension from array, table, timetable, resident, or distributed shape metadata and returns an exact host double scalar.",
    keywords: &["height", "rows", "table", "timetable", "array shape", "gpu metadata", "distributed"],
    related: RELATED,
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: EVIDENCE,
    introduced: Some("R2013b"),
    status: Some(BuiltinDocumentationStatus::Stable),
};
