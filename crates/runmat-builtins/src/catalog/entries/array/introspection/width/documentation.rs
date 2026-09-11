use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Variables in tables and columns in arrays",
        paragraphs: &[
            "`width(A)` returns a table's variable count or an ordinary array's second MATLAB-visible dimension as a host double scalar. For ordinary arrays, `width(A)` is equivalent to `size(A, 2)`.",
            "Each table variable counts once even when that variable stores a matrix with several columns. Numeric, complex, logical, character, string, cell, datetime, duration, categorical, sparse, and object arrays use their outer column count.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Structural metadata",
        paragraphs: &[
            "All eight fixed-width integer classes use the same structural path. Their payloads are not converted or inspected. Empty arrays preserve their second-dimension extent.",
            "gpuArray and distributed inputs use validated handle metadata. `width` launches no kernel, calls no acceleration provider, and transfers no payload. The result is returned on the host at a fusion boundary.",
            "RunMat reports an error if the variable or column count cannot be represented exactly by the documented double result instead of rounding the structural value.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "matrix",
        title: "Count array columns",
        program: "A = zeros(3, 5, 'uint16');\nn = width(A)",
        display_output: Some("n = 5"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(n == 5);\nassert(isa(n, 'double'));" },
    },
    BuiltinExample {
        id: "table-variables",
        title: "Count table variables",
        program: "samples = [1 2; 3 4; 5 6];\nlabel = [10; 20; 30];\nT = table(samples, label);\nn = width(T)",
        display_output: Some("n = 2"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(n == 2);\nassert(size(samples, 2) == 2);" },
    },
    BuiltinExample {
        id: "empty",
        title: "Preserve an empty array's column count",
        program: "A = zeros(0, 7);\nn = width(A)",
        display_output: Some("n = 7"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(n == 7);" },
    },
    BuiltinExample {
        id: "gpu-shape",
        title: "Read a resident column count",
        program: "G = gpuArray(ones(4, 256));\nn = width(G)",
        display_output: Some("n = 256"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(n == 256);\nassert(~isgpuarray(n));" },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "How is `width` related to `size`?", answer: "For ordinary arrays, `width(A)` returns the same count as `size(A, 2)`. For tables and timetables, it returns the number of variables." },
    BuiltinDocumentationFaq { question: "How is a matrix-valued table variable counted?", answer: "It counts as one table variable. The columns inside that variable do not increase table width." },
    BuiltinDocumentationFaq { question: "Does `width` gather a gpuArray?", answer: "No. RunMat reads the second dimension from resident shape metadata." },
    BuiltinDocumentationFaq { question: "Where does the result live?", answer: "The result is a host double scalar because it describes shape metadata rather than array payload." },
];

const RELATED: &[&str] = &["height", "size", "length", "numel", "table", "gpuArray"];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "height", target: BuiltinDocumentationLinkTarget::Builtin("height") },
    BuiltinDocumentationLink { label: "size", target: BuiltinDocumentationLinkTarget::Builtin("size") },
    BuiltinDocumentationLink { label: "table", target: BuiltinDocumentationLinkTarget::Builtin("table") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/introspection/width.rs") },
];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Shared dimension metadata", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/introspection/dimension_metadata.rs") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Array, integer, table-variable, empty, resident, and output-count behavior", location: "crates/runmat-runtime/src/builtins/array/introspection/width/tests.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Validated distributed global-shape behavior", location: "crates/runmat-runtime/src/builtins/array/introspection/shape_scalar_query.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::WgpuTest, label: "Assertion-backed resident metadata example", location: "width::gpu-shape" },
    ],
    notes: &[],
};

pub(super) const WIDTH_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("width"),
    slug: Some("width"),
    summary: "Return the number of table variables or array columns.",
    description: "`width` reads a table variable count or the second visible array dimension from validated metadata and returns an exact host double scalar.",
    keywords: &["width", "columns", "variables", "table", "array shape", "gpu metadata", "distributed"],
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
