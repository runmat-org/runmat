use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Dimension queries",
        paragraphs: &[
            "`size(A)` returns the complete MATLAB-visible shape as a 1-by-N host double row vector. Scalars and string scalars have shape 1-by-1; character vectors retain their character-array shape; cells, structs, object arrays, tables, and timetables report their outer array dimensions.",
            "`size(A, dim)` returns one extent. A vector selects several dimensions, an empty vector returns a 1-by-0 row, and separate scalar arguments select several dimensions. A selector beyond the visible rank contributes 1. The vector and separate-argument forms are available in the compatible language surface introduced in R2019b.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Multiple outputs",
        paragraphs: &[
            "With no explicit dimensions, multiple outputs receive leading dimensions separately. If fewer outputs than visible dimensions are requested, the final output is the checked product of every remaining dimension. Extra outputs are padded with 1.",
            "With explicit queried dimensions, the number of outputs must equal the number of queried dimensions. Every output is a host double scalar.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Exact structural values",
        paragraphs: &[
            "All fixed-width integer selector classes are decoded from native storage without conversion through double. Extents, selected dimensions, and collapsed products use checked structural arithmetic. RunMat reports an error instead of rounding a structural result that cannot be represented exactly by the documented double output.",
            "gpuArray and distributed inputs are inspected through validated handle metadata. `size` launches no kernel, calls no acceleration provider, transfers no payload, and returns host values at a fusion boundary.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "matrix",
        title: "Read a matrix shape",
        program: "A = [1 2 3; 4 5 6];\nsz = size(A)",
        display_output: Some("sz = [2 3]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(sz, [2 3]));",
        },
    },
    BuiltinExample {
        id: "single-dimension",
        title: "Read one dimension",
        program: "A = randn(8, 4);\nrows = size(A, 1)",
        display_output: Some("rows = 8"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(rows == 8);",
        },
    },
    BuiltinExample {
        id: "dimension-vector",
        title: "Read selected dimensions",
        program: "A = zeros(5, 4, 3);\nselected = size(A, [1 3])",
        display_output: Some("selected = [5 3]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(selected, [5 3]));",
        },
    },
    BuiltinExample {
        id: "dimension-list",
        title: "Use separate dimension selectors",
        program: "A = zeros(5, 4, 3);\nselected = size(A, 2, 3)",
        display_output: Some("selected = [4 3]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(selected, [4 3]));",
        },
    },
    BuiltinExample {
        id: "collapsed-output",
        title: "Collapse trailing dimensions into the final output",
        program: "A = zeros(3, 4, 5);\n[first, remainder] = size(A)",
        display_output: Some("first = 3; remainder = 20"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(first == 3);\nassert(remainder == 20);",
        },
    },
    BuiltinExample {
        id: "empty-query",
        title: "Return an empty dimension query",
        program: "A = ones(2, 3);\nsz = size(A, [])",
        display_output: Some("sz is a 1-by-0 double array"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(sz), [1 0]));\nassert(isa(sz, 'double'));",
        },
    },
    BuiltinExample {
        id: "cell",
        title: "Inspect a cell array",
        program: "C = {1, 2, 3; 4, 5, 6};\nsz = size(C)",
        display_output: Some("sz = [2 3]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(sz, [2 3]));",
        },
    },
    BuiltinExample {
        id: "gpu-shape",
        title: "Inspect a resident array without gathering",
        program: "G = gpuArray(ones(256, 512));\nsz = size(G)",
        display_output: Some("sz = [256 512]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(sz, [256 512]));\nassert(~isgpuarray(sz));",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "How is `size` different from `length` and `numel`?", answer: "`size` exposes individual dimensions, `length` returns the largest dimension, and `numel` returns the product of all dimensions." },
    BuiltinDocumentationFaq { question: "What happens when a selector exceeds the array rank?", answer: "The corresponding result is 1." },
    BuiltinDocumentationFaq { question: "How are row and column vectors reported?", answer: "A row vector is 1-by-N and a column vector is N-by-1." },
    BuiltinDocumentationFaq { question: "What happens for empty arrays?", answer: "Zero extents remain present in the returned shape and participate in collapsed products." },
    BuiltinDocumentationFaq { question: "Can dimensions be queried as a vector or list?", answer: "Yes. A vector or separate positive integer scalar selectors return the corresponding extents in order." },
    BuiltinDocumentationFaq { question: "Does `size` gather gpuArray or distributed data?", answer: "No. Complete shape metadata is part of each validated handle." },
    BuiltinDocumentationFaq { question: "Where does the result live?", answer: "All `size` outputs are host double values because they describe metadata, not array payload." },
];

const RELATED: &[&str] = &[
    "length", "numel", "ndims", "isempty", "ismatrix", "isscalar", "isvector", "gpuArray",
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "length", target: BuiltinDocumentationLinkTarget::Builtin("length") },
    BuiltinDocumentationLink { label: "numel", target: BuiltinDocumentationLinkTarget::Builtin("numel") },
    BuiltinDocumentationLink { label: "ndims", target: BuiltinDocumentationLinkTarget::Builtin("ndims") },
    BuiltinDocumentationLink { label: "isempty", target: BuiltinDocumentationLinkTarget::Builtin("isempty") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/introspection/size.rs") },
];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Shared shape-query primitives", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/tree/main/crates/runmat-runtime/src/builtins/array/introspection/shape_query") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Full, selected, variadic, collapsed-output, empty, container, table, and resident semantics", location: "crates/runmat-runtime/src/builtins/array/introspection/size.rs::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed inference for vector and multiple-output forms", location: "crates/runmat-builtins/src/catalog/inference/array/introspection/shape_query.rs::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::WgpuTest, label: "Assertion-backed resident metadata example", location: "size::gpu-shape" },
    ],
    notes: &[],
};

pub(super) const SIZE_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("size"), slug: Some("size"),
    summary: "Return array dimension sizes in vector or multiple-output forms.",
    description: "`size` reads outer array shape metadata, supports scalar, vector, and separate dimension queries, and implements requested-output collapse rules.",
    keywords: &["size", "dimensions", "shape", "multiple outputs", "gpu metadata", "distributed"],
    related: RELATED, sections: SECTIONS, examples: EXAMPLES, example_exemption: None, faqs: FAQS, links: LINKS, media: &[], evidence: EVIDENCE,
    introduced: Some("Before R2006a"), status: Some(BuiltinDocumentationStatus::Stable),
};
