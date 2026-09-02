use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Dimensions and empty reductions",
        paragraphs: &[
            "`all(A)` tests whether every element is nonzero along the first nonsingleton dimension. `all(A, dim)` selects one positive dimension, `all(A, vecdim)` reduces a vector of dimensions, and `all(A, \"all\")` reduces the complete array to one logical scalar.",
            "Reduced dimensions remain present with extent one. Dimensions beyond the input rank leave the value unchanged. An empty reduction uses the logical identity `true`; consequently `all(A, \"all\")` is true for an empty array.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Values and NaN policy",
        paragraphs: &[
            "Numeric, logical, complex, and character arrays are supported. A complex element is nonzero when either component is nonzero. Fixed-width integers are compared with zero in native storage, so wide values are not converted through double.",
            "The default behavior treats NaN as nonzero. In RunMat compatibility mode, `\"omitnan\"` and `\"includenan\"` may be supplied before or after the dimension selector. MATLAB compatibility mode rejects those explicit policy forms.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "GPU execution",
        paragraphs: &[
            "An acceleration provider may execute whole-array or dimension-wise AND reductions. Unsupported hooks fall back to a host reduction after gathering the input. The compact logical result is host-resident in either case.",
            "The fusion planner may combine compatible floating-point reductions with preceding work. Explicit `gpuArray` inputs remain valid, but callers do not need to move arrays manually for automatic acceleration.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "columns",
        title: "Test every column",
        program: "A = [1 2 3; 4 5 0];\ntf = all(A)",
        display_output: Some("tf = [1 1 0]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, logical([1 1 0])));",
        },
    },
    BuiltinExample {
        id: "rows",
        title: "Test every row",
        program: "A = [1 0 3; 4 5 6; 0 7 8];\ntf = all(A, 2)",
        display_output: Some("tf = [0; 1; 0]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, logical([0; 1; 0])));",
        },
    },
    BuiltinExample {
        id: "vecdim",
        title: "Reduce two dimensions",
        program: "A = reshape(1:24, [3 4 2]);\ntf = reshape(all(A > 0, [1 2]), 1, 2)",
        display_output: Some("tf = [1 1]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, logical([1 1])));",
        },
    },
    BuiltinExample {
        id: "all-elements",
        title: "Reduce every element",
        program: "A = [2 4; 6 8];\ntf = all(A, \"all\")",
        display_output: Some("tf = true"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isscalar(tf));\nassert(tf);",
        },
    },
    BuiltinExample {
        id: "nan-default",
        title: "Treat NaN as nonzero by default",
        program: "A = [NaN 1 2; NaN 0 3];\ntf = all(A)",
        display_output: Some("tf = [1 0 1]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, logical([1 0 1])));",
        },
    },
    BuiltinExample {
        id: "character",
        title: "Test character code points",
        program: "A = ['a' char(0) 'c'];\ntf = all(A, 1)",
        display_output: Some("tf = [1 0 1]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, logical([1 0 1])));",
        },
    },
    BuiltinExample {
        id: "gpu",
        title: "Reduce resident rows",
        program: "G = gpuArray([1 1 1; 1 1 0]);\ntf = all(G, 2)",
        display_output: Some("tf = [1; 0]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, logical([1; 0])));\nassert(~isgpuarray(tf));",
        },
    },
    BuiltinExample {
        id: "omitnan",
        title: "Select an explicit NaN policy",
        program: "A = [NaN 0; NaN 2];\ntf = all(A, \"omitnan\")",
        display_output: Some("tf = [1 0]"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Native,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, logical([1 0])));",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "What does `all(A)` test?", answer: "It tests every slice along the first nonsingleton dimension and returns a logical result for each slice." },
    BuiltinDocumentationFaq { question: "How do I reduce a particular dimension?", answer: "Pass a positive scalar dimension, or a vector such as `[1 3]` to reduce several dimensions." },
    BuiltinDocumentationFaq { question: "What does the `\"all\"` selector do?", answer: "It reduces every input dimension and returns one logical scalar." },
    BuiltinDocumentationFaq { question: "How are NaN values handled?", answer: "NaN counts as nonzero by default. RunMat mode also provides explicit `\"omitnan\"` and `\"includenan\"` forms." },
    BuiltinDocumentationFaq { question: "Does `all` support complex values?", answer: "Yes. A complex value is nonzero when either its real or imaginary component is nonzero." },
    BuiltinDocumentationFaq { question: "What happens for empty input?", answer: "Each empty reduction returns the AND identity, logical true, with the corresponding reduced shape." },
    BuiltinDocumentationFaq { question: "Does `all` preserve integer exactness?", answer: "Yes. Fixed-width integer elements are tested directly in native storage." },
    BuiltinDocumentationFaq { question: "Where does a GPU result live?", answer: "The logical result is returned on the host after provider execution or an exact host fallback." },
];

const RELATED: &[&str] = &[
    "any", "prod", "sum", "gpuArray", "gather", "cummax", "cummin", "cumprod", "cumsum", "diff",
    "max", "mean", "median", "min", "nnz", "std", "var",
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "any", target: BuiltinDocumentationLinkTarget::Builtin("any") },
    BuiltinDocumentationLink { label: "sum", target: BuiltinDocumentationLinkTarget::Builtin("sum") },
    BuiltinDocumentationLink { label: "prod", target: BuiltinDocumentationLinkTarget::Builtin("prod") },
    BuiltinDocumentationLink { label: "gpuArray", target: BuiltinDocumentationLinkTarget::Builtin("gpuArray") },
    BuiltinDocumentationLink { label: "gather", target: BuiltinDocumentationLinkTarget::Builtin("gather") },
    BuiltinDocumentationLink { label: "cummax", target: BuiltinDocumentationLinkTarget::Builtin("cummax") },
    BuiltinDocumentationLink { label: "cummin", target: BuiltinDocumentationLinkTarget::Builtin("cummin") },
    BuiltinDocumentationLink { label: "cumprod", target: BuiltinDocumentationLinkTarget::Builtin("cumprod") },
    BuiltinDocumentationLink { label: "cumsum", target: BuiltinDocumentationLinkTarget::Builtin("cumsum") },
    BuiltinDocumentationLink { label: "diff", target: BuiltinDocumentationLinkTarget::Builtin("diff") },
    BuiltinDocumentationLink { label: "max", target: BuiltinDocumentationLinkTarget::Builtin("max") },
    BuiltinDocumentationLink { label: "mean", target: BuiltinDocumentationLinkTarget::Builtin("mean") },
    BuiltinDocumentationLink { label: "median", target: BuiltinDocumentationLinkTarget::Builtin("median") },
    BuiltinDocumentationLink { label: "min", target: BuiltinDocumentationLinkTarget::Builtin("min") },
    BuiltinDocumentationLink { label: "nnz", target: BuiltinDocumentationLinkTarget::Builtin("nnz") },
    BuiltinDocumentationLink { label: "std", target: BuiltinDocumentationLinkTarget::Builtin("std") },
    BuiltinDocumentationLink { label: "var", target: BuiltinDocumentationLinkTarget::Builtin("var") },
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/reduction/all.rs") },
];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Shared logical reduction boundary", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/reduction/logical/mod.rs") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Host values, dimensions, NaN policies, errors, and typed integers", location: "crates/runmat-runtime/src/builtins/math/reduction/all/tests.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::WgpuTest, label: "Provider and fallback behavior", location: "all::gpu" },
    ],
    notes: &[],
};

pub(super) const ALL_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("all"),
    slug: Some("all"),
    summary: "Test whether every element is nonzero along selected dimensions.",
    description: "`all` performs a logical AND reduction over the first nonsingleton dimension, selected dimensions, or the complete array.",
    keywords: &[
        "all",
        "logical reduction",
        "dimension",
        "vecdim",
        "omitnan",
        "gpu",
        "vectorization",
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
