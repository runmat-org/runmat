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
            "`any(A)` tests whether at least one element is nonzero along the first nonsingleton dimension. `any(A, dim)` selects one positive dimension, `any(A, vecdim)` reduces a vector of dimensions, and `any(A, \"all\")` reduces the complete array to one logical scalar.",
            "Reduced dimensions remain present with extent one. Dimensions beyond the input rank leave the value unchanged. An empty reduction uses the logical identity `false`; consequently `any(A, \"all\")` is false for an empty array.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Values and NaN policy",
        paragraphs: &[
            "Numeric, logical, complex, and character arrays are supported. A complex element is nonzero when either component is nonzero. Fixed-width integers are compared with zero in native storage, so wide values are not converted through double.",
            "The default behavior ignores NaN. In RunMat compatibility mode, `\"omitnan\"` and `\"includenan\"` may be supplied before or after the dimension selector. MATLAB compatibility mode rejects those explicit policy forms.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "GPU execution",
        paragraphs: &[
            "An acceleration provider may execute whole-array or dimension-wise OR reductions. Unsupported hooks fall back to a host reduction after gathering the input. The compact logical result is host-resident in either case.",
            "The fusion planner may combine compatible floating-point reductions with preceding work. Explicit `gpuArray` inputs remain valid, but callers do not need to move arrays manually for automatic acceleration.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "columns",
        title: "Find nonzero columns",
        program: "A = [0 2 0; 0 0 0];\ntf = any(A)",
        display_output: Some("tf = [0 1 0]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, logical([0 1 0])));",
        },
    },
    BuiltinExample {
        id: "rows",
        title: "Find nonzero rows",
        program: "A = [0 4 0; 1 0 0; 0 0 0];\ntf = any(A, 2)",
        display_output: Some("tf = [1; 1; 0]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, logical([1; 1; 0])));",
        },
    },
    BuiltinExample {
        id: "vecdim",
        title: "Reduce two dimensions",
        program: "A = reshape(1:24, [3 4 2]);\ntf = reshape(any(A > 20, [1 2]), 1, 2)",
        display_output: Some("tf = [0 1]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, logical([0 1])));",
        },
    },
    BuiltinExample {
        id: "all-elements",
        title: "Reduce every element",
        program: "A = [0 0; 0 5];\ntf = any(A, \"all\")",
        display_output: Some("tf = true"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isscalar(tf));\nassert(tf);",
        },
    },
    BuiltinExample {
        id: "nan-default",
        title: "Ignore NaN by default",
        program: "A = [NaN 0 0; 0 0 0];\ntf = any(A)",
        display_output: Some("tf = [0 0 0]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, logical([0 0 0])));",
        },
    },
    BuiltinExample {
        id: "character",
        title: "Test character code points",
        program: "A = ['a' char(0) 'c'];\ntf = any(A, 1)",
        display_output: Some("tf = [1 0 1]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, logical([1 0 1])));",
        },
    },
    BuiltinExample {
        id: "gpu",
        title: "Reduce resident rows",
        program: "G = gpuArray([0 1 0; 0 0 0]);\ntf = any(G, 2)",
        display_output: Some("tf = [1; 0]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, logical([1; 0])));\nassert(~isgpuarray(tf));",
        },
    },
    BuiltinExample {
        id: "includenan",
        title: "Count NaN as nonzero",
        program: "A = [NaN 0; 0 0];\ntf = any(A, \"includenan\")",
        display_output: Some("tf = [1 0]"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Native,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf, logical([1 0])));",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "What does `any(A)` test?", answer: "It tests each slice along the first nonsingleton dimension and reports whether that slice contains a nonzero element." },
    BuiltinDocumentationFaq { question: "How do I reduce a particular dimension?", answer: "Pass a positive scalar dimension, or a vector such as `[1 3]` to reduce several dimensions." },
    BuiltinDocumentationFaq { question: "What does the `\"all\"` selector do?", answer: "It reduces every input dimension and returns one logical scalar." },
    BuiltinDocumentationFaq { question: "How are NaN values handled?", answer: "NaN is ignored by default. RunMat mode also provides explicit `\"omitnan\"` and `\"includenan\"` forms." },
    BuiltinDocumentationFaq { question: "Does `any` support complex values?", answer: "Yes. A complex value is nonzero when either its real or imaginary component is nonzero." },
    BuiltinDocumentationFaq { question: "What happens for empty input?", answer: "Each empty reduction returns the OR identity, logical false, with the corresponding reduced shape." },
    BuiltinDocumentationFaq { question: "Does `any` preserve integer exactness?", answer: "Yes. Fixed-width integer elements are tested directly in native storage." },
    BuiltinDocumentationFaq { question: "Where does a GPU result live?", answer: "The logical result is returned on the host after provider execution or an exact host fallback." },
];

const RELATED: &[&str] = &[
    "sum", "prod", "mean", "gpuArray", "gather", "all", "cummax", "cummin", "cumprod", "cumsum",
    "diff", "max", "median", "min", "nnz", "std", "var",
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink { label: "all", target: BuiltinDocumentationLinkTarget::Builtin("all") },
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
    BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/reduction/any.rs") },
];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Shared logical reduction boundary", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/reduction/logical/mod.rs") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Host values, dimensions, NaN policies, errors, and typed integers", location: "crates/runmat-runtime/src/builtins/math/reduction/any/tests.rs" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::WgpuTest, label: "Provider and fallback behavior", location: "any::gpu" },
    ],
    notes: &[],
};

pub(super) const ANY_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("any"),
    slug: Some("any"),
    summary: "Test whether at least one element is nonzero along selected dimensions.",
    description: "`any` performs a logical OR reduction over the first nonsingleton dimension, selected dimensions, or the complete array.",
    keywords: &[
        "any",
        "logical reduction",
        "dimension",
        "vecdim",
        "omitnan",
        "all",
        "gpu",
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
