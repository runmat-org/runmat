use crate::*;
const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &["Operates along the first non-singleton dimension by default. Pass a dimension argument to override.", "Accepts the same direction keywords as `sort`: `'ascend'` (default) or `'descend'`.", "Supports `'ComparisonMethod'` values `'auto'`, `'real'`, and `'abs'` for real and complex inputs.", "Returns indices as double-precision tensors using MATLAB's one-based indexing.", "Treats NaN values as missing: they appear at the end for ascending permutations and at the beginning for descending permutations.", "Acts as a residency sink. GPU tensors are gathered when the active provider does not expose a specialised sort kernel."],
    },
    BuiltinDocumentationSection {
        heading: "Types, placement and limitations",
        paragraphs: &["argsort publishes only one-based double permutation indices. It currently uses the same evaluation as sort, including computing sorted values. Resident indices are restored to the input owner; a provider hook may stage work through host memory."],
    },
    BuiltinDocumentationSection {
        heading: "Resident execution",
        paragraphs: &["These operations are eager fusion boundaries, not elementwise kernels or reductions, and do not broadcast. Provider availability does not imply a device-native sorting or set kernel: current providers may stage host work. An unsupported input can use exact host fallback before provider work starts. Errors after an attempt, including cancellation, are returned without repeating the operation on the host. Supported resident outputs return to their owning provider; predicates and all-indices cells have the host placement described above. Distributed execution is not supported."],
    },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",

        title: "Getting indices that sort a vector",

        program: "A = [4; 1; 3];\nidx = argsort(A)",

        display_output: Some("idx =\n     2\n     3\n     1"),

        compatibility: BuiltinExampleCompatibility::RunMat,

        harness: BuiltinExampleHarness::Portable,

        fixture: BuiltinExampleFixture::None,

        requirements: BuiltinExampleRequirements::NONE,

        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(idx,[2;3;1]) && isa(idx,'double'));",
        },
    },
    BuiltinExample {
        id: "example-2",

        title: "Reordering data with the permutation indices",

        program: "A = [3 9 1 5];\nidx = argsort(A);\nsorted = A(idx)",

        display_output: Some("sorted =\n     1     3     5     9"),

        compatibility: BuiltinExampleCompatibility::RunMat,

        harness: BuiltinExampleHarness::Portable,

        fixture: BuiltinExampleFixture::None,

        requirements: BuiltinExampleRequirements::NONE,

        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(sorted,[1 3 5 9]));",
        },
    },
    BuiltinExample {
        id: "example-3",

        title: "Sorting along a specific dimension",

        program: "A = [1 6 4; 2 3 5];\nidx = argsort(A, 2)",

        display_output: Some("idx =\n     1     3     2\n     1     2     3"),

        compatibility: BuiltinExampleCompatibility::RunMat,

        harness: BuiltinExampleHarness::Portable,

        fixture: BuiltinExampleFixture::None,

        requirements: BuiltinExampleRequirements::NONE,

        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(idx,[1 3 2;1 2 3]));",
        },
    },
    BuiltinExample {
        id: "example-4",

        title: "Descending order permutations",

        program: "A = [10 4 7 9];\nidx = argsort(A, 'descend')",

        display_output: Some("idx =\n     1     4     3     2"),

        compatibility: BuiltinExampleCompatibility::RunMat,

        harness: BuiltinExampleHarness::Portable,

        fixture: BuiltinExampleFixture::None,

        requirements: BuiltinExampleRequirements::NONE,

        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(idx,[1 4 3 2]));",
        },
    },
    BuiltinExample {
        id: "example-5",

        title: "Using `ComparisonMethod` to sort by magnitude",

        program: "A = [-8 -1 3 -2];\nidx = argsort(A, 'ComparisonMethod', 'abs')",

        display_output: Some("idx =\n     2     4     3     1"),

        compatibility: BuiltinExampleCompatibility::RunMat,

        harness: BuiltinExampleHarness::Portable,

        fixture: BuiltinExampleFixture::None,

        requirements: BuiltinExampleRequirements::NONE,

        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(idx,[2 4 3 1]));",
        },
    },
    BuiltinExample {
        id: "example-6",

        title: "Handling NaN values during permutation",

        program: "A = [NaN 4 1 2];\nidx = argsort(A)",

        display_output: Some("idx =\n     3     4     2     1"),

        compatibility: BuiltinExampleCompatibility::RunMat,

        harness: BuiltinExampleHarness::Portable,

        fixture: BuiltinExampleFixture::None,

        requirements: BuiltinExampleRequirements::NONE,

        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(idx,[3 4 2 1]));",
        },
    },
    BuiltinExample {
        id: "example-7",

        title: "Argsort on GPU tensors falls back gracefully",

        program: "G = gpuArray(randn(5, 1));\nidx = argsort(G)",

        display_output: None,

        compatibility: BuiltinExampleCompatibility::RunMat,

        harness: BuiltinExampleHarness::Wgpu,

        fixture: BuiltinExampleFixture::None,

        requirements: BuiltinExampleRequirements::NONE,

        verification: BuiltinExampleVerification::Assertions {
            source: "[~, expected] = sort(gather(G)); assert(isequal(gather(idx),expected));",
        },
    },
];
const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "How is `argsort` different from `sort`?",
        answer: "`argsort` returns only the permutation indices. It behaves like calling `[~, I] = sort(X, ...)` and publishes only the indices; the current evaluation also computes sorted values.",
    },
    BuiltinDocumentationFaq {
        question: "Are the indices one-based like MATLAB?",
        answer: "Yes. All indices follow MATLAB's one-based convention so they can be used directly with subsequent indexing operations.",
    },
    BuiltinDocumentationFaq {
        question: "Does `argsort` support the same arguments as `sort`?",
        answer: "Yes. Dimension arguments, direction keywords, and `'ComparisonMethod'` behave exactly like they do for `sort`.",
    },
    BuiltinDocumentationFaq {
        question: "How are NaN values ordered?",
        answer: "NaNs are treated as missing. They appear at the end for ascending permutations and at the beginning for descending permutations, matching MATLAB.",
    },
    BuiltinDocumentationFaq {
        question: "Can I call `argsort` on GPU arrays?",
        answer: "argsort publishes only one-based double permutation indices. It currently uses the same evaluation as sort, including computing sorted values. Resident indices are restored to the input owner; a provider hook may stage work through host memory. These operations are eager fusion boundaries, not elementwise kernels or reductions, and do not broadcast. Provider availability does not imply a device-native sorting or set kernel: current providers may stage host work. An unsupported input can use exact host fallback before provider work starts. Errors after an attempt, including cancellation, are returned without repeating the operation on the host. Supported resident outputs return to their owning provider; predicates and all-indices cells have the host placement described above. Distributed execution is not supported.",
    },
    BuiltinDocumentationFaq {
        question: "Is the permutation stable?",
        answer: "Yes. Equal elements keep their relative order so that `argsort` remains consistent with MATLAB's stable sorting semantics.",
    },
    BuiltinDocumentationFaq {
        question: "What type is returned?",
        answer: "A double-precision tensor (or scalar) with the same shape as the input, containing permutation indices.",
    },
    BuiltinDocumentationFaq {
        question: "Does `argsort` mutate its input?",
        answer: "No. It only returns indices. Combine the result with indexing (`A(idx)`) to obtain reordered values when needed.",
    },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "sort",

        target: BuiltinDocumentationLinkTarget::Builtin("sort"),
    },
    BuiltinDocumentationLink {
        label: "sortrows",

        target: BuiltinDocumentationLinkTarget::Builtin("sortrows"),
    },
    BuiltinDocumentationLink {
        label: "randperm",

        target: BuiltinDocumentationLinkTarget::Builtin("randperm"),
    },
    BuiltinDocumentationLink {
        label: "max",

        target: BuiltinDocumentationLinkTarget::Builtin("max"),
    },
    BuiltinDocumentationLink {
        label: "min",

        target: BuiltinDocumentationLinkTarget::Builtin("min"),
    },
    BuiltinDocumentationLink {
        label: "intersect",

        target: BuiltinDocumentationLinkTarget::Builtin("intersect"),
    },
    BuiltinDocumentationLink {
        label: "ismember",

        target: BuiltinDocumentationLinkTarget::Builtin("ismember"),
    },
    BuiltinDocumentationLink {
        label: "issorted",

        target: BuiltinDocumentationLinkTarget::Builtin("issorted"),
    },
    BuiltinDocumentationLink {
        label: "setdiff",

        target: BuiltinDocumentationLinkTarget::Builtin("setdiff"),
    },
    BuiltinDocumentationLink {
        label: "union",

        target: BuiltinDocumentationLinkTarget::Builtin("union"),
    },
    BuiltinDocumentationLink {
        label: "unique",

        target: BuiltinDocumentationLinkTarget::Builtin("unique"),
    },
    BuiltinDocumentationLink {
        label: "MATLAB reference",

        target: BuiltinDocumentationLinkTarget::External(
            "https://www.mathworks.com/help/matlab/ref/double.sort.html",
        ),
    },
];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("argsort"),
    slug: Some("argsort"),
    summary: "Return one-based permutation indices using RunMat sort options.",
    description: "`argsort(X)` returns the permutation indices that order `X` the same way `sort(X)` would. This RunMat convenience operation uses the index behavior of `[~, I] = sort(X, ...)` for supported dimensions and directions.",
    keywords: &["argsort", "sort", "indices", "permutation", "gpu"],
    related: &["sort", "sortrows", "randperm", "max", "min", "intersect", "ismember", "issorted", "setdiff", "union", "unique"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "`crates/runmat-runtime/src/builtins/array/sorting_sets/argsort.rs`",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/sorting_sets/argsort.rs"),
        }],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "Runtime value, option, shape and error tests",
                location: "builtins::array::sorting_sets::argsort::tests",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "Runtime resident-path tests",
                location: "builtins::array::sorting_sets::argsort::tests::argsort_gpu_roundtrip",
            },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
