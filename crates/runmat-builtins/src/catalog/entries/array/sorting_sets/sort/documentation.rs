use crate::*;
const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "Scalars remain unchanged; vectors are reordered into ascending or descending order.",
            "For matrices and higher-dimensional tensors, the sort operates along a specified dimension (default: first non-singleton). All other dimensions remain untouched.",
            "`[B, I] = sort(A, ...)` returns both the sorted values `B` and the permutation indices `I`, using MATLAB's one-based indexing.",
            "Direction arguments accept `'ascend'` (default) or `'descend'`.",
            "Name-value pairs support `'ComparisonMethod'` with values `'auto'`, `'real'`, or `'abs'`. The `'abs'` option sorts by absolute value while breaking ties using the signed value.",
            "For complex inputs, the default `'ComparisonMethod'='auto'` behaves like `'abs'` (magnitude ordering) and `'real'` compares the real component first with the imaginary component used for tie-breaking.",
            "NaN values are treated as missing: they appear at the end for ascending sorts and at the beginning for descending sorts (matching MATLAB's `'MissingPlacement','auto'` behaviour for doubles).",
            "Dimensions greater than `ndims(A)` are treated as singleton dimensions (size 1) and therefore leave `A` unchanged while returning index values of `1`.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Types, placement and limitations",
        paragraphs: &["Sorting is not in-place. It preserves the input shape and native numeric or logical class; indices are doubles of the same shape. Resident values and indices are restored to the input owner. Eligible same-class real inputs may use a provider hook; unsupported candidates are gathered exactly before host evaluation."],
    },
    BuiltinDocumentationSection {
        heading: "Resident execution",
        paragraphs: &["These operations are eager fusion boundaries, not elementwise kernels or reductions, and do not broadcast. Provider availability does not imply a device-native sorting or set kernel: current providers may stage host work. An unsupported input can use exact host fallback before provider work starts. Errors after an attempt, including cancellation, are returned without repeating the operation on the host. Supported resident outputs return to their owning provider; predicates and all-indices cells have the host placement described above. Distributed execution is not supported."],
    },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "Sorting a column vector in ascending order",
        program: "A = [3; 1; 2];\nB = sort(A)",
        display_output: Some("B =\n     1\n     2\n     3"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(B,[1;2;3]));",
        },
    },
    BuiltinExample {
        id: "example-2",
        title: "Sorting rows by specifying the dimension",
        program: "A = [1 4 2; 3 2 5];\nB = sort(A, 2)",
        display_output: Some("B =\n     1     2     4\n     2     3     5"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(B,[1 2 4;2 3 5]));",
        },
    },
    BuiltinExample {
        id: "example-3",
        title: "Sorting values in descending order",
        program: "A = [10 4 7 9];\nB = sort(A, 'descend')",
        display_output: Some("B =\n    10     9     7     4"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(B,[10 9 7 4]));",
        },
    },
    BuiltinExample {
        id: "example-4",
        title: "Retrieving permutation indices alongside the sorted values",
        program: "A = [4 1 9 2];\n[B, I] = sort(A)",
        display_output: Some("B =\n     1     2     4     9\nI =\n     2     4     1     3"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(B,[1 2 4 9]) && isequal(I,[2 4 1 3]));",
        },
    },
    BuiltinExample {
        id: "example-5",
        title: "Sorting by absolute value using `ComparisonMethod`",
        program: "A = [-8 -1 3 -2];\nB = sort(A, 'ComparisonMethod', 'abs')",
        display_output: Some("B =\n    -1    -2     3    -8"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(B,[-1 -2 3 -8]));",
        },
    },
    BuiltinExample {
        id: "example-6",
        title: "Sorting tensors containing NaN values",
        program: "A = [NaN 4 1 2];\n[B, I] = sort(A)",
        display_output: Some("B =\n     1     2     4   NaN\nI =\n     3     4     2     1"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequaln(B,[1 2 4 NaN]) && isequal(I,[3 4 2 1]));",
        },
    },
    BuiltinExample {
        id: "example-7",
        title: "Sorting GPU tensors with automatic host fallback",
        program: "G = gpuArray(randn(5, 1));\n[B, I] = sort(G, 'descend')",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "[expected, indices] = sort(gather(G),'descend'); assert(isequal(gather(B),expected) && isequal(gather(I),indices));",
        },
    },
];
const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Can `sort` return both values and indices?",
        answer: "Yes. Use `[B, I] = sort(A, ...)` to receive the permutation indices alongside the sorted values.",
    },
    BuiltinDocumentationFaq {
        question: "How are NaN values handled?",
        answer: "NaN values are considered missing. They appear last for ascending sorts and first for descending sorts, matching MATLAB's `'MissingPlacement','auto'` default for doubles.",
    },
    BuiltinDocumentationFaq {
        question: "What happens when I sort along a dimension that does not exist?",
        answer: "`sort(A, dim)` treats dimensions beyond `ndims(A)` as singleton dimensions (size 1). The data remains unchanged and the index output is filled with ones.",
    },
    BuiltinDocumentationFaq {
        question: "Does `sort` support name-value arguments?",
        answer: "Yes. `'ComparisonMethod'` accepts `'auto'`, `'real'`, or `'abs'`. 'MissingPlacement' accepts 'auto', 'first', or 'last'.",
    },
    BuiltinDocumentationFaq {
        question: "Are GPU tensors sorted in-place?",
        answer: concat!("Sorting is not in-place. It preserves the input shape and native numeric or logical class; indices are doubles of the same shape. Resident values and indices are restored to the input owner. Eligible same-class real inputs may use a provider hook; unsupported candidates are gathered exactly before host evaluation. These operations are eager fusion ", "boundaries, not elementwise kernels or reductions, and do not broadcast. Provider availability does not imply a device-native sorting or set kernel: current providers may stage host work. An unsupported input can use exact host fallback before provider work starts. Errors after an attempt, including cancellation, are returned without repeating the ", "operation on the host. Supported resident outputs return to their owning provider; predicates and all-indices cells have the host placement described above. Distributed execution is not supported.",),
    },
    BuiltinDocumentationFaq {
        question: "Does `sort` preserve the shape of the input?",
        answer: "Yes. The output is the same size as the input tensor. Only values along the selected dimension are reordered.",
    },
    BuiltinDocumentationFaq {
        question: "What numeric type do the index outputs use?",
        answer: "Permutation indices are returned as double-precision tensors (or scalars) using MATLAB's one-based indexing.",
    },
    BuiltinDocumentationFaq {
        question: "Is the sorting stable?",
        answer: "Yes. Equal comparison keys preserve their original order. For real absolute-value comparison, signed value breaks equal-magnitude ties.",
    },
    BuiltinDocumentationFaq {
        question: "How does `ComparisonMethod` behave with real inputs?",
        answer: "`'auto'` and `'real'` behave identically. `'abs'` sorts by absolute value and uses the signed value to break ties so that results match MATLAB.",
    },
    BuiltinDocumentationFaq {
        question: "How does `ComparisonMethod` behave with complex inputs?",
        answer: "`'auto'` (the default) and `'abs'` order values by magnitude. `'real'` compares the real component first and falls back to the imaginary component to break ties while preserving stability.",
    },
    BuiltinDocumentationFaq {
        question: "Do logical or scalar inputs work?",
        answer: "Yes. Logical sort inputs retain their logical class. Scalars are returned unchanged with an index output of `1` when requested.",
    },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "sortrows",

        target: BuiltinDocumentationLinkTarget::Builtin("sortrows"),
    },
    BuiltinDocumentationLink {
        label: "unique",

        target: BuiltinDocumentationLinkTarget::Builtin("unique"),
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
        label: "permute",

        target: BuiltinDocumentationLinkTarget::Builtin("permute"),
    },
    BuiltinDocumentationLink {
        label: "argsort",

        target: BuiltinDocumentationLinkTarget::Builtin("argsort"),
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
        label: "MATLAB reference",

        target: BuiltinDocumentationLinkTarget::External(
            "https://www.mathworks.com/help/matlab/ref/double.sort.html",
        ),
    },
];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("sort"),
    slug: Some("sort"),
    summary: "Sort array elements along a dimension with MATLAB-compatible direction options and optional index outputs.",
    description: "`sort` orders elements along a selected dimension, defaulting to ascending order on the first non-singleton dimension. It also supports MATLAB-compatible index outputs that return permutation positions.",
    keywords: &["sort", "ascending", "descending", "indices", "comparisonmethod", "gpu"],
    related: &["sortrows", "unique", "max", "min", "permute", "argsort", "intersect", "ismember", "issorted", "setdiff", "union"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "`crates/runmat-runtime/src/builtins/array/sorting_sets/sort.rs`",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/sorting_sets/sort.rs"),
        }],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "Runtime value, option, shape and error tests",
                location: "builtins::array::sorting_sets::sort::tests",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "Runtime resident-path tests",
                location: "builtins::array::sorting_sets::sort::tests::sort_gpu_round_trip",
            },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
