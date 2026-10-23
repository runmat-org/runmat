use crate::*;
const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "`sortrows(A)` sorts by column `1`, then column `2`, and so on, all in ascending order.",
            "`sortrows(A, C)` treats the vector `C` as the column order. Positive entries sort ascending; negative entries sort descending.",
            "`sortrows(A, 'descend')` sorts all columns in descending order. Combine this with a column vector to mix directions.",
            "`[B, I] = sortrows(A, ...)` also returns `I`, the 1-based row permutation indices.",
            "`sortrows` is stable: rows that compare equal keep their original order.",
            "For complex inputs, `'ComparisonMethod'` accepts `'auto'`, `'real'`, or `'abs'`, using the supported comparison rules.",
            "NaN handling mirrors MATLAB: in ascending sorts rows containing NaN values move to the end; in descending sorts they move to the beginning.",
            "`'MissingPlacement'` lets you choose whether NaN (and other missing) rows appear `'first'`, `'last'`, or follow MATLAB's `'auto'` default.",
            "Character arrays are sorted lexicographically using their character codes.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Types, placement and limitations",
        paragraphs: &["Matrix and character rows, and table rows with variable selectors, are supported. String arrays are not supported. Matrix values retain their class and shape; row indices are double columns. Resident outputs are restored to the input owner after provider or host evaluation. Missing placement is applied lexicographically at each selected key, not as a separate whole-row bucket."],
    },
    BuiltinDocumentationSection {
        heading: "Resident execution",
        paragraphs: &["These operations are eager fusion boundaries, not elementwise kernels or reductions, and do not broadcast. Provider availability does not imply a device-native sorting or set kernel: current providers may stage host work. An unsupported input can use exact host fallback before provider work starts. Errors after an attempt, including cancellation, are returned without repeating the operation on the host. Supported resident outputs return to their owning provider; predicates and all-indices cells have the host placement described above. Distributed execution is not supported."],
    },
    BuiltinDocumentationSection {
        heading: "Syntax",
        paragraphs: &["example: {\"description\":\"Syntax\",\"input\":\"B = sortrows(A)\\nB = sortrows(A, column)\\nB = sortrows(A, column, direction)\\n[B, index] = sortrows(___)\"}", "points: [\"`A` is the data to sort — typically a numeric or character matrix. Tables and timetables are also accepted by MATLAB's `sortrows` API (RunMat currently supports matrix and char-array inputs fully).\",\"`column` selects the sort keys. It can be a single column index or a vector of indices; negative entries sort that column in descending order (for example, `[1 -3]` sorts ascending by column `1` then descending by column `3`).\",\"`direction` is an optional `'ascend'` or `'descend'` string, or a cell array of strings matching `column` entry-for-entry. It can be combined with signed `column` vectors to override per-column directions.\",\"`index` is the optional second output: a row-permutation vector such that `B = A(index, :)`. Use it to reorder paired arrays or to undo the sort.\"]"],
    },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",

        title: "Sorting rows of a matrix in ascending order",

        program: "A = [3 2; 1 4; 2 1];\nB = sortrows(A)",

        display_output: Some("B =\n     1     4\n     2     1\n     3     2"),

        compatibility: BuiltinExampleCompatibility::Matlab,

        harness: BuiltinExampleHarness::Portable,

        fixture: BuiltinExampleFixture::None,

        requirements: BuiltinExampleRequirements::NONE,

        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(B,[1 4;2 1;3 2]));",
        },
    },
    BuiltinExample {
        id: "example-2",

        title: "Sorting by a custom column order",

        program: "A = [1 4 2; 3 2 5; 3 2 1];\nB = sortrows(A, [2 3 1])",

        display_output: Some("B =\n     3     2     1\n     3     2     5\n     1     4     2"),

        compatibility: BuiltinExampleCompatibility::Matlab,

        harness: BuiltinExampleHarness::Portable,

        fixture: BuiltinExampleFixture::None,

        requirements: BuiltinExampleRequirements::NONE,

        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(B,[3 2 1;3 2 5;1 4 2]));",
        },
    },
    BuiltinExample {
        id: "example-3",

        title: "Sorting rows in descending order",

        program: "A = [2 8; 4 1; 3 5];\nB = sortrows(A, 'descend')",

        display_output: Some("B =\n     4     1\n     3     5\n     2     8"),

        compatibility: BuiltinExampleCompatibility::Matlab,

        harness: BuiltinExampleHarness::Portable,

        fixture: BuiltinExampleFixture::None,

        requirements: BuiltinExampleRequirements::NONE,

        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(B,[4 1;3 5;2 8]));",
        },
    },
    BuiltinExample {
        id: "example-4",

        title: "Mixing ascending and descending directions",

        program: "A = [1 7 3; 1 2 9; 1 2 3];\nB = sortrows(A, [1 -2 3])",

        display_output: Some("B =\n     1     7     3\n     1     2     3\n     1     2     9"),

        compatibility: BuiltinExampleCompatibility::Matlab,

        harness: BuiltinExampleHarness::Portable,

        fixture: BuiltinExampleFixture::None,

        requirements: BuiltinExampleRequirements::NONE,

        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(B,[1 7 3;1 2 3;1 2 9]));",
        },
    },
    BuiltinExample {
        id: "example-5",

        title: "Sorting rows of a character array",

        program: "names = ['bob '; 'al  '; 'ally'];\nsorted = sortrows(names)",

        display_output: Some("sorted =\nal\nally\nbob"),

        compatibility: BuiltinExampleCompatibility::Matlab,

        harness: BuiltinExampleHarness::Portable,

        fixture: BuiltinExampleFixture::None,

        requirements: BuiltinExampleRequirements::NONE,

        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(sorted,['al  ';'ally';'bob ']));",
        },
    },
    BuiltinExample {
        id: "example-6",

        title: "Sorting rows of complex data by magnitude",

        program: "Z = [3+4i, 3; 1+2i, 4];\nB = sortrows(Z, 'ComparisonMethod', 'abs')",

        display_output: Some("B =\n    1.0000 + 2.0000i    4.0000\n    3.0000 + 4.0000i    3.0000"),

        compatibility: BuiltinExampleCompatibility::Matlab,

        harness: BuiltinExampleHarness::Portable,

        fixture: BuiltinExampleFixture::None,

        requirements: BuiltinExampleRequirements::NONE,

        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(B,[1+2i 4;3+4i 3]));",
        },
    },
    BuiltinExample {
        id: "example-7",

        title: "Forcing NaN rows to the top",

        program: "A = [1 NaN; NaN 2];\nB = sortrows(A, 'MissingPlacement', 'first')",

        display_output: Some("B =\n   NaN     2\n     1   NaN"),

        compatibility: BuiltinExampleCompatibility::Matlab,

        harness: BuiltinExampleHarness::Portable,

        fixture: BuiltinExampleFixture::None,

        requirements: BuiltinExampleRequirements::NONE,

        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequaln(B,[NaN 2;1 NaN]));",
        },
    },
    BuiltinExample {
        id: "example-8",

        title: "Sorting GPU-resident data with automatic host fallback",

        program: "G = gpuArray([3 1; 2 4; 1 2]);\n[B, I] = sortrows(G)",

        display_output: None,

        compatibility: BuiltinExampleCompatibility::Matlab,

        harness: BuiltinExampleHarness::Wgpu,

        fixture: BuiltinExampleFixture::None,

        requirements: BuiltinExampleRequirements::NONE,

        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(gather(B),[1 2;2 4;3 1]) && isequal(gather(I),[3;2;1]));",
        },
    },
];
const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Can I request the permutation indices?",
        answer: "Yes. Call `[B, I] = sortrows(A, ...)` to receive the 1-based row permutation indices in `I`.",
    },
    BuiltinDocumentationFaq {
        question: "How do I sort specific columns?",
        answer: "Provide a column vector, e.g. `sortrows(A, [2 -3])` sorts by column `2` ascending and column `3` descending.",
    },
    BuiltinDocumentationFaq {
        question: "What happens when rows contain NaN values?",
        answer: "Rows containing NaNs move to the bottom for ascending sorts and to the top for descending sorts when `'MissingPlacement'` is left at its `'auto'` default, matching MATLAB.",
    },
    BuiltinDocumentationFaq {
        question: "How can I force NaNs or missing values to the top or bottom?",
        answer: "Use the name-value pair `'MissingPlacement','first'` to place missing rows before finite ones, or `'MissingPlacement','last'` to move them to the end regardless of direction.",
    },
    BuiltinDocumentationFaq {
        question: "Does `sortrows` work with complex numbers?",
        answer: "Yes. Use `'ComparisonMethod','real'` to sort by the real component or `'abs'` to sort by magnitude (the default behaviour matches MATLAB's `'auto'` rules).",
    },
    BuiltinDocumentationFaq {
        question: "Can I combine a direction string with a column vector?",
        answer: "Yes. `sortrows(A, [1 3], 'descend')` applies descending order to both columns after applying the specified column order.",
    },
    BuiltinDocumentationFaq {
        question: "Is the operation stable?",
        answer: "Yes. Rows that compare equal remain in their original order.",
    },
    BuiltinDocumentationFaq {
        question: "Does `sortrows` mutate its input?",
        answer: "No. It returns a sorted copy of the input. GPU inputs are gathered to host memory when required.",
    },
    BuiltinDocumentationFaq {
        question: "Are string arrays supported?",
        answer: "String arrays are not yet supported. Convert them to character matrices or use tables before sorting.",
    },
    BuiltinDocumentationFaq {
        question: "What does sortrows do in MATLAB?",
        answer: "`sortrows(A)` sorts the rows of matrix `A` in ascending order based on the elements in the first column. Rows with equal first-column values are sorted by subsequent columns.",
    },
    BuiltinDocumentationFaq {
        question: "How do I sort by a specific column in MATLAB?",
        answer: "Use `sortrows(A, col)` where `col` is the column index. Use negative values for descending order, e.g., `sortrows(A, -2)` sorts by the second column in descending order.",
    },
    BuiltinDocumentationFaq {
        question: "Is sortrows a stable sort?",
        answer: "Yes. `sortrows` uses a stable sorting algorithm, meaning rows with equal sort keys retain their original relative order.",
    },
    BuiltinDocumentationFaq {
        question: "Is there a `sortrows` equivalent in NumPy or pandas?",
        answer: "— In pandas use `df.sort_values(by=['col1', 'col2'], ascending=[True, False])`. In NumPy there is no direct `sortrows`, but `A[np.lexsort(A.T[::-1])]` sorts rows lexicographically (NumPy's `lexsort` reads its keys in reverse priority, so the `[::-1]` restores MATLAB's left-to-right order). For descending columns, negate the corresponding key: `np.lexsort([-A[:, 1], A[:, 0]])`.",
    },
    BuiltinDocumentationFaq {
        question: "How do I sort by multiple columns with different directions?",
        answer: "— Pass a signed column vector as the second argument. Positive entries sort ascending, negative entries sort descending. For example, `sortrows(A, [1 -2])` sorts column `1` ascending and column `2` descending, and `sortrows(A, [3 -1 2])` orders by column `3` ascending, column `1` descending, then column `2` ascending as a tiebreaker. You can also write `sortrows(A, [1 2], {'ascend','descend'})` for the equivalent name-based form.",
    },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "sort",

        target: BuiltinDocumentationLinkTarget::Builtin("sort"),
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
            "https://www.mathworks.com/help/matlab/ref/double.sortrows.html",
        ),
    },
];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("sortrows"),
    slug: Some("sortrows"),
    summary: "Sort matrix rows lexicographically with MATLAB-compatible column selection and direction controls.",
    description: "`sortrows` reorders rows of matrices or character arrays in lexicographic order. It supports MATLAB-compatible controls for comparison columns and per-column ascending/descending direction.",
    keywords: &["sortrows", "row sort", "lexicographic", "gpu"],
    related: &["sort", "unique", "max", "min", "permute", "argsort", "intersect", "ismember", "issorted", "setdiff", "union"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "`crates/runmat-runtime/src/builtins/array/sorting_sets/sortrows.rs`",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/sorting_sets/sortrows.rs"),
        }],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "Runtime value, option, shape and error tests",
                location: "builtins::array::sorting_sets::sortrows::tests",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "Runtime resident-path tests",
                location: "builtins::array::sorting_sets::sortrows::tests::sortrows_gpu_roundtrip",
            },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
