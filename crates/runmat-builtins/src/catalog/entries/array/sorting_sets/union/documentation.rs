use crate::*;
const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &["`union(A, B)` flattens the inputs column-major, removes duplicates, and returns the sorted union of their values.", "`[C, IA, IB] = union(A, B)` also returns index vectors where `IA` points to the elements in `A` that contribute to `C` and `IB` refers to the elements in `B` that only appear in `B`.", "`union(A, B, 'stable')` preserves the first-seen order: unique values from `A` appear first, followed by new values appearing in `B`.", "`union(A, B, 'rows')` treats each row as a record. Inputs must share the same number of columns.", "Character arrays and string arrays are fully supported in both element and row modes.", "Complex values follow MATLAB's ordering rules (magnitude first, then real/imaginary parts).", "Legacy compatibility flags (`'legacy'` or `'R2012a'`) are not supported in RunMat."],
    },
    BuiltinDocumentationSection {
        heading: "Types, placement and limitations",
        paragraphs: &["Element results are rows when both inputs are row vectors, and columns otherwise. Index outputs remain columns. Matching numeric classes are preserved. Combining double with single or one integer class preserves the nondouble class; unlike nondouble numeric classes are rejected. Logical inputs currently produce double values. Each NaN occurrence is distinct, including repeated NaNs in the same input. Sorted output places them last; stable output preserves their occurrence order."],
    },
    BuiltinDocumentationSection {
        heading: "Resident execution",
        paragraphs: &["These operations are eager fusion boundaries, not elementwise kernels or reductions, and do not broadcast. Provider availability does not imply a device-native sorting or set kernel: current providers may stage host work. An unsupported input can use exact host fallback before provider work starts. Errors after an attempt, including cancellation, are returned without repeating the operation on the host. Supported resident outputs return to their owning provider; predicates and all-indices cells have the host placement described above. Distributed execution is not supported."],
    },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "Combining Two Numeric Vectors",
        program: "A = [5 7 1];\nB = [3 1 1];\n[C, IA, IB] = union(A, B)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(C,[1 3 5 7]) && isequal(IA,[3;1;2]) && isequal(IB,1));",
        },
    },
    BuiltinExample {
        id: "example-2",
        title: "Preserving Input Order with `'stable'`",
        program: "A = [5 7 1];\nB = [3 2 4];\nC = union(A, B, 'stable')",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(C,[5 7 1 3 2 4]));",
        },
    },
    BuiltinExample {
        id: "example-3",
        title: "Union of Matrix Rows",
        program: "A = [1 2; 3 4; 1 2];\nB = [3 4; 5 6];\n[C, IA, IB] = union(A, B, 'rows')",
        display_output: Some("C =\n     1     2\n     3     4\n     5     6\nIA =\n     1\n     2\nIB =\n     2"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(C,[1 2;3 4;5 6]) && isequal(IA,[1;2]) && isequal(IB,2));",
        },
    },
    BuiltinExample {
        id: "example-4",
        title: "Working with Character Arrays",
        program: "A = ['m','z'; 'm','a'];\nB = ['a','x'; 'm','a'];\nunion(A, B)",
        display_output: Some("ans =\n  4x1 char array\n    a\n    m\n    x\n    z"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(union(A,B),['a';'m';'x';'z']));",
        },
    },
    BuiltinExample {
        id: "example-5",
        title: "Union on GPU Data",
        program: "G = gpuArray([1 4 2 5]);\nH = gpuArray([2 6]);\n[C, IA, IB] = union(G, H)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(gather(C),[1 2 4 5 6]) && isequal(gather(IA),[1;3;2;4]) && isequal(gather(IB),2));",
        },
    },
    BuiltinExample {
        id: "distinct-missing",
        title: "Preserve distinct NaNs and row orientation",
        program: "A = [NaN NaN 1]; B = [1 NaN]; [C,IA,IB] = union(A,B); [S,SA,SB] = union(A,B,'stable');",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(C),[1 4]) && C(1)==1 && all(isnan(C(2:4)))); assert(isequal(IA,[3;1;2]) && IB==2); assert(isnan(S(1)) && isnan(S(2)) && S(3)==1 && isnan(S(4))); assert(isequal(SA,[1;2;3]) && SB==2);",
        },
    },
];
const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "How are the index outputs defined?",
        answer: "`IA` lists the positions in `A` that contribute to `C`. `IB` lists the positions in `B` for union values that do not already appear in `A`. Both use MATLAB's one-based indexing.",
    },
    BuiltinDocumentationFaq {
        question: "What happens if I request `'stable'` ordering?",
        answer: "The output keeps the first occurrence order: all unique values from `A` appear first, followed by new values from `B` in the order they are encountered.",
    },
    BuiltinDocumentationFaq {
        question: "Can I use `'rows'` with string or character arrays?",
        answer: "Yes. `union` accepts string arrays and character arrays with the `'rows'` option, provided the inputs share the same number of columns.",
    },
    BuiltinDocumentationFaq {
        question: "How does `union` treat `NaN` values?",
        answer: "Element results are rows when both inputs are row vectors, and columns otherwise. Index outputs remain columns. Matching numeric classes are preserved. Combining double with single or one integer class preserves the nondouble class; unlike nondouble numeric classes are rejected. Logical inputs currently produce double values. Each NaN occurrence is distinct, including repeated NaNs in the same input. Sorted output places them last; stable output preserves their occurrence order.",
    },
    BuiltinDocumentationFaq {
        question: "Does GPU execution change the behaviour?",
        answer: concat!(
            "Element results are rows when both inputs are row vectors, and columns otherwise. Index outputs remain columns. Matching numeric classes are preserved. Combining double with single or one integer class preserves the nondouble class; unlike nondouble numeric classes are rejected. Logical inputs currently produce double values. Each NaN occurrence is",
            " distinct, including repeated NaNs in the same input. Sorted output places them last; stable output preserves their occurrence order. These operations are eager fusion boundaries, not elementwise kernels or reductions, and do not broadcast. Provider availability does not imply a device-native sorting or set kernel: current providers may stage host ",
            "work. An unsupported input can use exact host fallback before provider work starts. Errors after an attempt, including cancellation, are returned without repeating the operation on the host. Supported resident outputs return to their owning provider; predicates and all-indices cells have the host placement described above. Distributed execution is ",
            "not supported.",
        ),
    },
    BuiltinDocumentationFaq {
        question: "What if the inputs have different shapes?",
        answer: "Element-wise unions accept any shapes; both inputs are flattened column-major. Row-wise unions require matching column counts and two-dimensional inputs.",
    },
    BuiltinDocumentationFaq {
        question: "Can I mix data types?",
        answer: "No. Inputs must belong to the same class (numeric/logical, complex, char, or string). Mixing classes produces a descriptive error.",
    },
    BuiltinDocumentationFaq {
        question: "Are scalar outputs returned as vectors?",
        answer: "Scalars follow MATLAB's behaviour: a `1×1` union result is represented as a scalar double.",
    },
    BuiltinDocumentationFaq {
        question: "Do trailing spaces matter for character data?",
        answer: "Yes. Character arrays treat every character, including trailing spaces, as significant—matching MATLAB's behaviour.",
    },
    BuiltinDocumentationFaq {
        question: "Is the `'legacy'` flag supported?",
        answer: "No. RunMat only implements the modern MATLAB semantics. Passing `'legacy'` or `'R2012a'` raises an error.",
    },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "unique",

        target: BuiltinDocumentationLinkTarget::Builtin("unique"),
    },
    BuiltinDocumentationLink {
        label: "sort",

        target: BuiltinDocumentationLinkTarget::Builtin("sort"),
    },
    BuiltinDocumentationLink {
        label: "sortrows",

        target: BuiltinDocumentationLinkTarget::Builtin("sortrows"),
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
        label: "MATLAB reference",

        target: BuiltinDocumentationLinkTarget::External(
            "https://www.mathworks.com/help/matlab/ref/double.union.html",
        ),
    },
];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("union"),
    slug: Some("union"),
    summary: "Return unions of input arrays with MATLAB-compatible ordering and optional index outputs.",
    description: "`union(A, B)` returns distinct values (or rows) appearing in either input and supports MATLAB-compatible ordering modes and optional index outputs that map back to `A` and `B`.",
    keywords: &["union", "set", "stable", "rows", "indices", "gpu"],
    related: &["unique", "sort", "sortrows", "argsort", "intersect", "ismember", "issorted", "setdiff"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "`crates/runmat-runtime/src/builtins/array/sorting_sets/union.rs`",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/sorting_sets/union.rs"),
        }],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "Runtime value, option, shape and error tests",
                location: "builtins::array::sorting_sets::union::tests",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "Runtime resident-path tests",
                location: "builtins::array::sorting_sets::union::tests::union_gpu_roundtrip",
            },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
