use crate::*;
const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &["`issorted(A)` examines the first non-singleton dimension and returns a logical scalar.", "`issorted(A, dim)` selects the dimension explicitly (1-based).", "Direction flags include `'ascend'`, `'descend'`, `'monotonic'`, `'strictascend'`, `'strictdescend'`, and `'strictmonotonic'`.", "Name-value options mirror MATLAB:", "`'MissingPlacement'` accepts `'auto'`, `'first'`, or `'last'`.", "`'ComparisonMethod'` accepts `'auto'`, `'real'`, or `'abs'` for numeric and complex data.", "`issorted(A,'rows', ...)` verifies lexicographic ordering of rows.", "Logical and character arrays are promoted to double precision for comparison; string arrays are compared lexicographically, applying missing-placement rules only to typed missing elements.", "Empty arrays, scalars, and singleton dimensions are always reported as sorted."],
    },
    BuiltinDocumentationSection {
        heading: "Types, placement and limitations",
        paragraphs: &["The result is a host logical scalar. Real, complex, logical, character and string data use dimension or row checks without changing the input. GPU input is gathered; nonvector GPU data and explicit GPU MissingPlacement require declared RunMat extensions."],
    },
    BuiltinDocumentationSection {
        heading: "Resident execution",
        paragraphs: &["These operations are eager fusion boundaries, not elementwise kernels or reductions, and do not broadcast. Provider availability does not imply a device-native sorting or set kernel: current providers may stage host work. An unsupported input can use exact host fallback before provider work starts. Errors after an attempt, including cancellation, are returned without repeating the operation on the host. Supported resident outputs return to their owning provider; predicates and all-indices cells have the host placement described above. Distributed execution is not supported."],
    },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",

        title: "Checking an ascending vector",

        program: "A = [5 12 33 39 78 90 95 107];\ntf = issorted(A)",

        display_output: Some("tf =\n     1"),

        compatibility: BuiltinExampleCompatibility::Matlab,

        harness: BuiltinExampleHarness::Portable,

        fixture: BuiltinExampleFixture::None,

        requirements: BuiltinExampleRequirements::NONE,

        verification: BuiltinExampleVerification::Assertions {
            source: "assert(tf && islogical(tf));",
        },
    },
    BuiltinExample {
        id: "example-2",

        title: "Verifying strict increase",

        program: "B = [1 1 2 3];\ntf = issorted(B, 'strictascend')",

        display_output: Some("tf =\n     0"),

        compatibility: BuiltinExampleCompatibility::Matlab,

        harness: BuiltinExampleHarness::Portable,

        fixture: BuiltinExampleFixture::None,

        requirements: BuiltinExampleRequirements::NONE,

        verification: BuiltinExampleVerification::Assertions {
            source: "assert(~tf);",
        },
    },
    BuiltinExample {
        id: "example-3",

        title: "Using a specific dimension",

        program: "C = [1 4 2; 3 6 5];\ntf = issorted(C, 2, 'descend')",

        display_output: Some("tf =\n     0"),

        compatibility: BuiltinExampleCompatibility::Matlab,

        harness: BuiltinExampleHarness::Portable,

        fixture: BuiltinExampleFixture::None,

        requirements: BuiltinExampleRequirements::NONE,

        verification: BuiltinExampleVerification::Assertions {
            source: "assert(~tf);",
        },
    },
    BuiltinExample {
        id: "example-4",

        title: "Allowing either ascending or descending order",

        program: "D = [10 8 8 5 1];\ntf = issorted(D, 'monotonic')",

        display_output: Some("tf =\n     1"),

        compatibility: BuiltinExampleCompatibility::Matlab,

        harness: BuiltinExampleHarness::Portable,

        fixture: BuiltinExampleFixture::None,

        requirements: BuiltinExampleRequirements::NONE,

        verification: BuiltinExampleVerification::Assertions {
            source: "assert(tf);",
        },
    },
    BuiltinExample {
        id: "example-5",

        title: "Controlling missing placements",

        program: "E = [NaN NaN 4 7];\ntf = issorted(E, 'MissingPlacement', 'first')",

        display_output: Some("tf = 1"),

        compatibility: BuiltinExampleCompatibility::Matlab,

        harness: BuiltinExampleHarness::Portable,

        fixture: BuiltinExampleFixture::None,

        requirements: BuiltinExampleRequirements::NONE,

        verification: BuiltinExampleVerification::Assertions {
            source: "assert(tf);",
        },
    },
    BuiltinExample {
        id: "example-6",

        title: "Comparing complex data by magnitude",

        program: "Z = [1+1i, 2+3i, 4+0i];\ntf = issorted(Z, 'ComparisonMethod', 'abs')",

        display_output: Some("tf =\n     1"),

        compatibility: BuiltinExampleCompatibility::Matlab,

        harness: BuiltinExampleHarness::Portable,

        fixture: BuiltinExampleFixture::None,

        requirements: BuiltinExampleRequirements::NONE,

        verification: BuiltinExampleVerification::Assertions {
            source: "assert(tf);",
        },
    },
    BuiltinExample {
        id: "example-7",

        title: "Checking row order",

        program: "R = [1 2 3; 1 2 4; 2 0 1];\ntf = issorted(R, 'rows')",

        display_output: Some("tf =\n     1"),

        compatibility: BuiltinExampleCompatibility::Matlab,

        harness: BuiltinExampleHarness::Portable,

        fixture: BuiltinExampleFixture::None,

        requirements: BuiltinExampleRequirements::NONE,

        verification: BuiltinExampleVerification::Assertions {
            source: "assert(tf);",
        },
    },
    BuiltinExample {
        id: "example-8",

        title: "Working with string arrays",

        program: "str = [ \"apple\" \"banana\"; \"apple\" \"carrot\" ];\ntf = issorted(str, 2)",

        display_output: Some("tf = 1"),

        compatibility: BuiltinExampleCompatibility::Matlab,

        harness: BuiltinExampleHarness::Portable,

        fixture: BuiltinExampleFixture::None,

        requirements: BuiltinExampleRequirements::NONE,

        verification: BuiltinExampleVerification::Assertions {
            source: "assert(tf);",
        },
    },
];
const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Does `issorted` modify its input?",
        answer: "No. The function inspects the data in-place and returns a logical scalar.",
    },
    BuiltinDocumentationFaq {
        question: "What is the difference between `'ascend'` and `'strictascend'`?",
        answer: "`'ascend'` allows consecutive equal values, while `'strictascend'` requires every element to be strictly greater than its predecessor and rejects missing values.",
    },
    BuiltinDocumentationFaq {
        question: "How does `'monotonic'` work?",
        answer: "`'monotonic'` succeeds when the data is entirely non-decreasing *or* non-increasing. Use `'strictmonotonic'` to forbid repeated or missing values.",
    },
    BuiltinDocumentationFaq {
        question: "Can I control where NaN values appear?",
        answer: "Yes. `'MissingPlacement','first'` requires missing values to precede finite ones, `'last'` requires them to trail, and `'auto'` follows MATLAB’s default (end for ascending checks, start for descending).",
    },
    BuiltinDocumentationFaq {
        question: "Is `'ComparisonMethod'` relevant for real data?",
        answer: "For real data `'auto'` and `'real'` are identical. `'abs'` compares magnitudes first and breaks ties using the signed value, matching MATLAB.",
    },
    BuiltinDocumentationFaq {
        question: "Does the function support GPU arrays?",
        answer: "The result is a host logical scalar. Real, complex, logical, character and string data use dimension or row checks without changing the input. GPU input is gathered; nonvector GPU data and explicit GPU MissingPlacement require declared RunMat extensions. These operations are eager fusion boundaries, not elementwise kernels or reductions, and do not broadcast. Provider availability does not imply a device-native sorting or set kernel: current providers may stage host work. An unsupported input can use exact host fallback before provider work starts. Errors after an attempt, including cancellation, are returned without repeating the operation on the host. Supported resident outputs return to their owning provider; predicates and all-indices cells have the host placement described above. Distributed execution is not supported.",
    },
    BuiltinDocumentationFaq {
        question: "How are string arrays treated?",
        answer: "Strings are compared lexicographically using Unicode code-point order. `MissingPlacement` rules apply to typed missing values. Literal `<missing>` text participates in ordinary lexicographic comparison.",
    },
    BuiltinDocumentationFaq {
        question: "What about empty arrays or dimensions of length 0?",
        answer: "Empty slices are considered sorted. Passing a dimension larger than `ndims(A)` also returns `true`.",
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
        label: "unique",

        target: BuiltinDocumentationLinkTarget::Builtin("unique"),
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
            "https://www.mathworks.com/help/matlab/ref/double.issorted.html",
        ),
    },
];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("issorted"),
    slug: Some("issorted"),
    summary: "Test whether arrays are sorted in MATLAB and RunMat.",
    description: "`issorted(A)` tests whether data is already sorted. Dimension, order direction, strictness, missing placement, and row-wise modes follow MATLAB semantics.",
    keywords: &["issorted", "sorted", "monotonic", "strictascend", "rows", "comparisonmethod", "missingplacement"],
    related: &["sort", "sortrows", "unique", "argsort", "intersect", "ismember", "setdiff", "union"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "`crates/runmat-runtime/src/builtins/array/sorting_sets/issorted.rs`",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/sorting_sets/issorted.rs"),
        }],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "Runtime value, option, shape and error tests",
                location: "builtins::array::sorting_sets::issorted::tests",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "Runtime resident-path tests",
                location: "builtins::array::sorting_sets::issorted::tests::issorted_gpu_roundtrip",
            },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
