use crate::*;
const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "`setxor(A, B)` flattens inputs column-major, removes duplicate non-NaN values within each input, and returns the sorted symmetric difference. Element outputs are row vectors when both inputs are row vectors; otherwise they are column vectors.",
            "`[C, IA, IB] = setxor(A, B)` returns one-based indices so `C` is a sorted or stable combination of `A(IA)` and `B(IB)`.",
            "`setxor(A, B, 'stable')` preserves first-seen order: exclusive values from `A` appear first, followed by exclusive values from `B`.",
            "`setxor(A, B, 'rows')` treats each row as one set element. Numeric, complex, character, and string row inputs must have matching column counts.",
            "Numeric outputs preserve matching input classes. A double input may be combined with one nondouble numeric class, in which case the nondouble class is preserved; two different nondouble classes are rejected.",
            "Character arrays can be compared with numeric character-code arrays. String arrays can be compared with character vectors by treating each character row as a string scalar.",
            "`NaN` values are treated as distinct, matching MATLAB `setxor` behavior.",
            "Legacy flags (`'legacy'`, `'R2012a'`) and requests for more than three outputs are rejected; RunMat implements modern MATLAB semantics.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Types, placement and limitations",
        paragraphs: &["Outputs retain supported native numeric classes, including single and wide integer values. Logical pairs currently produce double values. Both row-vector inputs produce a row-vector result. NaNs remain distinct. Host execution is authoritative for resident inputs, and public outputs are restored to the first resident input owner."],
    },
    BuiltinDocumentationSection {
        heading: "Resident execution",
        paragraphs: &["These operations are eager fusion boundaries, not elementwise kernels or reductions, and do not broadcast. Provider availability does not imply a device-native sorting or set kernel: current providers may stage host work. An unsupported input can use exact host fallback before provider work starts. Errors after an attempt, including cancellation, are returned without repeating the operation on the host. Supported resident outputs return to their owning provider; predicates and all-indices cells have the host placement described above. Distributed execution is not supported."],
    },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "Symmetric difference of two numeric vectors",
        program: "A = [5 1 3 3 3];\nB = [4 1 2];\n[C, IA, IB] = setxor(A, B)",
        display_output: Some("C =\n     2     3     4     5\nIA =\n     3\n     1\nIB =\n     3\n     1"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(C,[2 3 4 5]) && isequal(IA,[3;1]) && isequal(IB,[3;1]));",
        },
    },
    BuiltinExample {
        id: "example-2",
        title: "Preserving input order",
        program: "[C, IA, IB] = setxor([5 1 3 3 3], [4 1 2], 'stable')",
        display_output: Some("C =\n     5     3     4     2\nIA =\n     1\n     3\nIB =\n     1\n     3"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(C,[5 3 4 2]) && isequal(IA,[1;3]) && isequal(IB,[1;3]));",
        },
    },
    BuiltinExample {
        id: "example-3",
        title: "Symmetric difference of rows",
        program: "A = [7 8 9; 7 7 1; 7 7 1; 1 2 3; 4 5 6];\nB = [1 2 3; 4 5 6; 7 7 2];\n[C, IA, IB] = setxor(A, B, 'rows')",
        display_output: Some("C =\n     7     7     1\n     7     7     2\n     7     8     9\nIA =\n     2\n     1\nIB =\n     3"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(C,[7 7 1;7 7 2;7 8 9]) && isequal(IA,[2;1]) && isequal(IB,3));",
        },
    },
    BuiltinExample {
        id: "example-4",
        title: "NaN values remain distinct",
        program: "C = setxor([5 NaN NaN], [5 NaN NaN])",
        display_output: Some("C =\n   NaN   NaN   NaN   NaN"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(C),[1 4]) && all(isnan(C)));",
        },
    },
];
const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "What ordering does `setxor` use by default?",
        answer: "Results are sorted ascending by default. Specify `'stable'` to preserve the first occurrence order from `A` followed by `B`.",
    },
    BuiltinDocumentationFaq {
        question: "How are `IA` and `IB` defined?",
        answer: "`IA` indexes exclusive values or rows from `A`, and `IB` indexes exclusive values or rows from `B`. Both use MATLAB's one-based indexing.",
    },
    BuiltinDocumentationFaq {
        question: "Can I use `setxor` with rows?",
        answer: "Yes. Use `'rows'` for numeric, complex, character, or string matrices with matching column counts.",
    },
    BuiltinDocumentationFaq {
        question: "How are `NaN` values handled?",
        answer: "Unlike several other set helpers, MATLAB `setxor` treats `NaN` values as distinct. RunMat preserves that behavior.",
    },
    BuiltinDocumentationFaq {
        question: "Does GPU execution change results?",
        answer: concat!("Outputs retain supported native numeric classes, including single and wide integer values. Logical pairs currently produce double values. Both row-vector inputs produce a row-vector result. NaNs remain distinct. Host execution is authoritative for resident inputs, and public outputs are restored to the first resident input owner. These operations a", "re eager fusion boundaries, not elementwise kernels or reductions, and do not broadcast. Provider availability does not imply a device-native sorting or set kernel: current providers may stage host work. An unsupported input can use exact host fallback before provider work starts. Errors after an attempt, including cancellation, are returned withou", "t repeating the operation on the host. Supported resident outputs return to their owning provider; predicates and all-indices cells have the host placement described above. Distributed execution is not supported.",),
    },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "union",

        target: BuiltinDocumentationLinkTarget::Builtin("union"),
    },
    BuiltinDocumentationLink {
        label: "intersect",

        target: BuiltinDocumentationLinkTarget::Builtin("intersect"),
    },
    BuiltinDocumentationLink {
        label: "setdiff",

        target: BuiltinDocumentationLinkTarget::Builtin("setdiff"),
    },
    BuiltinDocumentationLink {
        label: "unique",

        target: BuiltinDocumentationLinkTarget::Builtin("unique"),
    },
    BuiltinDocumentationLink {
        label: "ismember",

        target: BuiltinDocumentationLinkTarget::Builtin("ismember"),
    },
    BuiltinDocumentationLink {
        label: "MATLAB reference",

        target: BuiltinDocumentationLinkTarget::External(
            "https://www.mathworks.com/help/matlab/ref/double.setxor.html",
        ),
    },
];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("setxor"),
    slug: Some("setxor"),
    summary: "Return values or rows that appear in exactly one of two inputs.",
    description: "`setxor(A, B)` returns the symmetric difference of two arrays: values or rows that occur in `A` or `B`, but not both. Results are sorted by default and can be returned in first-seen order with `'stable'`.",
    keywords: &["setxor", "symmetric difference", "exclusive or", "stable", "rows", "indices", "gpu"],
    related: &["union", "intersect", "setdiff", "unique", "ismember"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "`crates/runmat-runtime/src/builtins/array/sorting_sets/setxor.rs`",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/sorting_sets/setxor.rs"),
        }],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "Runtime value, option, shape and error tests",
                location: "builtins::array::sorting_sets::setxor::tests",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "Runtime resident-path tests",
                location: "builtins::array::sorting_sets::setxor::tests::setxor_gpu_roundtrip",
            },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
