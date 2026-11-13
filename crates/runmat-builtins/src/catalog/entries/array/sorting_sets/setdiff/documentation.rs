use crate::*;
const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &["`setdiff(A, B)` flattens inputs column-major, removes duplicates, subtracts the values of `B` from `A`, and returns the remaining elements **sorted** ascending by default.", "`[C, IA] = setdiff(A, B)` also returns indices so that `C = A(IA)`.", "`setdiff(A, B, 'stable')` preserves the first appearance order from `A` instead of sorting.", "`setdiff(A, B, 'rows')` treats each row as an element. Inputs must share the same number of columns.", "Character arrays, string arrays, logical arrays, numeric types, and complex values are all supported.", "Legacy flags (`'legacy'`, `'R2012a'`) are not supported; RunMat does not implement legacy flag semantics."],
    },
    BuiltinDocumentationSection {
        heading: "Types, placement and limitations",
        paragraphs: &["Element results follow A: a row-vector A gives a row result, otherwise a column. IA remains a column. Numeric and logical outputs retain A's class. A is never narrowed through the single comparison domain; supported mixed-class controls convert the B comparison operand instead. Unlike nondouble numeric classes are rejected. NaNs never match B or each other. Every NaN occurrence in A remains in C, including complex values or rows containing a NaN component."],
    },
    BuiltinDocumentationSection {
        heading: "Resident execution",
        paragraphs: &["These operations are eager fusion boundaries, not elementwise kernels or reductions, and do not broadcast. Provider availability does not imply a device-native sorting or set kernel: current providers may stage host work. An unsupported input can use exact host fallback before provider work starts. Errors after an attempt, including cancellation, are returned without repeating the operation on the host. Supported resident outputs return to their owning provider; predicates and all-indices cells have the host placement described above. Distributed execution is not supported."],
    },
    BuiltinDocumentationSection {
        heading: "RunMat exact mixed-integer comparison",
        paragraphs: &["With double A and native integer B, RunMat compares their original numeric values exactly, including int64 and uint64 values beyond the exact double range. B is not rounded through double storage. C still contains selected original A values in double class; IA is a double column. This is a RunMat exactness policy for this mixed form, separate from output-class conversion.", "The route uses indexed native comparisons for elements or rows and preserves the ordinary sorted/stable and distinct-NaN rules. Managed finite host allocation is not supported for this route; those calls fail before provider work or gathering because complete selector scratch and retained-output ownership is not available."],
    },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "Finding values exclusive to the first numeric vector",
        program: "A = [5 7 5 1];\nB = [7 1 3];\n[C, IA] = setdiff(A, B)",
        display_output: Some("C =\n     5\nIA =\n     1"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(C,5) && isequal(IA,1));",
        },
    },
    BuiltinExample {
        id: "example-2",
        title: "Preserving input order with `'stable'`",
        program: "A = [4 2 4 1 3];\nB = [3 4 5 1];\n[C, IA] = setdiff(A, B, 'stable')",
        display_output: Some("C =\n     2\nIA =\n     2"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(C,2) && isequal(IA,2));",
        },
    },
    BuiltinExample {
        id: "example-3",
        title: "Working with rows of numeric matrices",
        program: "A = [1 2; 3 4; 1 2];\nB = [3 4; 5 6];\n[C, IA] = setdiff(A, B, 'rows')",
        display_output: Some("C =\n     1     2\nIA =\n     1"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(C,[1 2]) && isequal(IA,1));",
        },
    },
    BuiltinExample {
        id: "example-4",
        title: "Computing set difference for character data",
        program: "A = ['m','z'; 'm','a'];\nB = ['a','x'; 'm','a'];\n[C, IA] = setdiff(A, B)",
        display_output: Some("C = 'z'; IA = 3"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(C,'z') && isequal(IA,3));",
        },
    },
    BuiltinExample {
        id: "example-5",
        title: "Subtracting string arrays by row",
        program: "A = [\"alpha\" \"beta\"; \"gamma\" \"beta\"];\nB = [\"gamma\" \"beta\"; \"delta\" \"beta\"];\n[C, IA] = setdiff(A, B, 'rows', 'stable')",
        display_output: Some("C =\n  1x2 string array\n    \"alpha\"    \"beta\"\nIA =\n     1"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(C,[\"alpha\" \"beta\"]) && isequal(IA,1));",
        },
    },
    BuiltinExample {
        id: "example-6",
        title: "Using `setdiff` with GPU arrays",
        program: "G = gpuArray([10 4 6 4]);\nH = gpuArray([6 4 2]);\nC = setdiff(G, H)",
        display_output: Some("C =\n    10"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(gather(C),10));",
        },
    },
    BuiltinExample {
        id: "left-class-and-missing",
        title: "Retain A precision, row orientation and every NaN",
        program: "A = [1+eps(1) 2 NaN NaN]; B = single([2 NaN]); [C,IA] = setdiff(A,B); L = setdiff(logical([1 0 1]),false);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(C,'double') && isequal(size(C),[1 3]) && C(1)==A(1) && all(isnan(C(2:3)))); assert(isequal(IA,[1;3;4]) && islogical(L) && L);",
        },
    },
    BuiltinExample {
        id: "wide-integer-comparison",
        title: "Compare a wide integer without rounding it into A",
        program: "A = [9007199254740992 1.5 NaN]; B = [0x0020000000000001u64 0u64]; [C,IA] = setdiff(A,B,'stable');",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(C,'double') && isequal(size(C),[1 3]) && C(1)==A(1) && C(2)==1.5 && isnan(C(3))); assert(isequal(IA,[1;2;3]) && isa(IA,'double'));",
        },
    },
];
const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "What ordering does `setdiff` use by default?",
        answer: "Results are sorted ascending. Specify `'stable'` to preserve the first appearance order from the first input.",
    },
    BuiltinDocumentationFaq {
        question: "How are the index outputs defined?",
        answer: "`IA` points to the positions in `A` that correspond to each element (or row) returned in `C`, using MATLAB's one-based indexing.",
    },
    BuiltinDocumentationFaq {
        question: "Can I combine `'rows'` with `'stable'`?",
        answer: "Yes. `'rows'` can be paired with either `'sorted'` (default) or `'stable'`. Other option combinations that conflict (e.g. `'sorted'` with `'stable'`) are rejected.",
    },
    BuiltinDocumentationFaq {
        question: "Does `setdiff` remove `NaN` values from `A` when they exist in `B`?",
        answer: "Element results follow A: a row-vector A gives a row result, otherwise a column. IA remains a column. Numeric and logical outputs retain A's class. A is never narrowed through the single comparison domain; supported mixed-class controls convert the B comparison operand instead. Unlike nondouble numeric classes are rejected. NaNs never match B or each other. Every NaN occurrence in A remains in C, including complex values or rows containing a NaN component.",
    },
    BuiltinDocumentationFaq {
        question: "Are complex numbers supported?",
        answer: "Absolutely. Complex values use MATLAB's ordering rules (magnitude, then real part, then imaginary part) for the sorted output.",
    },
    BuiltinDocumentationFaq {
        question: "Does GPU execution change the results?",
        answer: concat!(
            "Element results follow A: a row-vector A gives a row result, otherwise a column. IA remains a column. Numeric and logical outputs retain A's class. A is never narrowed through the single comparison domain; supported mixed-class controls convert the B comparison operand instead. Unlike nondouble numeric classes are rejected. NaNs never match B or ea",
            "ch other. Every NaN occurrence in A remains in C, including complex values or rows containing a NaN component. These operations are eager fusion boundaries, not elementwise kernels or reductions, and do not broadcast. Provider availability does not imply a device-native sorting or set kernel: current providers may stage host work. An unsupported in",
            "put can use exact host fallback before provider work starts. Errors after an attempt, including cancellation, are returned without repeating the operation on the host. Supported resident outputs return to their owning provider; predicates and all-indices cells have the host placement described above. Distributed execution is not supported.",
        ),
    },
    BuiltinDocumentationFaq {
        question: "What happens if the inputs have different classes?",
        answer: "Supported numeric inputs retain A's class, including when double is combined with single or one integer class. Unlike nondouble numeric classes are rejected.",
    },
    BuiltinDocumentationFaq {
        question: "Can I request `'legacy'` behaviour?",
        answer: "No. RunMat does not implement legacy flag semantics. Passing `'legacy'` or `'R2012a'` results in an error.",
    },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "unique",

        target: BuiltinDocumentationLinkTarget::Builtin("unique"),
    },
    BuiltinDocumentationLink {
        label: "union",

        target: BuiltinDocumentationLinkTarget::Builtin("union"),
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
        label: "gpuArray",

        target: BuiltinDocumentationLinkTarget::Builtin("gpuArray"),
    },
    BuiltinDocumentationLink {
        label: "gather",

        target: BuiltinDocumentationLinkTarget::Builtin("gather"),
    },
    BuiltinDocumentationLink {
        label: "argsort",

        target: BuiltinDocumentationLinkTarget::Builtin("argsort"),
    },
    BuiltinDocumentationLink {
        label: "issorted",

        target: BuiltinDocumentationLinkTarget::Builtin("issorted"),
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
        label: "MATLAB reference",

        target: BuiltinDocumentationLinkTarget::External(
            "https://www.mathworks.com/help/matlab/ref/double.setdiff.html",
        ),
    },
];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("setdiff"),
    slug: Some("setdiff"),
    summary: "Return values present in the first input but absent from the second, with MATLAB-compatible ordering modes.",
    description: "`setdiff(A, B)` returns unique values (or rows) that appear in `A` but not in `B`. It supports MATLAB-compatible `'sorted'` and `'stable'` ordering modes plus row-wise set differences.",
    keywords: &["setdiff", "difference", "stable", "rows", "indices", "gpu"],
    related: &["unique", "union", "intersect", "ismember", "gpuArray", "gather", "argsort", "issorted", "sort", "sortrows"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "crates/runmat-runtime/src/builtins/array/sorting_sets/setdiff.rs",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/sorting_sets/setdiff.rs"),
        }],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "Runtime value, option, shape and error tests",
                location: "builtins::array::sorting_sets::setdiff::tests",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "Runtime resident-path tests",
                location: "builtins::array::sorting_sets::setdiff::tests::setdiff_gpu_roundtrip",
            },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
