use crate::*;
const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &["`intersect(A, B)` flattens both inputs column-major, removes duplicates, and returns the sorted intersection.", "`[C, IA, IB] = intersect(A, B)` also returns index vectors so that `C = A(IA)` and `C = B(IB)`.", "`intersect(A, B, 'stable')` preserves the first appearance order from `A`.", "`intersect(A, B, 'rows')` treats each row as an element. Inputs must share the same number of columns.", "Supported character, string, logical and numeric inputs are accepted; numeric mixing and class limitations are described below.", "Legacy flags such as `'legacy'` or `'R2012a'` are not supported in RunMat."],
    },
    BuiltinDocumentationSection {
        heading: "Types, placement and limitations",
        paragraphs: &["Element results are rows when both inputs are row vectors, and columns otherwise. Index outputs remain columns. Matching numeric classes are preserved. Combining double with single or one integer class preserves the nondouble class; unlike nondouble numeric classes are rejected. Logical inputs currently produce double values. NaN values never match, so intersections do not contain NaNs. A complex value or row with any NaN component likewise does not match."],
    },
    BuiltinDocumentationSection {
        heading: "Resident execution",
        paragraphs: &["These operations are eager fusion boundaries, not elementwise kernels or reductions, and do not broadcast. Provider availability does not imply a device-native sorting or set kernel: current providers may stage host work. An unsupported input can use exact host fallback before provider work starts. Errors after an attempt, including cancellation, are returned without repeating the operation on the host. Supported resident outputs return to their owning provider; predicates and all-indices cells have the host placement described above. Distributed execution is not supported."],
    },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "Finding common values in numeric vectors",
        program: "A = [5 7 5 1];\nB = [7 1 3];\n[C, IA, IB] = intersect(A, B)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(C,[1 7]) && isequal(IA,[4;2]) && isequal(IB,[2;1]));",
        },
    },
    BuiltinExample {
        id: "example-2",
        title: "Preserving input order with `'stable'`",
        program: "A = [4 2 4 1 3];\nB = [3 4 5 1];\nC = intersect(A, B, 'stable')",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(C,[4 1 3]));",
        },
    },
    BuiltinExample {
        id: "example-3",
        title: "Intersecting matrix rows",
        program: "A = [1 2; 3 4; 1 2];\nB = [2 3; 1 2];\n[C, IA, IB] = intersect(A, B, 'rows')",
        display_output: Some("C =\n     1     2\nIA =\n     1\nIB =\n     2"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(C,[1 2]) && isequal(IA,1) && isequal(IB,2));",
        },
    },
    BuiltinExample {
        id: "example-4",
        title: "Working with strings and character arrays",
        program: "names1 = [\"apple\" \"orange\" \"pear\"];\nnames2 = [\"pear\" \"grape\" \"orange\"];\n[common, IA, IB] = intersect(names1, names2, 'stable')",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(common,[\"orange\" \"pear\"]) && isequal(IA,[2;3]) && isequal(IB,[3;1]));",
        },
    },
    BuiltinExample {
        id: "example-5",
        title: "Using `intersect` on GPU arrays",
        program: "G = gpuArray([10 4 6 4]);\nH = gpuArray([6 4 2]);\nresult = intersect(G, H)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(gather(result),[4 6]));",
        },
    },
    BuiltinExample {
        id: "nonmatching-missing",
        title: "Exclude NaNs from intersections",
        program: "A = [NaN 2 NaN 1]; B = [NaN;1;2]; [C,IA,IB] = intersect(A,B); R = intersect(A,B.');",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(C,[1;2]) && isequal(IA,[4;2]) && isequal(IB,[2;3])); assert(isequal(R,[1 2]));",
        },
    },
];
const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Does `intersect` keep duplicate values?",
        answer: "No. MATLAB and RunMat return each common value at most once. Use other logic (such as logical indexing) if you need multiset behavior.",
    },
    BuiltinDocumentationFaq {
        question: "What is the default ordering?",
        answer: "`intersect` sorts results ascending by default. Specify `'stable'` to preserve the input order from `A`.",
    },
    BuiltinDocumentationFaq {
        question: "Can I intersect rows that contain strings?",
        answer: "Yes. String arrays support both element and row intersections. When using `'rows'`, the inputs must have the same number of columns.",
    },
    BuiltinDocumentationFaq {
        question: "Are NaN values considered equal?",
        answer: "Element results are rows when both inputs are row vectors, and columns otherwise. Index outputs remain columns. Matching numeric classes are preserved. Combining double with single or one integer class preserves the nondouble class; unlike nondouble numeric classes are rejected. Logical inputs currently produce double values. NaN values never match, so intersections do not contain NaNs. A complex value or row with any NaN component likewise does not match.",
    },
    BuiltinDocumentationFaq {
        question: "Is the `'legacy'` flag supported?",
        answer: "No. RunMat only implements the modern MATLAB semantics. Passing `'legacy'` or `'R2012a'` raises an error.",
    },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "union",

        target: BuiltinDocumentationLinkTarget::Builtin("union"),
    },
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
            "https://www.mathworks.com/help/matlab/ref/double.intersect.html",
        ),
    },
];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("intersect"),
    slug: Some("intersect"),
    summary: "Return common elements or rows across arrays in MATLAB and RunMat.",
    description: "`intersect(A, B)` computes the set intersection of `A` and `B`. Default ordering and optional index outputs follow MATLAB semantics.",
    keywords: &["intersect", "set", "stable", "rows", "indices", "gpu"],
    related: &["union", "unique", "sort", "sortrows", "argsort", "ismember", "issorted", "setdiff"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "`crates/runmat-runtime/src/builtins/array/sorting_sets/intersect.rs`",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/sorting_sets/intersect.rs"),
        }],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "Runtime value, option, shape and error tests",
                location: "builtins::array::sorting_sets::intersect::tests",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "Runtime resident-path tests",
                location: "builtins::array::sorting_sets::intersect::tests::intersect_gpu_roundtrip",
            },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
