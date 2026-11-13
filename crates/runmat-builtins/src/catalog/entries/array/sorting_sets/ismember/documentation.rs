use crate::*;
const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &["The first output `tf` has the same shape as `A` (or `size(A,1) × 1` when using `'rows'`).", "The optional second output `loc` contains one-based indices into `B`, with `0` for values that are not found.", "Duplicate values in `A` return the index of the first occurrence in `B` every time they match.", "NaN values do not match any value, including NaNs in B; their mask is false and location is zero.", "Character arrays follow column-major linear indexing, mirroring MATLAB.", "The `'rows'` option compares complete rows; inputs must agree on the number of columns.", "Legacy flags (`'legacy'`, `'R2012a'`) are deliberately unsupported in RunMat."],
    },
    BuiltinDocumentationSection {
        heading: "Types, placement and limitations",
        paragraphs: &["The logical mask and double first-location output have the shape of A, or one row per record in rows mode. NaNs never match, even another NaN. Complex values or rows containing any NaN component also return false and location zero. Resident outputs are restored to the first resident input owner."],
    },
    BuiltinDocumentationSection {
        heading: "Resident execution",
        paragraphs: &["These operations are eager fusion boundaries, not elementwise kernels or reductions, and do not broadcast. Provider availability does not imply a device-native sorting or set kernel: current providers may stage host work. An unsupported input can use exact host fallback before provider work starts. Errors after an attempt, including cancellation, are returned without repeating the operation on the host. Supported resident outputs return to their owning provider; predicates and all-indices cells have the host placement described above. Distributed execution is not supported."],
    },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "Checking membership of numeric vectors",
        program: "A = [5 7 2 7];\nB = [7 9 5];\n[tf, loc] = ismember(A, B)",
        display_output: Some("tf =\n     1     1     0     1\nloc =\n     3     1     0     1"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf,logical([1 1 0 1])) && isequal(loc,[3 1 0 1]));",
        },
    },
    BuiltinExample {
        id: "example-2",
        title: "Finding row membership in a matrix",
        program: "A = [1 2; 3 4; 1 2];\nB = [3 4; 5 6; 1 2];\n[tf, loc] = ismember(A, B, 'rows')",
        display_output: Some("tf =\n     1\n     1\n     1\nloc =\n     3\n     1\n     3"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf,true(3,1)) && isequal(loc,[3;1;3]));",
        },
    },
    BuiltinExample {
        id: "example-3",
        title: "Locating values and retrieving the index",
        program: "values = [10 20 30];\nset = [30 10 40];\n[tf, loc] = ismember(values, set)",
        display_output: Some("tf =\n     1     0     1\nloc =\n     2     0     1"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf,logical([1 0 1])) && isequal(loc,[2 0 1]));",
        },
    },
    BuiltinExample {
        id: "example-4",
        title: "Testing characters against a set",
        program: "chars = ['r','u'; 'n','m'];\nset = ['m','a'; 'r','u'];\n[tf, loc] = ismember(chars, set)",
        display_output: Some("tf = [1 1; 0 1]; loc = [2 4; 0 1]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf,logical([1 1;0 1])) && isequal(loc,[2 4;0 1]));",
        },
    },
    BuiltinExample {
        id: "example-5",
        title: "Working with string arrays",
        program: "A = [\"apple\" \"pear\" \"banana\"];\nB = [\"pear\" \"orange\" \"apple\"];\n[tf, loc] = ismember(A, B)",
        display_output: Some("tf =\n  1×3 logical array\n   1   1   0\nloc =\n  1×3 double\n   3   1   0"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf,logical([1 1 0])) && isequal(loc,[3 1 0]));",
        },
    },
    BuiltinExample {
        id: "example-6",
        title: "Using `ismember` with `gpuArray` inputs",
        program: "G = gpuArray([1 4 2 4]);\nH = gpuArray([4 5]);\n[tf, loc] = ismember(G, H)",
        display_output: Some("tf =\n     0     1     0     1\nloc =\n     0     1     0     1"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(gather(tf),logical([0 1 0 1])) && isequal(gather(loc),[0 1 0 1]));",
        },
    },
    BuiltinExample {
        id: "nonmatching-missing",
        title: "Return false and zero for NaN membership",
        program: "[tf,loc] = ismember([NaN 1 NaN],[NaN 1]); [rows,positions] = ismember([1 NaN;2 3],[1 NaN;2 3],'rows');",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(tf,logical([0 1 0])) && isequal(loc,[0 2 0])); assert(isequal(rows,logical([0;1])) && isequal(positions,[0;2]));",
        },
    },
];
const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Does `ismember` treat `NaN` values as equal?",
        answer: "The logical mask and double first-location output have the shape of A, or one row per record in rows mode. NaNs never match, even another NaN. Complex values or rows containing any NaN component also return false and location zero. Resident outputs are restored to the first resident input owner.",
    },
    BuiltinDocumentationFaq {
        question: "What happens when an element of `A` is not found in `B`?",
        answer: "The corresponding logical entry is `false` and the index output stores `0`, matching MATLAB.",
    },
    BuiltinDocumentationFaq {
        question: "Can I use `ismember` with string arrays and character arrays?",
        answer: "Yes. String arrays, scalar strings, and character arrays are supported. Mixed string/char inputs should be normalised (for example, convert scalars with `string`).",
    },
    BuiltinDocumentationFaq {
        question: "How does the `'rows'` option change the output shape?",
        answer: "`'rows'` compares entire rows and returns outputs of size `size(A,1) × 1`, regardless of how many columns the input matrices contain.",
    },
    BuiltinDocumentationFaq {
        question: "Are the legacy flags supported?",
        answer: "No. RunMat does not implement legacy flag semantics. Passing `'legacy'` or `'R2012a'` raises an error, just like other set builtins in RunMat.",
    },
    BuiltinDocumentationFaq {
        question: "Will `ismember` run on the GPU automatically?",
        answer: concat!("The logical mask and double first-location output have the shape of A, or one row per record in rows mode. NaNs never match, even another NaN. Complex values or rows containing any NaN component also return false and location zero. Resident outputs are restored to the first resident input owner. These operations are eager fusion boundaries, not ele", "mentwise kernels or reductions, and do not broadcast. Provider availability does not imply a device-native sorting or set kernel: current providers may stage host work. An unsupported input can use exact host fallback before provider work starts. Errors after an attempt, including cancellation, are returned without repeating the operation on the ho", "st. Supported resident outputs return to their owning provider; predicates and all-indices cells have the host placement described above. Distributed execution is not supported.",),
    },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "unique",

        target: BuiltinDocumentationLinkTarget::Builtin("unique"),
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
        label: "union",

        target: BuiltinDocumentationLinkTarget::Builtin("union"),
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
            "https://www.mathworks.com/help/matlab/ref/double.ismember.html",
        ),
    },
];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("ismember"),
    slug: Some("ismember"),
    summary: "Test membership of elements or rows in MATLAB and RunMat.",
    description: "`ismember(A, B)` returns membership flags for elements (or rows) of `A` in `B`, with optional first-match indices. Supported ordering and comparison forms are described below.",
    keywords: &["ismember", "membership", "set", "rows", "indices", "gpu"],
    related: &["unique", "intersect", "setdiff", "union", "gpuArray", "gather", "argsort", "issorted", "sort", "sortrows"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "`crates/runmat-runtime/src/builtins/array/sorting_sets/ismember.rs`",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/sorting_sets/ismember.rs"),
        }],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "Runtime value, option, shape and error tests",
                location: "builtins::array::sorting_sets::ismember::tests",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "Runtime resident-path tests",
                location: "builtins::array::sorting_sets::ismember::tests::ismember_gpu_roundtrip",
            },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
