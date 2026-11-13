use crate::*;
const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "Without an explicit tolerance, RunMat uses MATLAB-compatible defaults: `1e-12` for double inputs and `1e-6` when either input is single precision.",
            "Element comparisons use `abs(A-B) <= tol * max(abs([A(:); B(:)]))` unless `DataScale` is supplied.",
            "`DataScale`, when scalar, changes the tolerance test to `abs(A-B) <= tol * DataScale`.",
            "`ByRows`, true compares complete rows and returns outputs of size `size(A,1) x 1`; A and B must have the same number of columns.",
            "In row mode, vector `DataScale` supplies one scale per column. A scale of `Inf` ignores that column during matching.",
            "The optional second output `LocB` contains the first one-based index in `B` for each match and `0` when absent.",
            "`OutputAllIndices`, true makes `LocB` a cell array whose cells contain all matching one-based indices, or `[0]` when no match exists.",
            "`NaN` values do not match. Exact equalities, including equal infinities, match before tolerance subtraction is considered.",
            "Complex, sparse, string, character, cell, struct, and object inputs are rejected. The documented host data and tolerance domains are `single` and `double`; RunMat mode additionally accepts host integer/logical data and typed integer/logical tolerance controls through declared compatibility extensions.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Types, placement and limitations",
        paragraphs: &["Documented resident input includes integer classes through 32 bits but excludes 64-bit integers, ByRows and OutputAllIndices; those excluded forms require RunMat extensions. Tolerance must be a finite positive real scalar. Ordinary mask and double-location outputs retain the first resident input owner; OutputAllIndices produces host cells, and requesting those cells also leaves the mask on the host. Complex, sparse and nonnumeric containers are unsupported. Host integer/logical data and typed controls, resident 64-bit integer inputs and resident row/all-index forms require their declared RunMat compatibility extensions."],
    },
    BuiltinDocumentationSection {
        heading: "Resident execution",
        paragraphs: &["These operations are eager fusion boundaries, not elementwise kernels or reductions, and do not broadcast. Provider availability does not imply a device-native sorting or set kernel: current providers may stage host work. An unsupported input can use exact host fallback before provider work starts. Errors after an attempt, including cancellation, are returned without repeating the operation on the host. Supported resident outputs return to their owning provider; predicates and all-indices cells have the host placement described above. Distributed execution is not supported."],
    },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "Testing approximate membership with default scaling",
        program: "A = [0.1 1e10];\nB = [0.1 + 1e-14, 1e10 + 5e-3];\nLIA = ismembertol(A, B)",
        display_output: Some("LIA =\n  1x2 logical array\n   1   1"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(LIA,logical([1 1])));",
        },
    },
    BuiltinExample {
        id: "example-2",
        title: "Returning first matching indices",
        program: "A = [1 2 4];\nB = [2.01 2.02 4.2];\n[LIA, LocB] = ismembertol(A, B, 0.05)",
        display_output: Some("LIA =\n  1x3 logical array\n   0   1   1\nLocB =\n     0     1     3"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(LIA,logical([0 1 1])) && isequal(LocB,[0 1 3]));",
        },
    },
    BuiltinExample {
        id: "example-3",
        title: "Using absolute tolerance with DataScale",
        program: "A = 1e10;\nB = 1e10 + 5e-3;\nLIA = ismembertol(A, B, 1e-6, 'DataScale', 1)",
        display_output: Some("LIA =\n     0"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(~LIA);",
        },
    },
    BuiltinExample {
        id: "example-4",
        title: "Comparing rows and ignoring one column",
        program: "A = [0 0.5; 10 20];\nB = [0.05 999; 10.2 -999];\n[LIA, LocB] = ismembertol(A, B, 0.1, 'ByRows', true, 'DataScale', [1 Inf])",
        display_output: Some("LIA =\n  2x1 logical array\n   1\n   0\nLocB =\n     1\n     0"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(LIA,logical([1;0])) && isequal(LocB,[1;0]));",
        },
    },
    BuiltinExample {
        id: "example-5",
        title: "Returning all matching locations",
        program: "A = 2;\nB = [1.99 2.01 9.5];\n[LIA, LocB] = ismembertol(A, B, 0.01, 'OutputAllIndices', true)",
        display_output: Some("LIA =\n     1\nLocB =\n  1x1 cell array\n    {[1; 2]}"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(LIA && iscell(LocB) && isequal(LocB{1},[1;2]));",
        },
    },
    BuiltinExample {
        id: "example-6",
        title: "Using gpuArray inputs",
        program: "G = gpuArray([1; 2]);\nH = gpuArray([1 + 1e-13; 3]);\nLIA = ismembertol(G, H)",
        display_output: Some("LIA =\n  2x1 logical array\n   1\n   0"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(gather(LIA),logical([1;0])));",
        },
    },
];
const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "How is the default tolerance chosen?",
        answer: "Double inputs use `1e-12`. If either input is single precision, RunMat uses `1e-6`.",
    },
    BuiltinDocumentationFaq {
        question: "Does `ismembertol` support string or cell inputs like `ismember`?",
        answer: "No. MATLAB's tolerance membership operation is numeric; RunMat accepts real full numeric and logical inputs only.",
    },
    BuiltinDocumentationFaq {
        question: "What does `OutputAllIndices` change?",
        answer: "The second output becomes a cell array. Each cell contains every one-based index in `B` within tolerance of the corresponding element or row of `A`, or `[0]` when no match exists.",
    },
    BuiltinDocumentationFaq {
        question: "Can `DataScale` be a vector?",
        answer: "Only when `ByRows` is true. Each vector element scales the tolerance for the corresponding column; `Inf` ignores that column.",
    },
    BuiltinDocumentationFaq {
        question: "Do NaN values match each other?",
        answer: "No. `NaN` does not match any value, including another `NaN`, for `ismembertol`.",
    },
    BuiltinDocumentationFaq {
        question: "Will `ismembertol` run on the GPU automatically?",
        answer: concat!(
            "Documented resident input includes integer classes through 32 bits but excludes 64-bit integers, ByRows and OutputAllIndices; those excluded forms require RunMat extensions. Tolerance must be a finite positive real scalar. Ordinary mask and double-location outputs retain the first resident input owner; OutputAllIndices produces host cells, and requ",
            "esting those cells also leaves the mask on the host. Complex, sparse and nonnumeric containers are unsupported. Host integer/logical data and typed controls, resident 64-bit integer inputs and resident row/all-index forms require their declared RunMat compatibility extensions. These operations are eager fusion boundaries, not elementwise kernels or",
            " reductions, and do not broadcast. Provider availability does not imply a device-native sorting or set kernel: current providers may stage host work. An unsupported input can use exact host fallback before provider work starts. Errors after an attempt, including cancellation, are returned without repeating the operation on the host. Supported resid",
            "ent outputs return to their owning provider; predicates and all-indices cells have the host placement described above. Distributed execution is not supported.",
        ),
    },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "ismember",

        target: BuiltinDocumentationLinkTarget::Builtin("ismember"),
    },
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
        label: "issorted",

        target: BuiltinDocumentationLinkTarget::Builtin("issorted"),
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
        label: "MATLAB reference",

        target: BuiltinDocumentationLinkTarget::External(
            "https://www.mathworks.com/help/matlab/ref/ismembertol.html",
        ),
    },
];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("ismembertol"),
    slug: Some("ismembertol"),
    summary: "Test numeric membership within tolerance in MATLAB and RunMat.",
    description: "`ismembertol(A, B)` returns logical membership flags for real numeric values in `A` that are within tolerance of values in `B`. Optional outputs return first or all matching one-based indices into `B`.",
    keywords: &["ismembertol", "membership", "tolerance", "set", "rows", "indices", "datascale"],
    related: &["ismember", "unique", "intersect", "setdiff", "union", "issorted", "gpuArray", "gather"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "`crates/runmat-runtime/src/builtins/array/sorting_sets/ismembertol.rs`",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/sorting_sets/ismembertol.rs"),
        }],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "Runtime value, option, shape and error tests",
                location: "builtins::array::sorting_sets::ismembertol::tests",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "Runtime resident-path tests",
                location: "builtins::array::sorting_sets::ismembertol::tests::documented_resident_integer_outputs_restore_and_excess_arity_rejects",
            },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
