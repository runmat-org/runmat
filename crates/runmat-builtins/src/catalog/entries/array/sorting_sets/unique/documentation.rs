use crate::*;
const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "`unique(A)` flattens numeric, logical, character, string, or complex arrays column-major into a vector of unique values sorted ascending.",
            "`[C, IA] = unique(A)` also returns the indices of the selected occurrences (`IA`) so that `C = A(IA)`.",
            "`[C, IA, IC] = unique(A)` provides `IC`, the mapping from each element of `A` to the corresponding index in `C`.",
            "`unique(A, 'stable')` preserves the first appearance order rather than sorting.",
            "`unique(A, 'rows')` treats each row as an observation and returns a matrix (or char/string array) whose rows are unique.",
            "`unique(A, 'last')` or `'first'` controls which occurrence contributes to `IA` (defaults to `'first'`).",
            "Combinations such as `unique(A, 'rows', 'stable', 'last')` follow MATLAB's precedence rules; mutually exclusive flags (e.g. `'sorted'` with `'stable'`) are rejected.",
            "Empty inputs return empty outputs with consistent dimensions.",
            "Legacy switches such as `'legacy'` or `'R2012a'` are not supported; RunMat does not implement the legacy flag forms.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Types, placement and limitations",
        paragraphs: &["Values retain their native numeric, logical, character or string class. Element results preserve row-vector orientation; other element results are columns. IA and IC are double columns. TreatMissingAsDistinct defaults to true; false groups missing values. Resident values and indices are restored to their input owner. Resident 64-bit integer inputs and explicit order combined with explicit occurrence selection are not supported."],
    },
    BuiltinDocumentationSection {
        heading: "Resident execution",
        paragraphs: &["These operations are eager fusion boundaries, not elementwise kernels or reductions, and do not broadcast. Provider availability does not imply a device-native sorting or set kernel: current providers may stage host work. An unsupported input can use exact host fallback before provider work starts. Errors after an attempt, including cancellation, are returned without repeating the operation on the host. Supported resident outputs return to their owning provider; predicates and all-indices cells have the host placement described above. Distributed execution is not supported."],
    },
];
const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "example-1",
        title: "Getting Sorted Unique Values",
        program: "A = [3 1 3 2];\nC = unique(A)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(C,[1 2 3]));",
        },
    },
    BuiltinExample {
        id: "example-2",
        title: "Preserving Input Order with `'stable'`",
        program: "A = [4 2 4 1 2];\nC = unique(A, 'stable')",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(C,[4 2 1]));",
        },
    },
    BuiltinExample {
        id: "example-3",
        title: "Returning Indices for Reconstruction",
        program: "A = [7 5 7 3];\n[C, IA, IC] = unique(A);\nreconstructed = C(IC)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(C,[3 5 7]) && isequal(IA,[4;2;1]) && isequal(IC,[3;2;3;1])); assert(isequal(reconstructed(:),A(:)));",
        },
    },
    BuiltinExample {
        id: "example-4",
        title: "Finding Unique Rows in a Matrix",
        program: "A = [1 3; 1 3; 2 4; 1 2];\n[C, IA, IC] = unique(A, 'rows')",
        display_output: Some("C =\n     1     2\n     1     3\n     2     4\nIA =\n     4\n     1\n     3\nIC =\n     2\n     2\n     3\n     1"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(C,[1 2;1 3;2 4]) && isequal(IA,[4;1;3]) && isequal(IC,[2;2;3;1]));",
        },
    },
    BuiltinExample {
        id: "example-5",
        title: "Selecting Last Occurrences",
        program: "A = [9 8 9 7 8];\n[C, IA] = unique(A, 'last')",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(C,[7 8 9]) && isequal(IA,[4;5;3]));",
        },
    },
    BuiltinExample {
        id: "example-6",
        title: "Working with Empty Arrays",
        program: "A = zeros(0, 3);\n[C, IA, IC] = unique(A, 'rows')",
        display_output: Some("C =\nIA =\nIC ="),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(C),[0 3]) && isempty(IA) && isempty(IC));",
        },
    },
    BuiltinExample {
        id: "example-7",
        title: "Using `unique` on GPU Arrays",
        program: "G = gpuArray([5 3 5 1]);\n[C, IA, IC] = unique(G, 'stable')",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Wgpu,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(gather(C),[5 3 1]) && isequal(gather(IA),[1;2;4]) && isequal(gather(IC),[1;2;1;3]));",
        },
    },
    BuiltinExample {
        id: "example-8",
        title: "Unique Characters in a Char Array",
        program: "chars = ['m','z'; 'm','a'];\n[C, IA] = unique(chars)",
        display_output: Some("C =\n    a\n    m\n    z\nIA =\n    4\n    1\n    3"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(C,['a';'m';'z']) && isequal(IA,[4;1;3]));",
        },
    },
    BuiltinExample {
        id: "example-9",
        title: "Unique Strings with Row Deduplication",
        program: "S = [\"alpha\" \"beta\"; \"alpha\" \"beta\"; \"gamma\" \"beta\"];\n[C, IA, IC] = unique(S, 'rows', 'stable')",
        display_output: Some("C =\n  2x2 string array\n    \"alpha\"    \"beta\"\n    \"gamma\"    \"beta\"\nIA =\n     1\n     3\nIC =\n     1\n     1\n     2"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(C,[\"alpha\" \"beta\";\"gamma\" \"beta\"]) && isequal(IA,[1;3]) && isequal(IC,[1;1;2]));",
        },
    },
];
const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Which ordering does `unique` use by default?",
        answer: "Results are sorted in ascending order unless you pass `'stable'`, which preserves the first occurrence order.",
    },
    BuiltinDocumentationFaq {
        question: "How are the index outputs defined?",
        answer: "`IA` indexes into the original data (or rows). `IC` is a column vector mapping each element (or row) of the input to the position of the corresponding unique value in `C`.",
    },
    BuiltinDocumentationFaq {
        question: "What do `'first'` and `'last'` control?",
        answer: "They determine whether `IA` references the first or last occurrence of each distinct value/row. They do not affect `C` or `IC`.",
    },
    BuiltinDocumentationFaq {
        question: "Can I combine `'rows'` with `'stable'` or `'last'`?",
        answer: "Yes. All permutations of `'rows'`, `'stable'`/`'sorted'`, and `'first'`/`'last'` are accepted. Conflicting flags are rejected; GPU combinations have the limitations described above.",
    },
    BuiltinDocumentationFaq {
        question: "Does `unique` support complex numbers or characters?",
        answer: "Yes. Complex values use magnitude ordering for the sorted output, and character or string arrays produce results in their native container types (char arrays and string arrays respectively).",
    },
    BuiltinDocumentationFaq {
        question: "How does `unique` treat NaN values?",
        answer: "NaNs are distinct by default. TreatMissingAsDistinct=false groups missing values. Sorted output places NaNs last; stable output follows input occurrence order.",
    },
    BuiltinDocumentationFaq {
        question: "Are GPU arrays supported?",
        answer: concat!(
            "Values retain their native numeric, logical, character or string class. Element results preserve row-vector orientation; other element results are columns. IA and IC are double columns. TreatMissingAsDistinct defaults to true; false groups missing values. Resident values and indices are restored to their input owner. Resident 64-bit integer inputs ",
            "and explicit order combined with explicit occurrence selection are not supported. These operations are eager fusion boundaries, not elementwise kernels or reductions, and do not broadcast. Provider availability does not imply a device-native sorting or set kernel: current providers may stage host work. An unsupported input can use exact host fallba",
            "ck before provider work starts. Errors after an attempt, including cancellation, are returned without repeating the operation on the host. Supported resident outputs return to their owning provider; predicates and all-indices cells have the host placement described above. Distributed execution is not supported.",
        ),
    },
    BuiltinDocumentationFaq {
        question: "Does `unique` preserve array shape?",
        answer: "Element results preserve row-vector orientation and otherwise are columns. Row results retain the input column count.",
    },
    BuiltinDocumentationFaq {
        question: "What happens with empty inputs?",
        answer: "Empty inputs (including empty matrices) return empty outputs with matching dimensions, and index outputs are empty column vectors.",
    },
    BuiltinDocumentationFaq {
        question: "Is `unique` stable?",
        answer: "Sorting is stable where applicable; ties preserve their relative order. You can also request `'stable'` explicitly.",
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
            "https://www.mathworks.com/help/matlab/ref/double.unique.html",
        ),
    },
];
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("unique"),
    slug: Some("unique"),
    summary: "Return unique elements or rows with MATLAB-compatible ordering and index mappings.",
    description: "`unique(A)` removes duplicates and returns distinct elements or rows. It supports MATLAB-compatible `sorted`/`stable` ordering plus optional index outputs linking original and unique values.",
    keywords: &["unique", "set", "distinct", "stable", "rows", "indices", "gpu", "string", "char"],
    related: &["sort", "sortrows", "argsort", "intersect", "ismember", "issorted", "setdiff", "union"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "`crates/runmat-runtime/src/builtins/array/sorting_sets/unique.rs`",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/sorting_sets/unique.rs"),
        }],
        verification: &[
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::UnitTest,
                label: "Runtime value, option, shape and error tests",
                location: "builtins::array::sorting_sets::unique::tests",
            },
            BuiltinEvidenceReference {
                kind: BuiltinEvidenceKind::IntegrationTest,
                label: "Runtime resident-path tests",
                location: "builtins::array::sorting_sets::unique::tests::unique_gpu_roundtrip",
            },
        ],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
