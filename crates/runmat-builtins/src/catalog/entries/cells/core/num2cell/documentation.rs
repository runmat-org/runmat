use crate::*;

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "elements", title: "Place each matrix element in a cell", program: "A = [1 2; 3 4];\nC = num2cell(A);", display_output: Some("C is a 2-by-2 cell array with one numeric scalar in each cell"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(C), [2 2]));\nassert(C{1,1} == 1);\nassert(C{2,1} == 3);\nassert(C{1,2} == 2);\nassert(C{2,2} == 4);" } },
    BuiltinExample { id: "rows", title: "Keep each matrix row together", program: "A = [1 2 3; 4 5 6];\nC = num2cell(A, 2);", display_output: Some("C is a 2-by-1 cell array of row vectors"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(C), [2 1]));\nassert(isequal(C{1}, [1 2 3]));\nassert(isequal(C{2}, [4 5 6]));" } },
    BuiltinExample { id: "columns", title: "Keep each matrix column together", program: "A = [1 2 3; 4 5 6];\nC = num2cell(A, 1);", display_output: Some("C is a 1-by-3 cell array of column vectors"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(C), [1 3]));\nassert(isequal(C{2}, [2; 5]));" } },
    BuiltinExample { id: "dimension-order", title: "Choose the dimension order inside each cell", program: "A = reshape(1:6, [2 3]);\nC = num2cell(A, [2 1]);", display_output: Some("C contains one 3-by-2 block because the requested dimension order is [2 1]"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(C), [1 1]));\nassert(isequal(size(C{1}), [3 2]));\nassert(isequal(C{1}, [1 2; 3 4; 5 6]));" } },
    BuiltinExample { id: "exact-integer", title: "Preserve exact integer storage", program: "A = [0x0020000000000001u64, 0xFFFFFFFFFFFFFFFFu64];\nC = num2cell(A);", display_output: Some("Each cell retains one exact uint64 value"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isa(C{1}, 'uint64'));\nassert(C{1} == 0x0020000000000001u64);\nassert(C{2} == 0xFFFFFFFFFFFFFFFFu64);" } },
    BuiltinExample { id: "text", title: "Split a string array", program: "A = [\"north\", \"south\"];\nC = num2cell(A);", display_output: Some("C is a 1-by-2 cell array of string scalars"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(C), [1 2]));\nassert(strcmp(C{1}, \"north\"));\nassert(strcmp(C{2}, \"south\"));" } },
    BuiltinExample { id: "invalid-dimension-matrix", title: "Reject a matrix dimension selector", program: "num2cell(reshape(1:8, [2 2 2]), [1 2; 2 1]);", display_output: Some("RunMat:num2cell:InvalidInput"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::ExpectedError { identifier: "RunMat:num2cell:InvalidInput" } },
];

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Element and block conversion", paragraphs: &["`num2cell(A)` creates a cell array with the same shape as `A`; each cell contains one element with its original class. `num2cell(A, dim)` keeps the selected dimensions together inside each cell and sets those dimensions to one in the outer cell array.", "A dimension selector is a positive integer scalar or vector. Its order controls the order of dimensions inside each contained block, so `[2 1]` and `[1 2]` can produce differently shaped blocks."] },
    BuiltinDocumentationSection { heading: "Types and storage", paragraphs: &["RunMat supports dense and sparse real, complex, logical, character, string, symbolic, cell, and object arrays. Sparse inputs produce ordinary dense values inside the cells. Cell inputs produce nested scalar or grouped cell arrays. Object values retain their class and handle identity.", "Fixed-width integers remain exact. Numeric blocks keep their input class, and container values are moved into their result cells without converting their payloads."] },
    BuiltinDocumentationSection { heading: "Execution", paragraphs: &["The result is a host cell array. A resident numeric input is gathered once before partitioning; no cell construction kernel is launched. Distributed input is materialized because grouped dimensions can cross partition boundaries.", "Categorical, datetime, duration, and calendar-duration arrays are not yet represented by the runtime. The catalog marks this contract incomplete until those documented forms are available."] },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Does num2cell convert numeric classes?", answer: "No. Each scalar or grouped block retains the input numeric class, including fixed-width integers and single precision." },
    BuiltinDocumentationFaq { question: "What happens to sparse input?", answer: "Sparse input is partitioned into ordinary dense scalars or blocks, matching the value semantics of the selected sparse elements." },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "cell",
        target: BuiltinDocumentationLinkTarget::Builtin("cell"),
    },
    BuiltinDocumentationLink {
        label: "cell2mat",
        target: BuiltinDocumentationLinkTarget::Builtin("cell2mat"),
    },
    BuiltinDocumentationLink {
        label: "mat2cell",
        target: BuiltinDocumentationLinkTarget::Builtin("mat2cell"),
    },
];

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/tree/main/crates/runmat-runtime/src/builtins/cells/core/num2cell") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Array classes, dimension ordering, shape, exact integer, and rejection behavior", location: "builtins::cells::core::num2cell::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Output facts and dimension diagnostics", location: "catalog::entries::cells::core::num2cell::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Executable catalog examples", location: "scripts/runtime/verify-builtin-examples.mjs" },
    ],
    notes: &["Compatible behavior was checked against the public num2cell reference for MATLAB R2026a."],
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("num2cell"),
    slug: Some("num2cell"),
    summary: "Convert an array into a cell array, optionally grouping dimensions.",
    description:
        "`num2cell` partitions an array into scalar cells or into blocks selected by dimension.",
    keywords: &[
        "num2cell",
        "cell array",
        "array conversion",
        "group dimensions",
    ],
    related: &["cell", "cell2mat", "mat2cell"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: EVIDENCE,
    introduced: Some("before R2006a"),
    status: Some(BuiltinDocumentationStatus::Partial),
};
