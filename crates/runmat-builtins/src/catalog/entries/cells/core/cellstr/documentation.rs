use crate::*;

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "character-rows", title: "Convert character-array rows", program: "A = ['abc '; 'defg'; 'hi  '];\nC = cellstr(A);", display_output: Some("C is a 3-by-1 cell array containing 'abc', 'defg', and 'hi'"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(C), [3 1]));\nassert(strcmp(C{1}, 'abc'));\nassert(strcmp(C{2}, 'defg'));\nassert(strcmp(C{3}, 'hi'));" } },
    BuiltinExample { id: "string-array", title: "Preserve a string array's shape", program: "A = [\"north\", \"south\"; \"east\", \"west\"];\nC = cellstr(A);", display_output: Some("C is a 2-by-2 cell array of character vectors"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(C), [2 2]));\nassert(strcmp(C{1, 1}, 'north'));\nassert(strcmp(C{2, 1}, 'east'));\nassert(strcmp(C{1, 2}, 'south'));\nassert(strcmp(C{2, 2}, 'west'));" } },
    BuiltinExample { id: "empty-string", title: "Convert an empty string scalar", program: "C = cellstr(\"\");", display_output: Some("C is a scalar cell containing an empty character array"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(C), [1 1]));\nassert(isempty(C{1}));\nassert(ischar(C{1}));" } },
    BuiltinExample { id: "string-scalar", title: "Convert one string scalar", program: "C = cellstr(\"RunMat\");", display_output: Some("C is a scalar cell containing 'RunMat'"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(C), [1 1]));\nassert(strcmp(C{1}, 'RunMat'));" } },
    BuiltinExample { id: "empty-character-array", title: "Convert an empty character array", program: "A = char(zeros(0, 5));\nC = cellstr(A);", display_output: Some("C is a 0-by-1 cell array"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(C), [0 1]));\nassert(isempty(C));" } },
    BuiltinExample { id: "cell-input", title: "Normalize an existing text cell array", program: "C = cellstr({'left', \"right\"});", display_output: Some("C contains two character vectors"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(C{1}, 'left'));\nassert(strcmp(C{2}, 'right'));" } },
    BuiltinExample { id: "invalid-numeric", title: "Reject numeric input", program: "cellstr(42);", display_output: Some("RunMat:cellstr:InvalidInput"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, verification: BuiltinExampleVerification::ExpectedError { identifier: "RunMat:cellstr:InvalidInput" } },
];

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Character and string arrays", paragraphs: &["`cellstr(A)` converts each string element to a character vector and preserves the string array's shape. A character array instead produces one output cell per row and removes trailing whitespace from each row, except for significant whitespace such as a nonbreaking space.", "An empty string scalar produces one cell containing an empty character array. An empty string array preserves its array shape, while a character array with zero rows produces a 0-by-1 cell array."] },
    BuiltinDocumentationSection { heading: "RunMat text inputs", paragraphs: &["RunMat compatibility mode also accepts cell arrays containing character vectors and string scalars, and symbolic scalars or arrays. Cell-array conversion preserves the outer shape and moves existing character vectors into the new container without inspecting resident numeric providers.", "Cell and symbolic inputs are explicit RunMat extensions. MATLAB compatibility mode rejects those forms. Numeric, logical, object, and resident tensor inputs are not converted to text."] },
    BuiltinDocumentationSection { heading: "Current coverage", paragraphs: &["Character and string arrays are supported in native and browser runtimes. Datetime, duration, calendar-duration, and categorical forms are not yet implemented; the catalog marks the contract incomplete until those documented forms have implementations.", "RunMat's string storage does not yet distinguish a missing string from the literal text `<missing>`. The documentation therefore makes no missing-value compatibility claim."] },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Which whitespace is removed from character-array rows?", answer: "Trailing whitespace is removed except for significant whitespace such as a nonbreaking space. Interior characters are unchanged; character vectors already stored in cell inputs are not trimmed." },
    BuiltinDocumentationFaq { question: "Does conversion gather GPU data?", answer: "No. RunMat has no resident text representation. Resident numeric values reject before provider lookup, including values nested in a cell input." },
    BuiltinDocumentationFaq { question: "Why are cell and symbolic inputs marked as extensions?", answer: "They are useful RunMat conversions but are not part of the documented compatible input set. They require RunMat compatibility mode." },
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
        label: "cellfun",
        target: BuiltinDocumentationLinkTarget::Builtin("cellfun"),
    },
    BuiltinDocumentationLink {
        label: "iscellstr",
        target: BuiltinDocumentationLinkTarget::Builtin("iscellstr"),
    },
    BuiltinDocumentationLink {
        label: "mat2cell",
        target: BuiltinDocumentationLinkTarget::Builtin("mat2cell"),
    },
    BuiltinDocumentationLink {
        label: "string",
        target: BuiltinDocumentationLinkTarget::Builtin("string"),
    },
];

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/tree/main/crates/runmat-runtime/src/builtins/cells/core/cellstr") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Character, string, cell, shape, and rejection behavior", location: "builtins::cells::core::cellstr::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed output facts and diagnostics", location: "catalog::entries::cells::core::cellstr::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Executable catalog examples", location: "scripts/runtime/verify-builtin-examples.mjs" },
    ],
    notes: &["Compatible behavior was checked against the public cellstr reference for MATLAB R2026a."],
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("cellstr"),
    slug: Some("cellstr"),
    summary: "Convert text arrays to a cell array of character vectors.",
    description: "`cellstr` converts character rows or string elements into character vectors stored in a cell array.",
    keywords: &["cellstr", "cell array", "character vector", "string array", "text conversion"],
    related: &[
        "cell",
        "cell2mat",
        "cellfun",
        "iscellstr",
        "mat2cell",
        "string",
    ],
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
