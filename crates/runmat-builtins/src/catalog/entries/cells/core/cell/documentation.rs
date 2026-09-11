use crate::*;

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "empty",
        title: "Create an empty cell array",
        program: "C = cell();",
        display_output: Some("C is a 0-by-0 cell array"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(iscell(C));\nassert(isequal(size(C), [0 0]));",
        },
    },
    BuiltinExample {
        id: "square",
        title: "Create a square cell array",
        program: "C = cell(3);",
        display_output: Some("C is a 3-by-3 cell array"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(C), [3 3]));\nassert(isequal(C{1}, []));",
        },
    },
    BuiltinExample {
        id: "rectangle",
        title: "Create a rectangular cell array",
        program: "C = cell(2, 4);",
        display_output: Some("C is a 2-by-4 cell array"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source:
                "assert(iscell(C));\nassert(isequal(size(C), [2 4]));\nassert(isequal(C{2,4}, []));",
        },
    },
    BuiltinExample {
        id: "size-vector",
        title: "Use an existing array size",
        program: "A = ones(5, 2);\nC = cell(size(A));",
        display_output: Some("C has the same 5-by-2 shape as A"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(C), [5 2]));",
        },
    },
    BuiltinExample {
        id: "nd",
        title: "Create an N-dimensional cell array",
        program: "C = cell(2, 3, 4);",
        display_output: Some("C is a 2-by-3-by-4 cell array"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(C), [2 3 4]));",
        },
    },
    BuiltinExample {
        id: "integer-size",
        title: "Use exact integer size controls",
        program: "sz = uint64([4 1]);\nC = cell(sz);",
        display_output: Some("C is a 4-by-1 cell array"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(C), [4 1]));\nassert(isequal(C{4}, []));",
        },
    },
    BuiltinExample {
        id: "like-logical",
        title: "Select an empty element representation",
        program: "prototype = false(2, 3);\nC = cell(2, 'like', prototype);",
        display_output: Some("Each cell contains an empty logical array"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source:
                "assert(isequal(size(C), [2 2]));\nassert(islogical(C{1}));\nassert(isempty(C{1}));",
        },
    },
];

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Shapes", paragraphs: &["`cell()` is RunMat's nonconflicting shorthand for an empty 0-by-0 cell array. The form is accepted under either compatibility policy because it does not replace documented behavior.", "`cell(n)` creates an `n`-by-`n` cell array. Separate scalar sizes create an N-dimensional result, while a numeric row vector supplies the complete size. One-element inputs use the square form. Trailing singleton dimensions after dimension two are omitted from the stored shape.", "Negative finite sizes become zero. Fractional, nonfinite, nonnumeric, column-vector, unrepresentable, and overflowing size forms return structured errors before allocation."] },
    BuiltinDocumentationSection { heading: "Initial values", paragraphs: &["Each cell created by a compatible constructor contains an independent empty 0-by-0 double array. Populate cells later with brace assignment.", "RunMat's `like` extension selects the empty element representation and, when no explicit size is present, the outer shape. The extension is rejected under a MATLAB compatibility pin."] },
    BuiltinDocumentationSection { heading: "Execution", paragraphs: &["Cell arrays are host containers and end GPU fusion. In RunMat mode, a resident numeric size control can be gathered from its owning provider before host allocation; MATLAB compatibility mode rejects that extension before provider access."] },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "What does a newly allocated cell contain?", answer: "Every compatible constructor cell contains its own empty 0-by-0 double array." },
    BuiltinDocumentationFaq { question: "Can a cell array have a zero dimension?", answer: "Yes. Any zero size produces an empty cell array with the normalized requested shape." },
    BuiltinDocumentationFaq { question: "Are negative or fractional sizes allowed?", answer: "Finite fractional sizes are invalid. Signed negative sizes become zero. All eight integer classes are decoded exactly, and allocation fails if the element count exceeds platform limits." },
    BuiltinDocumentationFaq { question: "Can size controls reside on the GPU?", answer: "The documented constructor is host-only. In RunMat mode, the named GPU-size extension can gather a resident size control before allocating the host cell array." },
    BuiltinDocumentationFaq { question: "What about N-dimensional cell arrays?", answer: "Supply separate scalar dimensions or a numeric row size vector. Trailing singleton dimensions after dimension two are omitted from the stored shape." },
    BuiltinDocumentationFaq { question: "Does cell copy values from another array?", answer: "No. Size inputs describe only the outer shape; the optional RunMat prototype chooses shape or empty representation, not payload values." },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "cellfun",
        target: BuiltinDocumentationLinkTarget::Builtin("cellfun"),
    },
    BuiltinDocumentationLink {
        label: "cell2mat",
        target: BuiltinDocumentationLinkTarget::Builtin("cell2mat"),
    },
    BuiltinDocumentationLink {
        label: "mat2cell",
        target: BuiltinDocumentationLinkTarget::Builtin("mat2cell"),
    },
    BuiltinDocumentationLink {
        label: "cellstr",
        target: BuiltinDocumentationLinkTarget::Builtin("cellstr"),
    },
    BuiltinDocumentationLink {
        label: "struct",
        target: BuiltinDocumentationLinkTarget::Builtin("struct"),
    },
    BuiltinDocumentationLink {
        label: "gpuArray",
        target: BuiltinDocumentationLinkTarget::Builtin("gpuArray"),
    },
    BuiltinDocumentationLink {
        label: "gather",
        target: BuiltinDocumentationLinkTarget::Builtin("gather"),
    },
];

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/tree/main/crates/runmat-runtime/src/builtins/cells/core/cell") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Shape grammar, exact sizes, prototypes, compatibility, and allocation", location: "builtins::cells::core::cell::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Compiled exact integer size dispatch", location: "runmat-vm/tests/logic.rs::cell_accepts_all_integer_size_classes_through_compiled_dispatch" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Executable catalog examples", location: "scripts/runtime/verify-builtin-examples.mjs" },
    ],
    notes: &["Compatible forms were checked against the public cell reference for MATLAB R2026a."],
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("cell"),
    slug: Some("cell"),
    summary: "Create a cell array whose elements begin as empty arrays.",
    description: "`cell` allocates a host cell array with a requested shape and empty elements.",
    keywords: &[
        "cell",
        "cell array",
        "container",
        "containers",
        "empty",
        "preallocation",
        "integer size",
    ],
    related: &["cellfun", "cell2mat", "mat2cell", "cellstr"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: EVIDENCE,
    introduced: Some("before R2006a"),
    status: Some(BuiltinDocumentationStatus::Stable),
};
