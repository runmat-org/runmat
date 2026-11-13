use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Grouping variables",
        paragraphs: &[
            "A grouping input can contain real numeric or logical values, strings, a cell array of character vectors, categorical values, datetime values, durations, or calendar durations. Multiple inputs define groups from row-wise tuples. Every input must have the same orientation and size.",
            "Table input uses every variable as a grouping variable. The two-output table form returns one identifier table with the same variable names and representations as the input table.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Ordering and missing observations",
        paragraphs: &[
            "Group numbers correspond to sorted unique values or sorted unique tuples. `G` contains one-based doubles from 1 through the number of observed groups.",
            "A missing string, empty character vector in a cell array, `NaN`, `NaT`, undefined categorical observation, or missing calendar duration receives `NaN` in `G` and does not appear in an identifier output.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "RunMat extensions",
        paragraphs: &[
            "RunMat mode can group each column of a matrix, accept a table variable selector, accept timetable input, and materialize provider-resident grouping data. These forms are rejected when the project compatibility mode is pinned to MATLAB.",
            "Fixed-width integers are compared from native storage. Identifier outputs preserve their source class and do not convert wide values through double.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "sorted-text",
        title: "Create group numbers from text",
        program: "[G, ID] = findgroups([\"b\"; \"a\"; \"b\"])",
        display_output: Some("G = [2; 1; 2], ID = [\"a\"; \"b\"]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(G, [2; 1; 2]));\nassert(isequal(ID, [\"a\"; \"b\"]));" },
    },
    BuiltinExample {
        id: "exact-integers",
        title: "Preserve wide integer identifiers",
        program: "A = uint64([0x0020000000000001u64; 0x0020000000000000u64; 0x0020000000000001u64]);\n[G, ID] = findgroups(A)",
        display_output: Some("G = [2; 1; 2]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(G, [2; 1; 2]));\nassert(isa(ID, \"uint64\"));\nassert(isequal(ID, uint64([0x0020000000000000u64; 0x0020000000000001u64])));" },
    },
    BuiltinExample {
        id: "tuple-groups",
        title: "Group combinations from multiple variables",
        program: "[G, ID1, ID2] = findgroups([\"a\"; \"a\"; \"b\"], [2; 1; 2])",
        display_output: Some("G = [2; 1; 3]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(G, [2; 1; 3]));\nassert(isequal(ID1, [\"a\"; \"a\"; \"b\"]));\nassert(isequal(ID2, [1; 2; 2]));" },
    },
    BuiltinExample {
        id: "missing-observations",
        title: "Leave missing observations ungrouped",
        program: "[G, ID] = findgroups([3; NaN; 1; 3])",
        display_output: Some("G = [2; NaN; 1; 2]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequaln(G, [2; NaN; 1; 2]));\nassert(isequal(ID, [1; 3]));" },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Does G have an integer class?",
        answer: "No. G is a double array containing positive integer-valued group numbers and NaN for missing observations. Identifier outputs retain their input classes.",
    },
    BuiltinDocumentationFaq {
        question: "How are groups ordered?",
        answer: "findgroups sorts the unique values or tuples. The numeric group numbers in G refer to that sorted identifier order.",
    },
    BuiltinDocumentationFaq {
        question: "Can findgroups consume GPU-resident values?",
        answer: "RunMat mode can materialize resident grouping values on the host before grouping. Explicit resident input is rejected under the MATLAB compatibility pin because the compatible surface does not document that form.",
    },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "splitapply",
        target: BuiltinDocumentationLinkTarget::Builtin("splitapply"),
    },
    BuiltinDocumentationLink {
        label: "groupcounts",
        target: BuiltinDocumentationLinkTarget::Builtin("groupcounts"),
    },
    BuiltinDocumentationLink {
        label: "Compatible findgroups reference",
        target: BuiltinDocumentationLinkTarget::External(
            "https://www.mathworks.com/help/matlab/ref/findgroups.html",
        ),
    },
];

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink {
        label: "Sorted-group runtime",
        target: BuiltinDocumentationLinkTarget::Source(
            "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/grouping/findgroups/mod.rs",
        ),
    }],
    verification: &[BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Ordering, missing values, exact integer identifiers, table forms, compatibility gates, and provider materialization",
        location: "crates/runmat-runtime/src/builtins/array/grouping/findgroups/tests",
    }],
    notes: &[],
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("findgroups"),
    slug: Some("findgroups"),
    summary: "Assign sorted group numbers and return the corresponding identifiers.",
    description: "`findgroups` creates one-based group numbers from one or more grouping variables and can return the sorted identifiers for those groups.",
    keywords: &[
        "findgroups",
        "groups",
        "table",
        "categorical",
        "integer",
        "missing",
    ],
    related: &["splitapply", "groupcounts", "grp2idx"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: EVIDENCE,
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
