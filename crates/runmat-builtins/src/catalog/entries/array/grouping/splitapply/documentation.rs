use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Groups and data orientation",
        paragraphs: &[
            "`G` contains the consecutive positive integers 1 through N. `NaN` observations are omitted. A column `G` splits data by rows; a row `G` splits array data by columns.",
            "Each data input must match `G` along the split dimension. For table input, each table variable becomes a separate argument to the group function.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Callback outputs",
        paragraphs: &[
            "RunMat invokes `func` once for each group in numeric order. Results for each requested callback output are concatenated vertically and must have compatible classes and trailing dimensions.",
            "A callback can return multiple outputs when the splitapply call requests them. Every group must return the requested number of outputs.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Typed values and execution",
        paragraphs: &[
            "Array slicing preserves numeric class, complex storage, logical values, strings, cells, and supported object representations. Fixed-width integer group-number storage is available in RunMat compatibility mode.",
            "Provider-resident inputs are materialized before host callback execution. The same behavior is available in native and WebAssembly runtimes when the callback itself is available there.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "sum-by-group",
        title: "Sum values by group",
        program: "Y = splitapply(@sum, [1; 2; 3; 4], [2; 1; 2; 1])",
        display_output: Some("Y = [6; 4]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(Y, [6; 4]));",
        },
    },
    BuiltinExample {
        id: "omit-missing-group",
        title: "Omit data marked with NaN",
        program: "Y = splitapply(@sum, [10; 20; 30], [1; NaN; 2])",
        display_output: Some("Y = [10; 30]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(Y, [10; 30]));",
        },
    },
    BuiltinExample {
        id: "row-oriented-groups",
        title: "Split matrix columns with a row group vector",
        program: "X = [1 2 3 4; 10 20 30 40];\nY = splitapply(@sum, X, [1 2 1 2])",
        display_output: Some("Y = [11 33; 22 44]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(Y, [11 33; 22 44]));",
        },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Can group numbers have gaps?",
        answer: "No. Nonmissing group numbers must include every positive integer from 1 through the largest group number.",
    },
    BuiltinDocumentationFaq {
        question: "What happens to rows whose group number is NaN?",
        answer: "They are omitted and are not passed to the group function.",
    },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "findgroups",
        target: BuiltinDocumentationLinkTarget::Builtin("findgroups"),
    },
    BuiltinDocumentationLink {
        label: "groupcounts",
        target: BuiltinDocumentationLinkTarget::Builtin("groupcounts"),
    },
    BuiltinDocumentationLink {
        label: "Compatible splitapply reference",
        target: BuiltinDocumentationLinkTarget::External(
            "https://www.mathworks.com/help/matlab/ref/splitapply.html",
        ),
    },
];

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink {
        label: "Grouped callback runtime",
        target: BuiltinDocumentationLinkTarget::Source(
            "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/grouping/splitapply/mod.rs",
        ),
    }],
    verification: &[BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Group validation, orientation, class preservation, callback output assembly, compatibility policy, and provider materialization",
        location: "crates/runmat-runtime/src/builtins/array/grouping/splitapply/tests",
    }],
    notes: &[],
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("splitapply"),
    slug: Some("splitapply"),
    summary: "Split data into groups and apply a function to each group.",
    description: "`splitapply` partitions one or more data inputs using consecutive group numbers, invokes a function once per group, and concatenates its results.",
    keywords: &[
        "splitapply",
        "groups",
        "function",
        "callback",
        "table",
        "apply",
    ],
    related: &["findgroups", "groupcounts", "grp2idx"],
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
