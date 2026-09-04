use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Array and table forms",
        paragraphs: &[
            "For array input, `B` contains one count per sorted group. Request `BG` for the corresponding group labels and `BP` for percentages. A matrix groups rows by its columns; a cell array can hold grouping vectors with different representations.",
            "For table or timetable input, select grouping variables with `groupvars`. The result is a table containing those grouping variables followed by `GroupCount` and `Percent`.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Missing and empty groups",
        paragraphs: &[
            "Missing groups are included by default. Set `IncludeMissingGroups` to false to omit rows with a missing value in any grouping role.",
            "Set `IncludeEmptyGroups` to true to include unobserved categorical levels, logical values, numeric bins, and their possible combinations. Empty groups have count and percentage zero.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Binning",
        paragraphs: &[
            "The current implementation supports explicit numeric edges and positive numeric bin counts for one numeric grouping vector. `IncludedEdge` selects left- or right-inclusive intervals. Multiple bin specifications and temporal bin methods return an explicit unsupported-form error.",
            "Numeric data and edges retain their fixed-width classes during comparison, so adjacent wide integers remain distinct. Bin labels are strings; unbinned labels retain the source representation.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Compatibility and residency",
        paragraphs: &[
            "Automatic provider placement can materialize transparently. Explicit `gpuArray` grouping input and fixed-width integer option controls require RunMat compatibility mode; the equivalent host, double, and logical forms remain available under the MATLAB compatibility pin.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "counts-and-labels",
        title: "Count sorted numeric groups",
        program: "[B, BG, BP] = groupcounts([2; 1; 2; 2])",
        display_output: Some("B = [1; 3], BG = [1; 2], BP = [25; 75]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(B, [1; 3]));\nassert(isequal(BG, [1; 2]));\nassert(isequal(BP, [25; 75]));" },
    },
    BuiltinExample {
        id: "missing-group",
        title: "Include missing observations as a group",
        program: "[B, BG] = groupcounts([2; NaN; 2])",
        display_output: Some("B = [2; 1]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(B, [2; 1]));\nassert(isequaln(BG, [2; NaN]));" },
    },
    BuiltinExample {
        id: "exact-wide-integers",
        title: "Keep wide integer groups distinct",
        program: "A = uint64([0x0020000000000000u64; 0x0020000000000001u64; 0x0020000000000000u64]);\n[B, BG] = groupcounts(A)",
        display_output: Some("B = [2; 1]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(B, [2; 1]));\nassert(isa(BG, \"uint64\"));\nassert(isequal(BG, uint64([0x0020000000000000u64; 0x0020000000000001u64])));" },
    },
    BuiltinExample {
        id: "empty-numeric-bin",
        title: "Retain empty numeric bins",
        program: "[B, BG] = groupcounts([0; 2], [0; 1; 2; 3], \"IncludeEmptyGroups\", true)",
        display_output: Some("B = [1; 0; 1]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(B, [1; 0; 1]));\nassert(numel(BG) == 3);" },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "What determines the order of B and BG?",
        answer: "Unbinned groups follow sorted unique grouping keys. Explicit categories retain their declared category order when empty groups are requested, and numeric bins follow edge order.",
    },
    BuiltinDocumentationFaq {
        question: "What does BP measure?",
        answer: "BP is each group count divided by the number of input observations, multiplied by 100. Empty groups therefore have percentage zero.",
    },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "findgroups",
        target: BuiltinDocumentationLinkTarget::Builtin("findgroups"),
    },
    BuiltinDocumentationLink {
        label: "groupsummary",
        target: BuiltinDocumentationLinkTarget::Builtin("groupsummary"),
    },
    BuiltinDocumentationLink {
        label: "Compatible groupcounts reference",
        target: BuiltinDocumentationLinkTarget::External(
            "https://www.mathworks.com/help/matlab/ref/double.groupcounts.html",
        ),
    },
];

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink {
        label: "Group-count runtime",
        target: BuiltinDocumentationLinkTarget::Source(
            "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/grouping/groupcounts/mod.rs",
        ),
    }],
    verification: &[BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Array, table, missing, empty, binning, exact-integer, provider, and compatibility behavior",
        location: "crates/runmat-runtime/src/builtins/array/grouping/groupcounts/tests",
    }],
    notes: &[],
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("groupcounts"),
    slug: Some("groupcounts"),
    summary: "Count observations in each group.",
    description: "`groupcounts` returns sorted group counts and optional labels and percentages for array data, or a summary table for table data.",
    keywords: &["groupcounts", "groups", "count", "table", "bins"],
    related: &["findgroups", "groupsummary", "splitapply"],
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
