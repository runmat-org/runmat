use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Group ordering",
        paragraphs: &[
            "Numeric and logical levels use ascending order. Character rows, strings, cell arrays of character vectors, datetime values, and duration values use the order in which each level first appears. Categorical levels follow the category order, including categories with no observations.",
            "Missing observations receive `NaN` in `g` and do not create a text, numeric, datetime, or duration level. Undefined categorical observations receive `NaN`; the declared category list still determines the categorical level outputs.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Outputs",
        paragraphs: &[
            "`g` is a double column vector of one-based group indices. `gN` is a cell array of character vectors containing the printable group names. `gL` contains the same levels in the input representation, except that string input produces a cell array of character vectors.",
            "For a character matrix, each row is one label and `gL(g,:)` reconstructs nonmissing input rows. For other representations, `gL(g)` reconstructs the corresponding nonmissing input observations.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Execution",
        paragraphs: &[
            "Grouping compares fixed-width integers from their native storage, including `int64` and `uint64` values that cannot be represented exactly as double. Supported provider-resident numeric input is grouped through the runtime's materialization boundary; numeric outputs are restored through the input's provider when that provider can represent them.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "text-first-appearance",
        title: "Index text levels by first appearance",
        program: "[g, gN, gL] = grp2idx([\"b\"; \"a\"; \"b\"])",
        display_output: Some("g = [1; 2; 1]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(g, [1; 2; 1]));\nassert(isequal(gN, {'b'; 'a'}));\nassert(isequal(gL, {'b'; 'a'}));" },
    },
    BuiltinExample {
        id: "numeric-sorted",
        title: "Sort numeric group levels",
        program: "[g, gN, gL] = grp2idx(uint64([9; 2; 9]))",
        display_output: Some("g = [2; 1; 2], gL = uint64([2; 9])"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(g, [2; 1; 2]));\nassert(isa(gL, \"uint64\"));\nassert(isequal(gL, uint64([2; 9])));" },
    },
    BuiltinExample {
        id: "missing-values",
        title: "Leave missing observations ungrouped",
        program: "[g, gN, gL] = grp2idx([3; NaN; 1; 3])",
        display_output: Some("g = [2; NaN; 1; 2]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isequaln(g, [2; NaN; 1; 2]));\nassert(isequal(gL, [1; 3]));" },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Why do text and numeric inputs produce different level orders?", answer: "The grouping-variable contract sorts numeric and logical levels, while text-like and temporal levels retain first appearance. Categorical input follows its declared category order." },
    BuiltinDocumentationFaq { question: "Does grp2idx preserve wide integers?", answer: "Yes. Integer keys are compared without conversion to double, and `gL` preserves the input integer class and exact values." },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "findgroups",
        target: BuiltinDocumentationLinkTarget::Builtin("findgroups"),
    },
    BuiltinDocumentationLink {
        label: "Compatible grp2idx reference",
        target: BuiltinDocumentationLinkTarget::External(
            "https://www.mathworks.com/help/stats/grp2idx.html",
        ),
    },
];

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Grouping-index runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/array/grouping/grp2idx/mod.rs") }],
    verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Ordering, missing values, representation preservation, integer exactness, and provider behavior", location: "crates/runmat-runtime/src/builtins/array/grouping/grp2idx/tests" }],
    notes: &[],
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("grp2idx"),
    slug: Some("grp2idx"),
    summary: "Create a one-based index from a grouping variable.",
    description: "`grp2idx(s)` assigns a one-based index to each nonmissing observation and returns the ordered group names and levels on request.",
    keywords: &["grp2idx", "groups", "index", "categorical", "statistics"],
    related: &["findgroups"],
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
