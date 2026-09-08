use crate::*;

use super::examples::EXAMPLES;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Block concatenation", paragraphs: &["`cell2mat(C)` concatenates compatible numeric, logical, complex, or character blocks from a cell array. The cell grid selects each block's position in the result.", "Blocks in each cell-grid row must agree on row extent, blocks in each grid column must agree on column extent, and higher-dimensional extents must agree throughout the grid. Empty blocks contribute zero extent."] },
    BuiltinDocumentationSection { heading: "Classes and storage", paragraphs: &["The result keeps the content class. Fixed-width integer blocks retain exact storage; compatible mixed integer blocks use the leftmost nonempty integer class and saturating assignment conversion. Scalar doubles cannot be combined with `int64` or `uint64` blocks.", "Strings, nested cells, structs, and objects are not valid contents. Character blocks produce a character array and must remain two-dimensional."] },
    BuiltinDocumentationSection { heading: "Execution", paragraphs: &["Cell arrays are host containers. Resident numeric elements are gathered before concatenation, and the result is host-resident. `cell2mat` ends an active fusion group.", "An empty cell array returns a 0-by-0 double array."] },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Which cell contents can cell2mat concatenate?", answer: "Numeric, logical, complex, and character arrays are supported. Every nonempty block must belong to one compatible fundamental kind; nested cells, strings, structs, and objects are rejected." },
    BuiltinDocumentationFaq { question: "Must every block have the same shape?", answer: "No. Widths may differ between cell-grid columns and heights may differ between rows. Extents must agree wherever blocks share the corresponding grid coordinate." },
    BuiltinDocumentationFaq { question: "What happens when cells contain empty arrays?", answer: "Empty blocks contribute zero extent along their tiling dimension while retaining the selected content class. A completely empty cell array returns a 0-by-0 double array." },
    BuiltinDocumentationFaq { question: "Does cell2mat preserve GPU residency?", answer: "Not currently. Resident elements gather and the concatenated result is allocated on the host." },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "cell",
        target: BuiltinDocumentationLinkTarget::Builtin("cell"),
    },
    BuiltinDocumentationLink {
        label: "mat2cell",
        target: BuiltinDocumentationLinkTarget::Builtin("mat2cell"),
    },
    BuiltinDocumentationLink {
        label: "num2cell",
        target: BuiltinDocumentationLinkTarget::Builtin("num2cell"),
    },
    BuiltinDocumentationLink {
        label: "cellfun",
        target: BuiltinDocumentationLinkTarget::Builtin("cellfun"),
    },
    BuiltinDocumentationLink {
        label: "cellstr",
        target: BuiltinDocumentationLinkTarget::Builtin("cellstr"),
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
    implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/tree/main/crates/runmat-runtime/src/builtins/cells/core/cell2mat") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Class, shape, concatenation, error, and provider behavior", location: "builtins::cells::core::cell2mat::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Catalog inference and diagnostics", location: "catalog::entries::cells::core::cell2mat::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Executable catalog examples", location: "scripts/runtime/verify-builtin-examples.mjs" },
    ],
    notes: &["Compatible behavior was checked against the public cell2mat reference for MATLAB R2026a."],
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("cell2mat"),
    slug: Some("cell2mat"),
    summary: "Concatenate compatible cell-array blocks into one dense array.",
    description: "`cell2mat` assembles array blocks using the cell grid as a block layout.",
    keywords: &[
        "cell2mat",
        "cell array",
        "block concatenation",
        "matrix conversion",
    ],
    related: &["cell", "mat2cell", "num2cell"],
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
