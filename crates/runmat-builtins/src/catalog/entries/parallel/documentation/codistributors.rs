use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink {
        label: "Codistributor definitions and resolution",
        target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/parallel/codistributor.rs"),
    }],
    verification: &[
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::IntegrationTest,
            label: "Factory and inspection tests",
            location: "crates/runmat-core/src/tests.rs::codistributor_factory_and_codistributed_inspection_use_canonical_runtime_values",
        },
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::IntegrationTest,
            label: "One-dimensional and block-cyclic layout tests",
            location: "crates/runmat-core/src/tests.rs distributed layout tests",
        },
    ],
    notes: &[],
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "A codistributor is an immutable distribution description. A one-dimensional description records a distribution dimension, one partition length per worker, and an optional global size. A two-dimensional block-cyclic description records a worker grid, block size, row- or column-major grid orientation, and an optional global matrix size.",
            "Constructors may leave properties unresolved. The distribution service resolves omitted values against the array shape and admitted worker count when a distributed value is created or redistributed. Declared partition lengths, grids, dimensions, and global sizes must agree at that boundary.",
            "Codistributor objects contain exact integer structural values. They describe placement but do not own partition payloads, a pool generation, or accelerator buffers.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Execution placement",
        paragraphs: &[
            "Constructing or inspecting a codistributor is local and does not launch workers. Distribution begins only when an operation such as `codistributed` or `redistribute` resolves the description against a pool.",
        ],
    },
];

const CODISTRIBUTOR_EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "default-one-dimensional",
        title: "Create the default one-dimensional description",
        program: "codist = codistributor()",
        display_output: Some("codist is an incomplete codistributor1d object"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Native,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(class(codist) == \"codistributor1d\");\nassert(~isComplete(codist));",
        },
    },
    BuiltinExample {
        id: "block-cyclic-factory",
        title: "Select a block-cyclic description",
        program: "codist = codistributor(\"2dbc\", uint32([1 2]), uint64(4))",
        display_output: Some("codist has worker grid [1 2] and block size 4"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Native,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(class(codist) == \"codistributor2dbc\");\nassert(all(codist.WorkerGrid == uint64([1 2])));\nassert(codist.BlockSize == uint64(4));",
        },
    },
];
const ONE_DIMENSIONAL_EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "complete-columns",
        title: "Describe a complete column partition",
        program: "codist = codistributor1d(uint32(2), uint64([2 3]), uint64([4 5]))",
        display_output: Some("dimension 2 is split into lengths 2 and 3"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Native,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(codist.Dimension == uint64(2));\nassert(all(codist.Partition == uint64([2 3])));\nassert(all(codist.GlobalSize == uint64([4 5])));\nassert(isComplete(codist));",
        },
    },
    BuiltinExample {
        id: "deferred-layout",
        title: "Defer layout resolution",
        program: "codist = codistributor1d()",
        display_output: Some("dimension, partition, and global size are resolved later"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Native,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isempty(codist.Dimension));\nassert(isempty(codist.Partition));\nassert(~isComplete(codist));",
        },
    },
];
const TWO_DIMENSIONAL_EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "complete-grid",
        title: "Describe a complete block-cyclic matrix layout",
        program: "codist = codistributor2dbc(uint32([2 2]), uint64(1), \"col\", uint64([4 4]))",
        display_output: Some("a 2-by-2 column-oriented worker grid with unit blocks"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Native,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(all(codist.WorkerGrid == uint64([2 2])));\nassert(codist.BlockSize == uint64(1));\nassert(codist.Orientation == \"col\");\nassert(isComplete(codist));",
        },
    },
    BuiltinExample {
        id: "default-orientation",
        title: "Use row-oriented worker-grid numbering",
        program: "codist = codistributor2dbc(uint32([1 2]), uint64(8))",
        display_output: Some("Orientation is row"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Native,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(codist.Orientation == \"row\");\nassert(~isComplete(codist));",
        },
    },
];
const COMPLETE_EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "complete",
        title: "Recognize a declared global size",
        program: "codist = codistributor1d(uint32(1), uint64([2 2]), uint64([4 1]));\ntf = isComplete(codist)",
        display_output: Some("tf = true"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Native,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(tf);" },
    },
    BuiltinExample {
        id: "incomplete",
        title: "Recognize a deferred global size",
        program: "tf = isComplete(codistributor1d())",
        display_output: Some("tf = false"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Native,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(~tf);" },
    },
];
const IS_CODISTRIBUTED_EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "distributed-value",
        title: "Test a distributed value without gathering it",
        program: "value = distributed(uint16([1 2]));\ntf = iscodistributed(value)",
        display_output: Some("tf = true"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Native,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(tf);\nassert(isequal(gather(value), uint16([1 2])));\ndelete(gcp(\"nocreate\"));",
        },
    },
    BuiltinExample {
        id: "ordinary-value",
        title: "Test an ordinary array",
        program: "tf = iscodistributed(uint16([1 2]))",
        display_output: Some("tf = false"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Native,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(~tf);" },
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Does construction start a pool?", answer: "No. A codistributor is a local immutable description; distribution resolves it against a pool later." },
    BuiltinDocumentationFaq { question: "What makes a codistributor complete?", answer: "Its `GlobalSize` property is present and valid. Other properties may still be resolved or validated when used." },
    BuiltinDocumentationFaq { question: "Are dimensions and partitions stored as doubles?", answer: "No. Structural properties use exact unsigned integer values." },
    BuiltinDocumentationFaq { question: "Can a codistributor be reused?", answer: "Yes when its declared shape and layout are valid for the target array and admitted worker count." },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "Parallel execution",
        target: BuiltinDocumentationLinkTarget::Documentation("/docs/runtime/execution/parallel"),
    },
    BuiltinDocumentationLink {
        label: "redistribute",
        target: BuiltinDocumentationLinkTarget::Builtin("redistribute"),
    },
];

macro_rules! documentation {
    ($constant:ident, $title:literal, $summary:literal, $description:literal, $keywords:expr, $related:expr, $examples:expr) => {
        pub(in super::super) const $constant: BuiltinDocumentation = BuiltinDocumentation {
            authority: BuiltinDocumentationAuthority::Catalog,
            title: Some($title),
            slug: Some($title),
            summary: $summary,
            description: $description,
            keywords: $keywords,
            related: $related,
            sections: SECTIONS,
            examples: $examples,
            example_exemption: None,
            faqs: FAQS,
            links: LINKS,
            media: &[],
            evidence: EVIDENCE,
            introduced: None,
            status: Some(BuiltinDocumentationStatus::Stable),
        };
    };
}

documentation!(CODISTRIBUTOR_DOCUMENTATION, "codistributor", "Create a one-dimensional or block-cyclic distribution description.", "`codistributor` selects a supported immutable distribution scheme through one factory entry point.", &["parallel", "distributed", "codistributor", "layout", "factory"], &["codistributor1d", "codistributor2dbc", "isComplete", "redistribute"], CODISTRIBUTOR_EXAMPLES);
documentation!(
    CODISTRIBUTOR_1D_DOCUMENTATION,
    "codistributor1d",
    "Create a one-dimensional distribution description.",
    "`codistributor1d` describes how one array dimension is partitioned across workers.",
    &[
        "parallel",
        "distributed",
        "codistributor",
        "1d",
        "partition"
    ],
    &[
        "codistributor",
        "codistributor2dbc",
        "globalIndices",
        "isComplete",
        "redistribute"
    ],
    ONE_DIMENSIONAL_EXAMPLES
);
documentation!(CODISTRIBUTOR_2DBC_DOCUMENTATION, "codistributor2dbc", "Create a two-dimensional block-cyclic distribution description.", "`codistributor2dbc` describes a matrix distributed in blocks across a two-dimensional worker grid.", &["parallel", "distributed", "codistributor", "2dbc", "block cyclic"], &["codistributor", "codistributor1d", "globalIndices", "isComplete", "redistribute"], TWO_DIMENSIONAL_EXAMPLES);
documentation!(IS_COMPLETE_DOCUMENTATION, "isComplete", "Test whether a codistributor declares its global size.", "`isComplete` validates a codistributor and reports whether its global size is already present.", &["parallel", "distributed", "codistributor", "complete", "global size"], &["codistributor", "codistributor1d", "codistributor2dbc"], COMPLETE_EXAMPLES);
documentation!(
    IS_CODISTRIBUTED_DOCUMENTATION,
    "iscodistributed",
    "Test whether a value is a codistributed array.",
    "`iscodistributed` inspects the value kind without materializing or reading its partitions.",
    &[
        "parallel",
        "distributed",
        "codistributed",
        "predicate",
        "inspection"
    ],
    &["codistributed", "distributed", "gather", "getCodistributor"],
    IS_CODISTRIBUTED_EXAMPLES
);
