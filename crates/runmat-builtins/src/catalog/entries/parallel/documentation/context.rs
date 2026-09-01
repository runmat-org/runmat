use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[
        BuiltinDocumentationLink {
            label: "SPMD context implementation",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/parallel/spmd_context.rs"),
        },
        BuiltinDocumentationLink {
            label: "SPMD execution",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-vm/src/interpreter/dispatch/parallel/spmd.rs"),
        },
    ],
    verification: &[
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::IntegrationTest,
            label: "Typed multi-rank SPMD execution tests",
            location: "crates/runmat-core/src/tests.rs parallel SPMD tests",
        },
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::IntegrationTest,
            label: "Native process-worker SPMD test",
            location: "crates/runmat-cli/tests/parallel_execution.rs::spmd_executes_as_one_typed_multi_process_gang",
        },
    ],
    notes: &[],
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "Each admitted worker in an `spmd` region has a stable one-based rank and observes the same gang size. `spmdIndex` and its `labindex` alias return the rank; `spmdSize` and its `numlabs` alias return the size.",
            "The values are ordinary scalar doubles inside the worker. When assigned in an `spmd` region, the driver receives one value per worker in a `Composite`, like other worker-local outputs.",
            "Outside an active `spmd` gang, the serial context is rank 1 of size 1. This makes helper functions that inspect the context usable in both serial and SPMD execution.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Execution placement",
        paragraphs: &[
            "The values come from the active typed collective context. Reading them does not communicate with another worker, move array data, or invoke an accelerator provider.",
        ],
    },
];

const INDEX_EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "two-worker-ranks",
        title: "Read each worker's one-based rank",
        program: "pool = parpool(2);\nspmd\n  rank = spmdIndex();\nend",
        display_output: Some("rank contains 1 and 2 in worker order"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Native,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(rank{1} == 1);\nassert(rank{2} == 2);\ndelete(pool);",
        },
    },
    BuiltinExample {
        id: "serial-rank",
        title: "Read the serial context",
        program: "rank = spmdIndex()",
        display_output: Some("rank = 1"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Native,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(rank == 1);",
        },
    },
];
const SIZE_EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "two-worker-size",
        title: "Read the admitted gang size",
        program: "pool = parpool(2);\nspmd\n  count = spmdSize();\nend",
        display_output: Some("each worker reports count = 2"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Native,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(count{1} == 2);\nassert(count{2} == 2);\ndelete(pool);",
        },
    },
    BuiltinExample {
        id: "serial-size",
        title: "Read the serial context size",
        program: "count = spmdSize()",
        display_output: Some("count = 1"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Native,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(count == 1);",
        },
    },
];
const LABINDEX_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "lab-ranks",
    title: "Use the compatible lab-rank name",
    program: "pool = parpool(2);\nspmd\n  rank = labindex;\nend",
    display_output: Some("rank contains 1 and 2 in worker order"),
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::Native,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(rank{1} == 1);\nassert(rank{2} == 2);\ndelete(pool);",
    },
}];
const NUMLABS_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "lab-count",
    title: "Use the compatible lab-count name",
    program: "pool = parpool(2);\nspmd\n  count = numlabs;\nend",
    display_output: Some("each worker reports count = 2"),
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::Native,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(count{1} == 2);\nassert(count{2} == 2);\ndelete(pool);",
    },
}];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Are worker indices zero-based or one-based?",
        answer: "They are one-based and range from 1 through the active gang size.",
    },
    BuiltinDocumentationFaq {
        question: "Can the rank change during one SPMD region?",
        answer: "No. Admission fixes each worker's rank for the lifetime of that region.",
    },
    BuiltinDocumentationFaq {
        question: "Do these functions synchronize workers?",
        answer: "No. They only read the current worker's collective context.",
    },
    BuiltinDocumentationFaq {
        question: "What do they return outside SPMD?",
        answer: "The serial context reports rank 1 and size 1.",
    },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "Parallel execution",
        target: BuiltinDocumentationLinkTarget::Documentation("/docs/runtime/execution/parallel"),
    },
    BuiltinDocumentationLink {
        label: "parpool",
        target: BuiltinDocumentationLinkTarget::Builtin("parpool"),
    },
];

macro_rules! context_documentation {
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

context_documentation!(
    SPMD_INDEX_DOCUMENTATION,
    "spmdIndex",
    "Return the current SPMD worker's one-based rank.",
    "`spmdIndex` reads the stable rank assigned to the current worker in an SPMD gang.",
    &["parallel", "spmd", "worker", "rank", "index"],
    &["labindex", "numlabs", "spmdSize"],
    INDEX_EXAMPLES
);
context_documentation!(
    SPMD_SIZE_DOCUMENTATION,
    "spmdSize",
    "Return the number of workers in the current SPMD gang.",
    "`spmdSize` reads the admitted size of the current SPMD gang.",
    &["parallel", "spmd", "workers", "gang", "size"],
    &["labindex", "numlabs", "spmdIndex"],
    SIZE_EXAMPLES
);
context_documentation!(
    LABINDEX_DOCUMENTATION,
    "labindex",
    "Return the current SPMD worker's one-based lab index.",
    "`labindex` is the compatible lab-oriented name for the rank returned by `spmdIndex`.",
    &["parallel", "spmd", "lab", "rank", "index"],
    &["numlabs", "spmdIndex", "spmdSize"],
    LABINDEX_EXAMPLES
);
context_documentation!(
    NUMLABS_DOCUMENTATION,
    "numlabs",
    "Return the number of labs in the current SPMD gang.",
    "`numlabs` is the compatible lab-oriented name for the size returned by `spmdSize`.",
    &["parallel", "spmd", "labs", "gang", "size"],
    &["labindex", "spmdIndex", "spmdSize"],
    NUMLABS_EXAMPLES
);
