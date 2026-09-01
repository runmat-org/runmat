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
            label: "Execution-context projection",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/parallel/current.rs"),
        },
        BuiltinDocumentationLink {
            label: "Builtin entry points",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/parallel/current.rs"),
        },
    ],
    verification: &[
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Scoped assignment and driver-context tests",
            location: "crates/runmat-runtime/src/parallel/current.rs tests",
        },
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::IntegrationTest,
            label: "Native process-worker context test",
            location: "crates/runmat-cli/tests/parallel_execution.rs::parfor_executes_in_process_workers_and_assembles_results",
        },
    ],
    notes: &[],
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "Current-execution functions project typed scheduler identity into read-only handle objects. Task objects expose `ID`, `AttemptID`, `PoolID`, `WorkerID`, and `State`; worker objects expose `ID`, `PoolID`, and `Backend`; job objects expose `ID`.",
            "An ordinary driver call has no task, worker, or durable-job assignment and returns `[]`. A worker task has task and worker assignments. A job object is present only when the execution belongs to a submitted durable job.",
            "Execution identity is dynamically scoped. Nested work sees its own assignment, and leaving that scope restores the previous one without leaking worker identity into the driver.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Execution placement",
        paragraphs: &[
            "These functions read the active runtime context. They do not schedule work or contact the pool; the returned IDs and backend name describe the assignment already admitted by the execution service.",
        ],
    },
];

const TASK_EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "driver-is-empty",
        title: "Detect ordinary driver execution",
        program: "task = getCurrentTask()",
        display_output: Some("task = []"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Native,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isempty(task));",
        },
    },
    BuiltinExample {
        id: "parfor-task",
        title: "Inspect task identity inside parfor",
        program: "pool = parpool(2);\nseen = false(1, 4);\nparfor (index = 1:4, 2)\n  task = getCurrentTask();\n  seen(index) = ~isempty(task) && task.ID == task.ID;\nend",
        display_output: Some("seen = [true true true true]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Native,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(all(seen));\ndelete(pool);",
        },
    },
];
const WORKER_EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "driver-is-empty",
        title: "Detect ordinary driver execution",
        program: "worker = getCurrentWorker()",
        display_output: Some("worker = []"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Native,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isempty(worker));",
        },
    },
    BuiltinExample {
        id: "parfor-worker",
        title: "Inspect worker identity inside parfor",
        program: "pool = parpool(2);\nseen = false(1, 4);\nparfor (index = 1:4, 2)\n  worker = getCurrentWorker();\n  seen(index) = ~isempty(worker) && worker.ID == worker.ID;\nend",
        display_output: Some("seen = [true true true true]"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Native,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(all(seen));\ndelete(pool);",
        },
    },
];
const JOB_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "driver-is-empty",
    title: "Check for a durable job assignment",
    program: "job = getCurrentJob()",
    display_output: Some("job = [] in ordinary driver execution"),
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::Native,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isempty(job));",
    },
}];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Why does the function return an empty matrix?",
        answer: "The current invocation has no assignment of that kind, as in ordinary driver execution.",
    },
    BuiltinDocumentationFaq {
        question: "Are the returned objects mutable scheduler records?",
        answer: "No. They are read-only projections of the active typed execution assignment.",
    },
    BuiltinDocumentationFaq {
        question: "Does reading the context contact a worker or cluster?",
        answer: "No. It reads identity already installed in the current runtime scope.",
    },
    BuiltinDocumentationFaq {
        question: "Can nested execution leak its identity?",
        answer: "No. Assignment scopes restore the prior identity when they exit.",
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

macro_rules! current_documentation {
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

current_documentation!(GET_CURRENT_TASK_DOCUMENTATION, "getCurrentTask", "Return the task assigned to the current invocation.", "`getCurrentTask` returns a read-only task identity inside scheduled work and `[]` when no task is active.", &["parallel", "current", "task", "assignment", "attempt"], &["getCurrentJob", "getCurrentWorker", "parfeval", "parpool"], TASK_EXAMPLES);
current_documentation!(GET_CURRENT_WORKER_DOCUMENTATION, "getCurrentWorker", "Return the worker assigned to the current invocation.", "`getCurrentWorker` returns a read-only worker identity inside scheduled work and `[]` when no worker is active.", &["parallel", "current", "worker", "pool", "backend"], &["getCurrentJob", "getCurrentTask", "parpool"], WORKER_EXAMPLES);
current_documentation!(GET_CURRENT_JOB_DOCUMENTATION, "getCurrentJob", "Return the durable job assigned to the current invocation.", "`getCurrentJob` returns a read-only durable-job identity when one is active and `[]` otherwise.", &["parallel", "current", "job", "assignment", "durable"], &["getCurrentTask", "getCurrentWorker", "parpool"], JOB_EXAMPLES);
