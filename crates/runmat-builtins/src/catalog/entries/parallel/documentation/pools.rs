use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference, BuiltinExample, BuiltinExampleCompatibility, BuiltinExampleHarness,
    BuiltinExampleVerification,
};

const IMPLEMENTATION: &[BuiltinDocumentationLink] = &[BuiltinDocumentationLink {
    label: "Session pool implementation",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/parallel/pool.rs",
    ),
}];
const VERIFICATION: &[BuiltinEvidenceReference] = &[
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::IntegrationTest,
        label: "Native process-pool execution tests",
        location: "crates/runmat-cli/tests/parallel_execution.rs",
    },
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::IntegrationTest,
        label: "Session pool and future lifecycle tests",
        location: "crates/runmat-core/src/tests.rs::parallel_pool_and_future_surface_uses_the_session_execution_service",
    },
];
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: IMPLEMENTATION,
    verification: VERIFICATION,
    notes: &[],
};

const PARPOOL_SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`parpool()` returns the compatible execution pool owned by the current session or creates one using the host's automatic worker count. A numeric argument requests a worker count; a pool-kind string can select a supported host backend.",
        "Native CLI and Desktop hosts use isolated worker processes. Browser hosts use browser workers when available, and remote execution can install a cluster-backed pool. A host rejects pool kinds or worker counts it cannot provide.",
        "Calling `delete(pool)` closes that pool generation and cancels unfinished child work. A later pool has a new generation; futures, distributed values, and other execution handles from the retired generation cannot be reused.",
    ] },
    BuiltinDocumentationSection { heading: "Execution placement", paragraphs: &[
        "The pool owns worker admission and transport, not language semantics. Workers consume versioned executable regions and typed values produced by the compiler; they do not reparse source or resolve functions from an ambient path.",
    ] },
];
const PARPOOL_EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "explicit-workers",
        title: "Create a two-worker process pool",
        program: "pool = parpool(2);\nworkers = pool.NumWorkers",
        display_output: Some("workers = 2"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Native,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(workers == 2);\ndelete(pool);",
        },
    },
    BuiltinExample {
        id: "reuse-compatible-pool",
        title: "Reuse the session's compatible pool",
        program: "first = parpool(2);\nsecond = parpool(2)",
        display_output: Some("first and second identify the same pool generation"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Native,
        fixture: crate::BuiltinExampleFixture::None,
        requirements: crate::BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source:
                "assert(first.ID == second.ID);\nassert(second.NumWorkers == 2);\ndelete(first);",
        },
    },
];
const PARPOOL_FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "What worker count does `parpool()` use?", answer: "It asks the active host for its automatic worker count and admission policy." },
    BuiltinDocumentationFaq { question: "Does a repeated call create another pool?", answer: "A compatible request returns the session's existing pool. An incompatible size request replaces it with a new generation." },
    BuiltinDocumentationFaq { question: "What happens when a pool is deleted?", answer: "The generation closes, unfinished work is cancelled, and handles owned by that generation become stale." },
    BuiltinDocumentationFaq { question: "Does the web runtime support pools?", answer: "Yes when the browser host provides worker support. The same typed execution contracts are used across native and browser hosts." },
];

const GCP_SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`gcp()` returns the session's current execution pool and creates an automatic pool when none exists. `gcp(\"nocreate\")` performs inspection only and returns an empty matrix when the session has no active pool.",
        "The returned handle identifies one pool generation. Closing or resizing the pool retires that handle together with futures and distributed values owned by it.",
    ] },
    BuiltinDocumentationSection { heading: "Execution placement", paragraphs: &[
        "`gcp` reads session execution state. It does not itself schedule work, launch provider kernels, or move numeric data.",
    ] },
];
const GCP_EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample { id: "nocreate-empty", title: "Inspect without creating a pool", program: "pool = gcp(\"nocreate\")", display_output: Some("pool = [] when the session has no active pool"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Native, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(isempty(pool));" } },
    BuiltinExample { id: "inspect-current", title: "Read the current pool generation", program: "created = parpool(2);\ncurrent = gcp(\"nocreate\")", display_output: Some("current identifies created"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Native, fixture: crate::BuiltinExampleFixture::None, requirements: crate::BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(current.ID == created.ID);\nassert(current.NumWorkers == 2);\ndelete(created);" } },
];
const GCP_FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Does `gcp()` create a pool?",
        answer: "Yes when no pool exists. Use `gcp(\"nocreate\")` for inspection without creation.",
    },
    BuiltinDocumentationFaq {
        question: "What does `nocreate` return without a pool?",
        answer: "It returns an empty matrix.",
    },
    BuiltinDocumentationFaq {
        question: "Is the pool global to every RunMat process?",
        answer: "No. It is owned by the current RunMat session and its execution service.",
    },
    BuiltinDocumentationFaq {
        question: "Can a stale pool handle become current again?",
        answer:
            "No. Closing or resizing creates a generation boundary; retired handles remain invalid.",
    },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "Parallel execution",
        target: BuiltinDocumentationLinkTarget::Documentation("/docs/runtime/execution/parallel"),
    },
    BuiltinDocumentationLink {
        label: "parfeval",
        target: BuiltinDocumentationLinkTarget::Builtin("parfeval"),
    },
    BuiltinDocumentationLink {
        label: "fetchOutputs",
        target: BuiltinDocumentationLinkTarget::Builtin("fetchOutputs"),
    },
];

pub(in super::super) const PARPOOL_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("parpool"), slug: Some("parpool"), summary: "Create or return the execution pool for the current session.",
    description: "`parpool` admits a host-supported worker pool and returns its generation-fenced session handle.",
    keywords: &["parallel", "pool", "workers", "parpool", "processes", "browser workers"],
    related: &["fetchOutputs", "gcp", "parfeval"], sections: PARPOOL_SECTIONS,
    examples: PARPOOL_EXAMPLES, example_exemption: None, faqs: PARPOOL_FAQS, links: LINKS,
    media: &[], evidence: EVIDENCE, introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};

pub(in super::super) const GCP_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("gcp"), slug: Some("gcp"), summary: "Return or inspect the current session execution pool.",
    description: "`gcp` reads the pool owned by the current RunMat session; the `nocreate` option avoids creating one.",
    keywords: &["parallel", "pool", "current", "gcp", "nocreate", "session"],
    related: &["fetchOutputs", "parfeval", "parpool"], sections: GCP_SECTIONS,
    examples: GCP_EXAMPLES, example_exemption: None, faqs: GCP_FAQS, links: LINKS,
    media: &[], evidence: EVIDENCE, introduced: None, status: Some(BuiltinDocumentationStatus::Stable),
};
