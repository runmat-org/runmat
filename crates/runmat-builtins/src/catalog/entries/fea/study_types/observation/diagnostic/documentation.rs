use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`kind` is a `fea.DiagnosticKind`: convergence, balance, constraint, contact, adaptation, resource, or executionEquivalence. `sampling` and `retention` are the typed policies; there is no storage, cost, or target argument.",
        "`id` must be non-empty and unique among the step's observations. Sampling must match the step's independent domain, and a bounded retention count must be nonzero.",
    ] },
    BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
        "Add the diagnostic to the step's observation list passed to `fea.ProcedureStep.configure`. Convergence diagnostics with `everyAcceptedIncrement` sampling and `latestSamples` retention are a common way to watch a nonlinear solve.",
    ] },
    BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
        "`RunMat:fea:InvalidInput` when `id` is empty, or `kind`, `sampling`, or `retention` is not the expected typed value.",
        "`RunMat:fea:Internal` if the request cannot be converted into a RunMat object.",
        "`RunMat:fea:TooManyOutputs` when more than one output is requested.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "convergence",
        title: "Record convergence at every accepted increment",
        program: "convergence = fea.Observation.diagnostic(\"newton_convergence\", ...\n    fea.DiagnosticKind.convergence(), fea.SamplingPolicy.everyAcceptedIncrement(), ...\n    fea.RetentionPolicy.latestSamples(200));",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(convergence), 'fea.Observation'));\nassert(strcmp(convergence.value.diagnostic.diagnostic, 'convergence'));\nassert(convergence.value.diagnostic.retention.latest_samples.count == 200);" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.Observation.diagnostic",
        summary: "Request solver diagnostics such as convergence or balance records.",
        description: "`fea.Observation.diagnostic(id, kind, sampling, retention)` builds a `fea.Observation` that records one kind of solver diagnostic for a step.",
        keywords: &["fea", "observation", "diagnostic", "convergence", "balance", "solver"],
        related: &["fea.Observation.monitor", "fea.SamplingPolicy.everyAcceptedIncrement", "fea.RetentionPolicy.latestSamples", "fea.ProcedureStep.configure"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "Can a monitor read a diagnostic?", answer: "No. A monitor's source must be a field, history, probe, or reduction in the same step." }],
        links: &[
            BuiltinDocumentationLink { label: "Procedure sequences, observations, and accuracy", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/study-contracts") },
            BuiltinDocumentationLink { label: "Outputs, fields, and observations", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/language/outputs-fields-and-observations") },
        ],
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/observations/requests.rs") }],
        verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Catalog inference contract", location: "crates/runmat-builtins/src/catalog/entries/fea/study_types/observation/tests.rs::observation_constructors_follow_their_positional_contract" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Convergence diagnostics", location: "crates/runmat-analysis/fea/src/tests.rs::convergence_diagnostics_are_emitted" },
        ],
        notes: &[],
    });
