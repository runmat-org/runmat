use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "Takes no arguments. Only accepted increments are sampled; rejected or cut-back attempts never appear in the history.",
        "Valid for time, frequency, load-factor, and response-spectrum domains. Model validation rejects it for static, modal-extraction, prestressed-modal, and eigenvalue-buckling procedures, which have no increments.",
        "The number of samples is decided by the solver. Bound the stored size with `fea.RetentionPolicy.latestSamples` or `explicitSamples` when the increment count is large.",
    ] },
    BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
        "Pass the policy as the `sampling` argument (11th) of `fea.Observation.field`, `history`, `probe`, or `reduction`, or as the third argument of `fea.Observation.diagnostic`. Model validation checks it against the independent domain of the step that owns the observation (`fea.IndependentDomain.*`).",
        "Typical uses are convergence diagnostics and nonlinear load-path histories.",
    ] },
    BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
        "`RunMat:fea:Internal` if the policy cannot be converted into a RunMat object.",
        "`RunMat:fea:TooManyOutputs` when more than one output is requested.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "convergence",
        title: "Record convergence at every accepted increment",
        program: "policy = fea.SamplingPolicy.everyAcceptedIncrement();\nconvergence = fea.Observation.diagnostic(\"convergence\", fea.DiagnosticKind.convergence(), policy, fea.RetentionPolicy.latestSamples(200));",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(policy.value, 'every_accepted_increment'));\nassert(strcmp(class(convergence), 'fea.Observation'));" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.SamplingPolicy.everyAcceptedIncrement",
        summary: "Retain a sample at every accepted solver increment.",
        description: "`fea.SamplingPolicy.everyAcceptedIncrement()` builds a `fea.SamplingPolicy` that keeps a sample each time the solver accepts an increment.",
        keywords: &["fea", "observation", "sampling", "increment", "transient", "nonlinear"],
        related: &["fea.SamplingPolicy.adaptive", "fea.RetentionPolicy.latestSamples", "fea.Observation.history", "fea.Observation.field", "fea.Observation.diagnostic", "fea.IndependentDomain.timeSeconds"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "How do I avoid unbounded storage?", answer: "Combine it with `fea.RetentionPolicy.latestSamples(n)` or `fea.RetentionPolicy.explicitSamples(n)`." }],
        links: &[
            BuiltinDocumentationLink { label: "Procedure sequences, observations, and accuracy", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/study-contracts") },
            BuiltinDocumentationLink { label: "Outputs, fields, and observations", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/language/outputs-fields-and-observations") },
        ],
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/observations/sampling.rs") }],
        verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Catalog inference contract", location: "crates/runmat-builtins/src/catalog/entries/fea/study_types/sampling_policy/tests.rs::sampling_policy_constructors_follow_their_positional_contract" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Convergence diagnostics", location: "crates/runmat-analysis/fea/src/tests.rs::convergence_diagnostics_are_emitted" },
        ],
        notes: &[],
    });
