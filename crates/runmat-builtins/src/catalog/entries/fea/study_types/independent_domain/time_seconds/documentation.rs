use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`origin`, `start`, and `end` are finite real scalars in seconds; integer classes are accepted. `intervalCount` must be a positive integer (exact for integer classes) and is rejected at construction when zero.",
        "The explicit `origin` keeps total time unambiguous when steps are chained: a continuation step can start where its producer ended without re-basing time.",
        "Model validation requires the domain to be finite and ordered and compatible with the step's procedure (for example a transient procedure).",
    ] },
    BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
        "Pass the domain as the second argument (`independentDomain`) of `fea.ProcedureStep.configure`. It replaces the step's default domain, and model validation checks it against the step's procedure and against every observation's sampling policy.",
        "Time-domain sampling policies (`fea.SamplingPolicy.timeSeconds`, `timeIntervalSeconds`, `everyAcceptedIncrement`, `adaptive`) must fall inside [start, end].",
    ] },
    BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
        "`RunMat:fea:InvalidInput` when a time is not a finite real scalar or `intervalCount` is zero or not a nonnegative integer.",
        "`RunMat:fea:Internal` if the value cannot be converted into a RunMat object.",
        "`RunMat:fea:TooManyOutputs` when more than one output is requested.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "transient",
        title: "0.5 s transient in 500 intervals",
        program: "domain = fea.IndependentDomain.timeSeconds(0, 0, 0.5, 500);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(domain), 'fea.IndependentDomain'));\nassert(domain.value.time_seconds.interval_count == 500);\nassert(domain.value.time_seconds.origin == 0);" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.IndependentDomain.timeSeconds",
        summary: "Declare a physical time interval in seconds as a step's independent domain.",
        description: "`fea.IndependentDomain.timeSeconds(origin, start, end, intervalCount)` builds a time `fea.IndependentDomain` running from `start` to `end` seconds in `intervalCount` equal intervals, measured from a total-time `origin`.",
        keywords: &["fea", "domain", "time", "transient", "seconds", "step"],
        related: &["fea.SamplingPolicy.timeSeconds", "fea.ProcedureStep.configure", "fea.SamplingPolicy.finalSample"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "What is `origin` for?", answer: "It fixes total time across chained steps; `start` and `end` are positions on that total-time axis." }, BuiltinDocumentationFaq { question: "Is the interval count the number of samples?", answer: "It is the number of equal time intervals of the discretization; observation sampling is declared separately." }],
        links: &[
            BuiltinDocumentationLink { label: "Procedure sequences, observations, and accuracy", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/study-contracts") },
            BuiltinDocumentationLink { label: "Studies, analyses, and state", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/language/studies-analyses-and-state") },
        ],
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/sequences.rs") }],
        verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Catalog inference contract", location: "crates/runmat-builtins/src/catalog/entries/fea/study_types/independent_domain/tests.rs::independent_domain_constructors_follow_their_positional_contract" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Time-domain construction", location: "crates/runmat-runtime/src/builtins/fea/sequences.rs::tests::time_domain_requires_and_retains_a_positive_interval_count" },
        ],
        notes: &[],
    });
