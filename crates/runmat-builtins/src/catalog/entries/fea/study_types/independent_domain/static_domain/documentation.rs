use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "Takes no arguments. A static domain has exactly one solution sample, so only `fea.SamplingPolicy.finalSample()` and static-compatible observations apply.",
    ] },
    BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
        "Pass the domain as the second argument (`independentDomain`) of `fea.ProcedureStep.configure`. It replaces the step's default domain, and model validation checks it against the step's procedure and against every observation's sampling policy.",
        "Static domains suit linear static and preload steps.",
    ] },
    BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
        "`RunMat:fea:Internal` if the value cannot be converted into a RunMat object.",
        "`RunMat:fea:TooManyOutputs` when more than one output is requested.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "static",
        title: "Static step domain",
        program: "domain = fea.IndependentDomain.staticDomain();",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(domain), 'fea.IndependentDomain'));\nassert(strcmp(domain.value, 'static'));" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.IndependentDomain.staticDomain",
        summary: "Declare a single static equilibrium state as a step's independent domain.",
        description: "`fea.IndependentDomain.staticDomain()` builds the `fea.IndependentDomain` for a step that solves one static equilibrium.",
        keywords: &["fea", "domain", "static", "equilibrium", "step"],
        related: &["fea.IndependentDomain.loadFactor", "fea.ProcedureStep.configure", "fea.SamplingPolicy.finalSample"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "Can a static step record a history?", answer: "No. It has one sample; use a load-factor domain to follow a load ramp." }],
        links: &[
            BuiltinDocumentationLink { label: "Procedure sequences, observations, and accuracy", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/study-contracts") },
            BuiltinDocumentationLink { label: "Studies, analyses, and state", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/language/studies-analyses-and-state") },
        ],
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/sequences.rs") }],
        verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Catalog inference contract", location: "crates/runmat-builtins/src/catalog/entries/fea/study_types/independent_domain/tests.rs::independent_domain_constructors_follow_their_positional_contract" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed sequence round trip", location: "crates/runmat-runtime/src/builtins/fea/sequences.rs::tests::configured_sequence_round_trips_with_typed_study_contracts" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Compiled sequence execution", location: "crates/runmat-vm/tests/fea_study_contracts.rs::typed_study_sequence_compiles_and_executes_through_the_vm" },
        ],
        notes: &[],
    });
