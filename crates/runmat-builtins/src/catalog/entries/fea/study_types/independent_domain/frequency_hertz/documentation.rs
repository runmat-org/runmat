use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`values` is a nonempty host numeric scalar or array of finite values in hertz; integer classes are accepted.",
        "Model validation requires the frequencies to be finite and strictly increasing and the domain to suit the step's procedure (for example a harmonic response).",
    ] },
    BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
        "Pass the domain as the second argument (`independentDomain`) of `fea.ProcedureStep.configure`. It replaces the step's default domain, and model validation checks it against the step's procedure and against every observation's sampling policy.",
        "`fea.SamplingPolicy.frequencyHertz` selects a subset of exactly these frequencies; frequency intervals must stay within the first and last value.",
    ] },
    BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
        "`RunMat:fea:InvalidInput` when `values` is not numeric, is empty, or contains NaN or Inf.",
        "`RunMat:fea:Internal` if the value cannot be converted into a RunMat object.",
        "`RunMat:fea:TooManyOutputs` when more than one output is requested.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "harmonic",
        title: "Harmonic sweep at four frequencies",
        program: "domain = fea.IndependentDomain.frequencyHertz([20 50 100 200]);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(domain), 'fea.IndependentDomain'));\nassert(isequal(domain.value.frequency_hertz, [20 50 100 200]));" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.IndependentDomain.frequencyHertz",
        summary: "Declare explicit frequencies in hertz as a step's independent domain.",
        description: "`fea.IndependentDomain.frequencyHertz(values)` builds a frequency `fea.IndependentDomain` that solves at each listed frequency.",
        keywords: &["fea", "domain", "frequency", "harmonic", "hertz", "step"],
        related: &["fea.SamplingPolicy.frequencyHertz", "fea.ProcedureStep.configure", "fea.SamplingPolicy.finalSample"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "Are the frequencies solved independently?", answer: "Yes. Independent frequency points may be distributed when the formulation supports it; results keep their canonical order." }],
        links: &[
            BuiltinDocumentationLink { label: "Procedure sequences, observations, and accuracy", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/study-contracts") },
            BuiltinDocumentationLink { label: "Studies, analyses, and state", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/language/studies-analyses-and-state") },
        ],
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/sequences.rs") }],
        verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Catalog inference contract", location: "crates/runmat-builtins/src/catalog/entries/fea/study_types/independent_domain/tests.rs::independent_domain_constructors_follow_their_positional_contract" },
        ],
        notes: &[],
    });
