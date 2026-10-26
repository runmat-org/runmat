use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`id` is non-empty text naming a study parameter; `values` is a nonempty array of finite values in the parameter's canonical SI unit. Integer classes are accepted.",
        "Model validation requires the values to be finite and strictly increasing.",
    ] },
    BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
        "Pass the domain as the second argument (`independentDomain`) of `fea.ProcedureStep.configure`. It replaces the step's default domain, and model validation checks it against the step's procedure and against every observation's sampling policy.",
        "`fea.SamplingPolicy.studyParameterValues` selects a subset of exactly these values, and trapezoidal series weighting integrates over them.",
    ] },
    BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
        "`RunMat:fea:InvalidInput` when `id` is empty or `values` is not a nonempty finite numeric array.",
        "`RunMat:fea:Internal` if the value cannot be converted into a RunMat object.",
        "`RunMat:fea:TooManyOutputs` when more than one output is requested.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "thickness",
        title: "Plate thickness from 2 mm to 4 mm",
        program: "domain = fea.IndependentDomain.studyParameter(\"plate_thickness\", [0.002 0.003 0.004]);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(domain), 'fea.IndependentDomain'));\nassert(strcmp(domain.value.study_parameter.parameter_id, 'plate_thickness'));\nassert(isequal(domain.value.study_parameter.canonical_values, [0.002 0.003 0.004]));" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.IndependentDomain.studyParameter",
        summary: "Declare explicit samples of a named study parameter as a step's independent domain.",
        description: "`fea.IndependentDomain.studyParameter(id, values)` builds a parameter `fea.IndependentDomain` that solves the step once per listed value of the parameter `id`.",
        keywords: &["fea", "domain", "parameter", "sweep", "step"],
        related: &["fea.SamplingPolicy.studyParameterValues", "fea.ProcedureStep.configure", "fea.SamplingPolicy.finalSample"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "Which units do the values use?", answer: "The parameter's canonical SI unit; a thickness in meters, for example." }],
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
