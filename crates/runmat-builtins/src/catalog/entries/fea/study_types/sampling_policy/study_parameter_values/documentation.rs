use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`values` is a nonempty host numeric scalar or array of finite values; integer classes are accepted and converted once to binary64. The values are stored in the order given.",
        "Model validation requires the values to be strictly increasing and every value to be one of the step's canonical parameter values exactly.",
        "Use it with a step whose independent domain is `fea.IndependentDomain.studyParameter`; any other domain fails model validation with `sampling policy is invalid for the step's independent domain`.",
    ] },
    BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
        "Pass the policy as the `sampling` argument (11th) of `fea.Observation.field`, `history`, `probe`, or `reduction`, or as the third argument of `fea.Observation.diagnostic`. Model validation checks it against the independent domain of the step that owns the observation (`fea.IndependentDomain.*`).",
    ] },
    BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
        "`RunMat:fea:InvalidInput` when `values` is not numeric, is empty, or contains NaN or Inf.",
        "`RunMat:fea:Internal` if the policy cannot be converted into a RunMat object.",
        "`RunMat:fea:TooManyOutputs` when more than one output is requested.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "thickness",
        title: "Keep two plate thicknesses from a parameter study",
        program: "domain = fea.IndependentDomain.studyParameter(\"plate_thickness\", [0.002 0.003 0.004]);\npolicy = fea.SamplingPolicy.studyParameterValues([0.002 0.004]);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(policy), 'fea.SamplingPolicy'));\nassert(isequal(policy.value.study_parameter_values, [0.002 0.004]));" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.SamplingPolicy.studyParameterValues",
        summary: "Retain samples at selected values of a study parameter.",
        description: "`fea.SamplingPolicy.studyParameterValues(values)` builds a `fea.SamplingPolicy` that retains a result at each listed value of the step's study parameter.",
        keywords: &["fea", "observation", "sampling", "parameter", "sweep"],
        related: &["fea.IndependentDomain.studyParameter", "fea.Observation.history", "fea.Observation.field", "fea.Observation.diagnostic", "fea.IndependentDomain.timeSeconds"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "Which units do the values use?", answer: "The canonical (SI) units of the parameter, matching the values given to `fea.IndependentDomain.studyParameter`." }],
        links: &[
            BuiltinDocumentationLink { label: "Procedure sequences, observations, and accuracy", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/study-contracts") },
            BuiltinDocumentationLink { label: "Outputs, fields, and observations", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/language/outputs-fields-and-observations") },
        ],
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/observations/sampling.rs") }],
        verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Catalog inference contract", location: "crates/runmat-builtins/src/catalog/entries/fea/study_types/sampling_policy/tests.rs::sampling_policy_constructors_follow_their_positional_contract" },
        ],
        notes: &[],
    });
