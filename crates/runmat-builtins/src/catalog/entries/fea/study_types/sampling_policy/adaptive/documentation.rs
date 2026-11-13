use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`minimumInterval` and `relativeChange` are finite real scalars; `maximumSamples` is a nonnegative integer (integer classes accepted exactly).",
        "Model validation requires `minimumInterval > 0`, `maximumSamples > 0`, and `relativeChange > 0`, and accepts the policy only for time, frequency, load-factor, and response-spectrum domains.",
        "`minimumInterval` is in the units of the step's independent domain (seconds, hertz, or load factor). The sample count never exceeds `maximumSamples`.",
    ] },
    BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
        "Pass the policy as the `sampling` argument (11th) of `fea.Observation.field`, `history`, `probe`, or `reduction`, or as the third argument of `fea.Observation.diagnostic`. Model validation checks it against the independent domain of the step that owns the observation (`fea.IndependentDomain.*`).",
    ] },
    BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
        "`RunMat:fea:InvalidInput` when an argument is not numeric, is nonfinite, or `maximumSamples` is not a nonnegative integer.",
        "`RunMat:fea:Internal` if the policy cannot be converted into a RunMat object.",
        "`RunMat:fea:TooManyOutputs` when more than one output is requested.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "transient",
        title: "Adaptive sampling of a 0.5 s transient",
        program: "policy = fea.SamplingPolicy.adaptive(0.001, 200, 0.05);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(policy), 'fea.SamplingPolicy'));\nassert(policy.value.adaptive.maximum_samples == 200);\nassert(policy.value.adaptive.relative_change == 0.05);" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.SamplingPolicy.adaptive",
        summary: "Retain samples adaptively when the result changes, within explicit bounds.",
        description: "`fea.SamplingPolicy.adaptive(minimumInterval, maximumSamples, relativeChange)` builds a bounded adaptive `fea.SamplingPolicy` that adds a sample when the observed value changes by more than `relativeChange`.",
        keywords: &["fea", "observation", "sampling", "adaptive", "bounded"],
        related: &["fea.SamplingPolicy.everyAcceptedIncrement", "fea.Observation.history", "fea.Observation.field", "fea.Observation.diagnostic", "fea.IndependentDomain.timeSeconds"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "Is `relativeChange` a percentage?", answer: "No. It is a fraction: 0.05 means a 5 % change." }],
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
