use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "All three arguments are finite real numeric scalars; integer classes are accepted. The constructor stores them as given.",
        "Model validation requires `interval > 0` and `end >= start`, and requires the interval to lie inside the step's independent domain. Frequency intervals must start at or after zero and stay within the first and last solved frequency.",
        "Use it with a step whose independent domain is `fea.IndependentDomain.frequencyHertz`.",
    ] },
    BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
        "Pass the policy as the `sampling` argument (11th) of `fea.Observation.field`, `history`, `probe`, or `reduction`, or as the third argument of `fea.Observation.diagnostic`. Model validation checks it against the independent domain of the step that owns the observation (`fea.IndependentDomain.*`).",
    ] },
    BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
        "`RunMat:fea:InvalidInput` when an argument is not a finite real numeric scalar.",
        "`RunMat:fea:Internal` if the policy cannot be converted into a RunMat object.",
        "`RunMat:fea:TooManyOutputs` when more than one output is requested.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "band",
        title: "Sample 10 Hz to 200 Hz every 5 Hz",
        program: "policy = fea.SamplingPolicy.frequencyIntervalHertz(10, 200, 5);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(policy), 'fea.SamplingPolicy'));\nassert(policy.value.frequency_interval_hertz.interval == 5);" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.SamplingPolicy.frequencyIntervalHertz",
        summary: "Retain samples on a uniform frequency grid in hertz.",
        description: "`fea.SamplingPolicy.frequencyIntervalHertz(start, end, interval)` builds a `fea.SamplingPolicy` that retains results from `start` to `end` every `interval` hertz.",
        keywords: &["fea", "observation", "sampling", "frequency", "interval", "harmonic"],
        related: &["fea.SamplingPolicy.frequencyHertz", "fea.Observation.history", "fea.Observation.field", "fea.Observation.diagnostic", "fea.IndependentDomain.timeSeconds"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "Does the grid have to match solved frequencies?", answer: "The interval must lie within the solved frequency range; it does not have to list solved frequencies exactly." }],
        links: &[
            BuiltinDocumentationLink { label: "Procedure sequences, observations, and accuracy", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/study-contracts") },
            BuiltinDocumentationLink { label: "Outputs, fields, and observations", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/language/outputs-fields-and-observations") },
        ],
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/observations/sampling.rs") }],
        verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Catalog inference contract", location: "crates/runmat-builtins/src/catalog/entries/fea/study_types/sampling_policy/tests.rs::sampling_policy_constructors_follow_their_positional_contract" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Interval admission", location: "crates/runmat-analysis/core/src/validate_outputs/sampling.rs::tests::intervals_require_finite_ordered_positive_bounds" },
        ],
        notes: &[],
    });
