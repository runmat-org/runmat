use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "All three arguments are finite real numeric scalars; integer classes are accepted. The constructor stores them as given.",
        "Model validation requires `interval > 0` and `end >= start`, and requires the interval to lie inside the step's independent domain. Unlike time and frequency intervals, a load-factor interval may start below zero.",
        "Use it with a step whose independent domain is `fea.IndependentDomain.loadFactor`.",
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
        id: "tenths",
        title: "Sample every tenth of the load",
        program: "policy = fea.SamplingPolicy.loadFactorInterval(0, 1, 0.1);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(policy), 'fea.SamplingPolicy'));\nassert(policy.value.load_factor_interval.interval == 0.1);" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.SamplingPolicy.loadFactorInterval",
        summary: "Retain samples on a uniform load-factor grid.",
        description: "`fea.SamplingPolicy.loadFactorInterval(start, end, interval)` builds a `fea.SamplingPolicy` that retains results from `start` to `end` every `interval` load-factor units.",
        keywords: &["fea", "observation", "sampling", "load", "factor", "interval"],
        related: &["fea.SamplingPolicy.loadFactors", "fea.Observation.history", "fea.Observation.field", "fea.Observation.diagnostic", "fea.IndependentDomain.timeSeconds"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "Can the interval extend past the domain?", answer: "No. `start` and `end` must lie inside the step's load-factor interval." }],
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
