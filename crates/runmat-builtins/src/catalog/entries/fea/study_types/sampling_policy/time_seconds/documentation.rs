use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`values` is a nonempty host numeric scalar or array of finite values; integer classes are accepted and converted once to binary64. The values are stored in the order given.",
        "Model validation requires the times to be nonnegative and strictly increasing, and to lie inside the step's time interval [start, end]. Exact schedules advance by ordinal, so a retry cannot skip or repeat a time.",
        "Use it with a step whose independent domain is `fea.IndependentDomain.timeSeconds`; any other domain fails model validation with `sampling policy is invalid for the step's independent domain`.",
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
        id: "impact",
        title: "Sample an impact response at four times",
        program: "policy = fea.SamplingPolicy.timeSeconds([0 0.010 0.020 0.050]);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(policy), 'fea.SamplingPolicy'));\nassert(isequal(policy.value.time_seconds, [0 0.01 0.02 0.05]));" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.SamplingPolicy.timeSeconds",
        summary: "Retain samples at explicit physical times in seconds.",
        description: "`fea.SamplingPolicy.timeSeconds(values)` builds a `fea.SamplingPolicy` that retains a result at each listed time in seconds.",
        keywords: &["fea", "observation", "sampling", "time", "transient", "seconds"],
        related: &["fea.SamplingPolicy.timeIntervalSeconds", "fea.Observation.history", "fea.Observation.field", "fea.Observation.diagnostic", "fea.IndependentDomain.timeSeconds"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "Do the times have to match solver increments?", answer: "No. They must lie inside the step's time interval; the result is evaluated at the requested times." }],
        links: &[
            BuiltinDocumentationLink { label: "Procedure sequences, observations, and accuracy", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/study-contracts") },
            BuiltinDocumentationLink { label: "Outputs, fields, and observations", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/language/outputs-fields-and-observations") },
        ],
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/observations/sampling.rs") }],
        verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Catalog inference contract", location: "crates/runmat-builtins/src/catalog/entries/fea/study_types/sampling_policy/tests.rs::sampling_policy_constructors_follow_their_positional_contract" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Explicit samples inside the domain", location: "crates/runmat-analysis/core/src/validate_outputs/sampling.rs::tests::explicit_samples_must_remain_inside_the_independent_domain" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Bounded histories end to end", location: "crates/runmat-analysis/language/tests/output_histories.rs::bounded_sampling_interpolation_and_series_reductions_are_typed_end_to_end" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed vectors are rows", location: "crates/runmat-runtime/src/builtins/fea/observations/tests.rs::typed_vectors_are_rows_and_still_decode" },
        ],
        notes: &[],
    });
