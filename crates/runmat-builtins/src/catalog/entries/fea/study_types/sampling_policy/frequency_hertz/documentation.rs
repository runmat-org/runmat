use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`values` is a nonempty host numeric scalar or array of finite values; integer classes are accepted and converted once to binary64. The values are stored in the order given.",
        "Model validation requires the frequencies to be nonnegative and strictly increasing, and every value must be one of the step's solved frequencies exactly.",
        "Use it with a step whose independent domain is `fea.IndependentDomain.frequencyHertz`; any other domain fails model validation with `sampling policy is invalid for the step's independent domain`.",
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
        id: "resonance",
        title: "Keep three frequencies of a harmonic sweep",
        program: "domain = fea.IndependentDomain.frequencyHertz([20 50 100 200]);\npolicy = fea.SamplingPolicy.frequencyHertz([20 50 100]);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(policy), 'fea.SamplingPolicy'));\nassert(isequal(policy.value.frequency_hertz, [20 50 100]));" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.SamplingPolicy.frequencyHertz",
        summary: "Retain samples at explicit frequencies in hertz.",
        description: "`fea.SamplingPolicy.frequencyHertz(values)` builds a `fea.SamplingPolicy` that retains a result at each listed frequency in hertz.",
        keywords: &["fea", "observation", "sampling", "frequency", "harmonic", "hertz"],
        related: &["fea.SamplingPolicy.frequencyIntervalHertz", "fea.IndependentDomain.frequencyHertz", "fea.Observation.history", "fea.Observation.field", "fea.Observation.diagnostic", "fea.IndependentDomain.timeSeconds"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "Can I request a frequency the step does not solve?", answer: "No. Each requested frequency must be one of the domain's frequencies; add it to `fea.IndependentDomain.frequencyHertz` first." }],
        links: &[
            BuiltinDocumentationLink { label: "Procedure sequences, observations, and accuracy", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/study-contracts") },
            BuiltinDocumentationLink { label: "Outputs, fields, and observations", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/language/outputs-fields-and-observations") },
        ],
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/observations/sampling.rs") }],
        verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Catalog inference contract", location: "crates/runmat-builtins/src/catalog/entries/fea/study_types/sampling_policy/tests.rs::sampling_policy_constructors_follow_their_positional_contract" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Explicit samples inside the domain", location: "crates/runmat-analysis/core/src/validate_outputs/sampling.rs::tests::explicit_samples_must_remain_inside_the_independent_domain" },
        ],
        notes: &[],
    });
