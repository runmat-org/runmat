use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`values` is a nonempty host numeric scalar or array of finite values; integer classes are accepted and converted once to binary64. The values are stored in the order given.",
        "Model validation requires the load factors to be strictly increasing and to lie inside the step's [start, end] load-factor interval. Negative load factors are allowed when the domain includes them.",
        "Use it with a step whose independent domain is `fea.IndependentDomain.loadFactor`; any other domain fails model validation with `sampling policy is invalid for the step's independent domain`.",
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
        id: "ramp",
        title: "Keep quarter, half, and full load",
        program: "policy = fea.SamplingPolicy.loadFactors([0.25 0.5 1.0]);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(policy), 'fea.SamplingPolicy'));\nassert(isequal(policy.value.load_factors, [0.25 0.5 1]));" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.SamplingPolicy.loadFactors",
        summary: "Retain samples at explicit load factors.",
        description: "`fea.SamplingPolicy.loadFactors(values)` builds a `fea.SamplingPolicy` that retains a result at each listed load factor.",
        keywords: &["fea", "observation", "sampling", "load", "factor", "nonlinear"],
        related: &["fea.SamplingPolicy.loadFactorInterval", "fea.IndependentDomain.loadFactor", "fea.Observation.history", "fea.Observation.field", "fea.Observation.diagnostic", "fea.IndependentDomain.timeSeconds"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "What is a load factor?", answer: "A dimensionless multiplier on the step's applied loads, used by nonlinear static procedures to ramp loading." }],
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
