use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`values` is a vector of positive integer mode indices; mode 1 is the lowest. Integer classes and integer-valued doubles are accepted exactly. Zero, negative, fractional, nonfinite, and matrix-shaped inputs are rejected.",
        "Model validation requires the indices to be strictly increasing and to lie within the count requested by `fea.IndependentDomain.eigenvalue(count)`.",
    ] },
    BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
        "Pass the policy as the `sampling` argument (11th) of `fea.Observation.field`, `history`, `probe`, or `reduction`, or as the third argument of `fea.Observation.diagnostic`. Model validation checks it against the independent domain of the step that owns the observation (`fea.IndependentDomain.*`).",
        "Use it to keep mode shapes or modal participation for selected modes only.",
    ] },
    BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
        "`RunMat:fea:InvalidInput` when an index is zero, negative, fractional, nonfinite, or the input is not a vector.",
        "`RunMat:fea:Internal` if the policy cannot be converted into a RunMat object.",
        "`RunMat:fea:TooManyOutputs` when more than one output is requested.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "first-three",
        title: "Keep the first three modes",
        program: "policy = fea.SamplingPolicy.modeIndices([1 2 3]);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(policy), 'fea.SamplingPolicy'));\nassert(isequal(double(policy.value.mode_indices), [1 2 3]));" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.SamplingPolicy.modeIndices",
        summary: "Retain selected modes of an eigenvalue step by one-based index.",
        description: "`fea.SamplingPolicy.modeIndices(values)` builds a `fea.SamplingPolicy` that keeps the listed modes of a modal or buckling step.",
        keywords: &["fea", "observation", "sampling", "modes", "modal", "eigenvalue"],
        related: &["fea.IndependentDomain.eigenvalue", "fea.Observation.field"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "Are indices zero-based?", answer: "No. Pass one-based indices as in RunMat array indexing; 0 is rejected." }],
        links: &[
            BuiltinDocumentationLink { label: "Procedure sequences, observations, and accuracy", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/study-contracts") },
            BuiltinDocumentationLink { label: "Outputs, fields, and observations", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/language/outputs-fields-and-observations") },
        ],
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/observations/sampling.rs") }],
        verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Catalog inference contract", location: "crates/runmat-builtins/src/catalog/entries/fea/study_types/sampling_policy/tests.rs::sampling_policy_constructors_follow_their_positional_contract" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "One-based mode selection", location: "crates/runmat-runtime/src/builtins/fea/observations/tests.rs::mode_indices_keep_the_public_one_based_selection" },
        ],
        notes: &[],
    });
