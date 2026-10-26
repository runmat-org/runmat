use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "All five arguments are nonnegative integer scalars; integer classes and integer-valued doubles are accepted exactly and stored as unsigned 64-bit counts.",
        "The result is an upper bound (`mode` is `upper_bound`, the counts are under `limit`). Topology-aware planning computes the exact cost before any solve work and fails admission if the plan would exceed this bound. It is not a measured result.",
        "Model validation requires a nonzero `storedBytes` when the observation uses `manifestAndChunks` access or `canonical`/`untilDependentsComplete` retention.",
    ] },
    BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
        "Pass the cost as the 15th argument (`cost`) of `fea.Observation.field`, `history`, `probe`, or `reduction`.",
    ] },
    BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
        "`RunMat:fea:InvalidInput` when any argument is not a nonnegative integer scalar or does not fit in 64 bits.",
        "`RunMat:fea:Internal` if the value cannot be converted into a RunMat object.",
        "`RunMat:fea:TooManyOutputs` when more than one output is requested.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "field-budget",
        title: "Budget a displacement field",
        program: "cost = fea.ObservationCost.new(25000, 400000, 24000, 24000, 0);\n",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(cost), 'fea.ObservationCost'));\nassert(strcmp(cost.mode, 'upper_bound'));\nassert(cost.limit.stored_bytes == 24000);\nassert(cost.limit.collective_count == 0);" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.ObservationCost.new",
        summary: "Declare an upper-bound resource estimate for one observation.",
        description: "`fea.ObservationCost.new(operationCount, peakResidentBytes, storedBytes, transferBytes, collectiveCount)` builds a `fea.ObservationCost` that caps the resources one observation may use.",
        keywords: &["fea", "observation", "cost", "resources", "memory", "budget"],
        related: &["fea.ObservationStorage.new", "fea.Observation.field"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "What happens if my estimate is too low?", answer: "Planning rejects the study before the solve starts instead of overrunning the declared budget." }, BuiltinDocumentationFaq { question: "Can I pass byte counts as `int64`?", answer: "Yes. All eight integer classes are accepted exactly." }],
        links: &[
            BuiltinDocumentationLink { label: "Procedure sequences, observations, and accuracy", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/study-contracts") },
            BuiltinDocumentationLink { label: "Outputs, fields, and observations", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/language/outputs-fields-and-observations") },
        ],
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/observations/storage.rs") }],
        verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Catalog inference contract", location: "crates/runmat-builtins/src/catalog/entries/fea/study_types/observation_cost/tests.rs::observation_cost_constructors_follow_their_positional_contract" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed sequence round trip", location: "crates/runmat-runtime/src/builtins/fea/sequences.rs::tests::configured_sequence_round_trips_with_typed_study_contracts" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Cost codec round trip", location: "crates/runmat-runtime/src/builtins/fea/observations/tests.rs::observation_cost_encodes_through_its_registered_schema" },
        ],
        notes: &[],
    });
