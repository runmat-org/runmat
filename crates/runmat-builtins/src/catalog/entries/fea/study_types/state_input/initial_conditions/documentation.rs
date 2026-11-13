use crate::fea_contract::catalog_entry::{self, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
        "`id` is non-empty text, unique among the step's inputs. `kind` is the required `fea.StateKind`, `useAs` a `fea.StateUse` (physical initial state, linearization point, prestress, initial guess, modal basis, coupled interface, or restart), and `conflict` a `fea.InitialStateConflictPolicy`.",
        "Model validation requires `conflict` to be `fea.InitialStateConflictPolicy.reject()` for initial-condition inputs, requires `useAs` to accept the kind and the step's physics family to consume it, and requires an authored initial condition that applies to the step and supplies that kind.",
    ] },
    BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
        "Collect state inputs in a cell array and pass them as the `stateInputs` argument (4th) of `fea.ProcedureStep.configure`.",
        "The initial conditions themselves are declared on the model, for example with `fea.StructuralInitialCondition.displacement` and the model's `InitialConditions` option.",
    ] },
    BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
        "`RunMat:fea:InvalidInput` when `id` is empty or an argument is not the expected typed value.",
        "`RunMat:fea:Internal` if the value cannot be converted into a RunMat object.",
        "`RunMat:fea:TooManyOutputs` when more than one output is requested.",
    ] },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "temperature",
        title: "Start a thermal step from authored temperatures",
        program: "input = fea.StateInput.initialConditions(\"initial_temperature\", fea.StateKind.thermal(), ...\n    fea.StateUse.physicalInitialState(), fea.InitialStateConflictPolicy.reject());",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(strcmp(class(input), 'fea.StateInput'));\nassert(strcmp(input.input_id, 'initial_temperature'));\nassert(strcmp(input.source, 'initial_conditions'));\nassert(strcmp(input.conflict, 'reject'));" },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.StateInput.initialConditions",
        summary: "Consume authored initial conditions as a step's typed starting state.",
        description: "`fea.StateInput.initialConditions(id, kind, useAs, conflict)` builds a `fea.StateInput` that takes a step's starting state from the model's authored initial conditions.",
        keywords: &["fea", "sequence", "state", "input", "initial", "conditions"],
        related: &["fea.StateInput.upstream", "fea.ProcedureStep.configure"],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: &[BuiltinDocumentationFaq { question: "Why must the conflict policy be `reject`?", answer: "Initial conditions are the only source here; the replace and augment policies are for reconciling an upstream state with authored conditions." }],
        links: &[
            BuiltinDocumentationLink { label: "Procedure sequences, observations, and accuracy", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/study-contracts") },
            BuiltinDocumentationLink { label: "Studies, analyses, and state", target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/language/studies-analyses-and-state") },
        ],
        implementation: &[BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/sequences.rs") }],
        verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Catalog inference contract", location: "crates/runmat-builtins/src/catalog/entries/fea/study_types/state_input/tests.rs::state_input_constructors_follow_their_positional_contract" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Initial-condition inputs", location: "crates/runmat-analysis/core/src/validate_sequence/tests.rs::initial_state_input_requires_a_condition_for_the_consuming_step" },
        ],
        notes: &[],
    });
