use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "The reference stores only the kind and the id (`value.initial_condition`); it does not hold the object. The id must match the id given to the fea.*InitialCondition object in the model, which is checked when the configured step is validated against the model. `id` must be nonempty text.",
            "All inputs are checked before anything is built: a wrong argument count, a request for more than one output, a value of the wrong class, an empty identifier, or an out-of-range value raises `RunMat:fea:InvalidInput` and no object is returned.",
            "The result is an immutable host value object, not a handle: assigning it to another variable copies the value, and its properties are read-only. It encodes portably, so it can cross an execution boundary unchanged.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Use in a study",
        paragraphs: &[
            "Pass the reference to `fea.StepObjectActivation.new(ref, lifecycle)`, or as the prior object of `fea.ActivationLifecycle.modify`.",
            "Activations are passed as the third argument of `fea.ProcedureStep.configure(step, independentDomain, activations, stateInputs, stateOutputs, observations, accuracy, restart)`, as one `fea.StepObjectActivation` or a cell array of them.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Errors",
        paragraphs: &[
            "`RunMat:fea:InvalidInput`: an argument has the wrong class or shape, an identifier is empty, or a value violates the constructor's range rules.",
            "`RunMat:fea:Internal`: the validated object could not be converted to a RunMat value (not expected in normal use).",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "initialCondition",
    title: "Refer to an initial condition by id for step activation",
    program: "ref = fea.StepObjectRef.initialCondition('item_1');",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(class(ref), 'fea.StepObjectRef'));\nassert(strcmp(ref.value.initial_condition, 'item_1'));",
    },
}];

const FAQS: &[BuiltinDocumentationFaq] = &[];

const LINKS: &[BuiltinDocumentationLink] = &[BuiltinDocumentationLink {
    label: "Studies, analyses, and state",
    target: BuiltinDocumentationLinkTarget::Documentation(
        "/docs/simulation/language/studies-analyses-and-state",
    ),
}];

const IMPLEMENTATION: &[BuiltinDocumentationLink] = &[BuiltinDocumentationLink {
    label: "Implementation",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/activations.rs"),
}];

const VERIFICATION: &[BuiltinEvidenceReference] = &[
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Configured sequence round-trips with typed activations",
        location: "crates/runmat-runtime/src/builtins/fea/sequences.rs::tests::configured_sequence_round_trips_with_typed_study_contracts",
    },
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Every step reference and lifecycle round-trips through an activation",
        location: "crates/runmat-runtime/src/builtins/fea/constraints/coverage_tests.rs::every_step_reference_and_lifecycle_round_trips_through_an_activation",
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    documentation(ConstructorDocumentationDefinition {
        title: "fea.StepObjectRef.initialCondition",
        summary: "Refer to an initial condition by id for step activation.",
        description: "`fea.StepObjectRef.initialCondition(id)` returns a `fea.StepObjectRef` that names an initial condition of the model, so a procedure step can activate, continue, modify, or deactivate it.",
        keywords: &[
            "fea",
            "step",
            "activation",
            "reference",
            "initialCondition",
        ],
        related: &[
            "fea.StepObjectActivation.new",
            "fea.ActivationLifecycle.activate",
        ],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: FAQS,
        links: LINKS,
        implementation: IMPLEMENTATION,
        verification: VERIFICATION,
        notes: &[
        ],
    });
