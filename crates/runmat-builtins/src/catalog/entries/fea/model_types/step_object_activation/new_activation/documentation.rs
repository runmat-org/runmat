use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "`object` must be a `fea.StepObjectRef` and `lifecycle` a `fea.ActivationLifecycle`; both are decoded and stored as `object` and `lifecycle`. One activation covers one object; a step lists as many as it needs.",
            "All inputs are checked before anything is built: a wrong argument count, a request for more than one output, a value of the wrong class, an empty identifier, or an out-of-range value raises `RunMat:fea:InvalidInput` and no object is returned.",
            "The result is an immutable host value object, not a handle: assigning it to another variable copies the value, and its properties are read-only. It encodes portably, so it can cross an execution boundary unchanged.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Use in a study",
        paragraphs: &[
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
    id: "new",
    title: "Bind an object reference and its lifecycle for one procedure step",
    program: "activation = fea.StepObjectActivation.new(fea.StepObjectRef.excitation('service_load'), fea.ActivationLifecycle.activate());",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(class(activation), 'fea.StepObjectActivation'));\nassert(strcmp(activation.object.excitation, 'service_load'));\nassert(strcmp(activation.lifecycle, 'activate'));",
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
        title: "fea.StepObjectActivation.new",
        summary: "Bind an object reference and its lifecycle for one procedure step.",
        description: "`fea.StepObjectActivation.new(object, lifecycle)` returns a `fea.StepObjectActivation` that says what happens to one model object in the step being configured.",
        keywords: &[
            "fea",
            "step",
            "activation",
        ],
        related: &[
            "fea.StepObjectRef.excitation",
            "fea.ActivationLifecycle.activate",
            "fea.ProcedureStep.configure",
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
