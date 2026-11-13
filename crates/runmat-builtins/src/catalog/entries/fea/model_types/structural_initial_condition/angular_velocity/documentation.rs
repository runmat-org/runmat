use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "The angular velocity vector in rad/s is sampled from a three-component angular-velocity signal; 3000 rpm is 314.16 rad/s. The signal must be a `fea.AngularVelocitySignal` with the shape this condition needs; another quantity or shape is rejected.",
            "`target` is either `fea.InitialConditionTarget.wholeModel()`, which applies the state everywhere and carries no geometry binding, or a `geometry.Region`, which is recorded as a volume target bound to the region's geometry id and revision.",
            "The initial value is a sample of `signal` at `abscissa` (a finite real scalar; integer classes are accepted and converted once to binary64), taken with `sampleApproach`: `fea.SignalSampleApproach.at()`, `.leftLimit()`, or `.rightLimit()`, which matters only where the signal jumps. The condition stores the signal id, the abscissa, and the approach, not the sampled value; the frame is taken from the signal (global for a constant signal).",
            "`beforeStepId` names the step that consumes this state; it must be nonempty. `id` must be nonempty text.",
            "All inputs are checked before anything is built: a wrong argument count, a request for more than one output, a value of the wrong class, an empty identifier, or an out-of-range value raises `RunMat:fea:InvalidInput` and no object is returned.",
            "The result is an immutable host value object, not a handle: assigning it to another variable copies the value, and its properties are read-only. It encodes portably, so it can cross an execution boundary unchanged.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Use in a study",
        paragraphs: &[
            "Pass the condition to `fea.model` in the `InitialConditions` option and its signal in the `Signals` option. `beforeStepId` must name one of the model's steps, normally a transient step. Step activations refer to the condition with `fea.StepObjectRef.initialCondition(id)`.",
            "Use it before a structural transient step.",
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
    id: "angularVelocity",
    title: "Set the initial angular velocity of a structural transient analysis",
    program: "signal = fea.AngularVelocitySignal.constantRadiansPerSecond('w0', [0 0 314.16]);\ncondition = fea.StructuralInitialCondition.angularVelocity('angular_velocity0', fea.InitialConditionTarget.wholeModel(), signal, 0, fea.SignalSampleApproach.at(), 'start');",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(class(condition), 'fea.StructuralInitialCondition'));\nc = condition.condition.angular_velocity;\nassert(strcmp(c.target, 'whole_model'));\nassert(strcmp(c.applies_before_step_id, 'start'));\nassert(c.value.signal_sample.abscissa == 0);",
    },
}];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Why sample a signal instead of passing a value?",
        answer: "Sampling a typed signal keeps units, frame, and provenance on the value and lets the same history provide both the initial state and later loading. For a constant initial value, use a constant signal and abscissa 0.",
    },
];

const LINKS: &[BuiltinDocumentationLink] = &[BuiltinDocumentationLink {
    label: "Initial conditions",
    target: BuiltinDocumentationLinkTarget::Documentation(
        "/docs/simulation/constraints/initial-conditions",
    ),
}];

const IMPLEMENTATION: &[BuiltinDocumentationLink] = &[BuiltinDocumentationLink {
    label: "Implementation",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/initial_conditions.rs"),
}];

const VERIFICATION: &[BuiltinEvidenceReference] = &[
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Whole-model initial state is typed and portable",
        location: "crates/runmat-runtime/src/builtins/fea/initial_conditions.rs::tests::whole_model_initial_state_is_typed_and_portable",
    },
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Region initial state keeps its geometry binding and rejects the wrong quantity",
        location: "crates/runmat-runtime/src/builtins/fea/initial_conditions.rs::tests::region_initial_state_retains_geometry_and_rejects_wrong_quantity",
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    documentation(ConstructorDocumentationDefinition {
        title: "fea.StructuralInitialCondition.angularVelocity",
        summary: "Set the initial angular velocity of a structural transient analysis.",
        description: "`fea.StructuralInitialCondition.angularVelocity(id, target, signal, abscissa, sampleApproach, beforeStepId)` builds a `fea.StructuralInitialCondition` for a spinning part at the start of a step.",
        keywords: &[
            "fea",
            "initial condition",
            "initial state",
            "transient",
            "angular velocity",
            "rotation",
            "spin",
        ],
        related: &[
            "fea.StructuralInitialCondition.velocity",
            "fea.InitialConditionTarget.wholeModel",
            "fea.SignalSampleApproach.at",
            "fea.model",
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
