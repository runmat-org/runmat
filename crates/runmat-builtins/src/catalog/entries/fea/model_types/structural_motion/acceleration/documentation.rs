use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "The displacement history is reconstructed exactly for a piecewise-linear acceleration: velocity is the integral of acceleration plus the initial velocity, and displacement the integral of velocity plus the initial displacement. Because the reconstruction is exact rather than numerically re-integrated, displacement, velocity, and acceleration stay consistent.",
            "`acceleration` must be a time-domain, real64, three-component acceleration signal in the global frame, for example from `fea.AccelerationSignal.fromTimeSamples`. A constant signal has no time domain and is rejected. Only the component selected by `dof` drives the support; the motion stores the signal id, not a copy of the samples.",
            "`initialDisplacement` is a `fea.Displacement` and `initialVelocity` a `fea.Velocity`; both are scalars for the selected component.",
            "All inputs are checked before anything is built: a wrong argument count, a request for more than one output, a value of the wrong class, an empty identifier, or an out-of-range value raises `RunMat:fea:InvalidInput` and no object is returned.",
            "The result is an immutable host value object, not a handle: assigning it to another variable copies the value, and its properties are read-only. It encodes portably, so it can cross an execution boundary unchanged.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Use in a study",
        paragraphs: &[
            "Pass one motion, or a cell array of motions for different DOFs, to `fea.StructuralConstraint.prescribedMotion`, and add the acceleration signal to the model's `Signals` option. Use it in a structural transient step.",
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
    id: "acceleration",
    title: "Define a support motion from an acceleration history",
    program: "shake = fea.AccelerationSignal.fromTimeSamples('base_shake', [0; 0.01; 0.02], [0 0 0; 0 0 2.5; 0 0 0], fea.SignalInterpolation.linear(), fea.SignalExtrapolation.reject());\nmotion = fea.StructuralMotion.acceleration(fea.StructuralDof.uz(), shake, fea.Displacement.meters(0), fea.Velocity.metersPerSecond(0));",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(class(motion), 'fea.StructuralTranslationMotion'));\nassert(strcmp(motion.dof, 'uz'));\nh = motion.history.acceleration_driven;\nassert(strcmp(h.acceleration, 'base_shake'));\nassert(strcmp(h.reconstruction, 'piecewise_linear_acceleration_exact'));",
    },
}];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Why is my constant acceleration signal rejected?",
        answer: "Support motion needs a time-domain history. Build the signal with `fea.AccelerationSignal.fromTimeSamples` or `fea.AccelerationSignal.fromTimeKnots`; a constant signal has no time axis.",
    },
];

const LINKS: &[BuiltinDocumentationLink] = &[BuiltinDocumentationLink {
    label: "Structural constraints",
    target: BuiltinDocumentationLinkTarget::Documentation(
        "/docs/simulation/constraints/structural",
    ),
}];

const IMPLEMENTATION: &[BuiltinDocumentationLink] = &[BuiltinDocumentationLink {
    label: "Implementation",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/constraints/support/motion.rs"),
}];

const VERIFICATION: &[BuiltinEvidenceReference] = &[
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Prescribed support motion uses a typed vector acceleration",
        location: "crates/runmat-runtime/src/builtins/fea/constraints.rs::tests::prescribed_support_motion_uses_typed_vector_acceleration_and_has_no_legacy_actions",
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    documentation(ConstructorDocumentationDefinition {
        title: "fea.StructuralMotion.acceleration",
        summary: "Define a support motion from an acceleration history.",
        description: "`fea.StructuralMotion.acceleration(dof, acceleration, initialDisplacement, initialVelocity)` returns a `fea.StructuralTranslationMotion` that drives one translational support DOF from a measured or designed acceleration record.",
        keywords: &[
            "fea",
            "structural",
            "support motion",
            "acceleration",
            "base excitation",
            "transient",
        ],
        related: &[
            "fea.StructuralConstraint.prescribedMotion",
            "fea.AccelerationSignal.fromTimeSamples",
            "fea.StructuralDof.uz",
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
