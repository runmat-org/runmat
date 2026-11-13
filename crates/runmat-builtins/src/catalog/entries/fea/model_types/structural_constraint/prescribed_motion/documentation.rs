use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::model_types::fixtures::BLOCK_GEOMETRY;
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "Each motion prescribes one global translational DOF of the support. The displacement history is reconstructed exactly from the piecewise-linear acceleration signal and the initial displacement and velocity, so displacement, velocity, and acceleration stay consistent with one another (no drift from separate integration). Components not listed stay free.",
            "`id` names the constraint. It must be nonempty text (a character vector or string scalar) and should be unique among objects of the same kind in one model; validation messages and run evidence report it. Step activations refer to it with `fea.StepObjectRef.constraint(id)`.",
            "`region` must be a `geometry.Region` returned by `geometry.region(asset, selector)`. The object stores the region id together with the region's `geometry_id` and `geometry_revision` (exposed as read-only properties, the revision as `uint32`), so `fea.model` rejects it when the model is built on a different asset or a different revision of the same asset.",
            "The region is recorded as a boundary-face target (`entity_dimension` `face`, cardinality one or more faces). Whether the selected region really is a set of faces is checked when the model is meshed and validated, not by the constructor.",
            "`motions` is one `fea.StructuralMotion` or a cell array, each built with `fea.StructuralMotion.acceleration(dof, accelerationSignal, initialDisplacement, initialVelocity)`. At least one motion is required, and each DOF may appear only once. The constraint is expressed in the global frame.",
            "All inputs are checked before anything is built: a wrong argument count, a request for more than one output, a value of the wrong class, an empty identifier, or an out-of-range value raises `RunMat:fea:InvalidInput` and no object is returned.",
            "The result is an immutable host value object, not a handle: assigning it to another variable copies the value, and its properties are read-only. It encodes portably, so it can cross an execution boundary unchanged.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Use in a study",
        paragraphs: &[
            "Pass the object to `fea.model` in the `Constraints` option, either alone or in a cell array with others of the same family, for example `model = fea.model('part', block, 'Steps', {step}, 'Constraints', {constraint})`.",
            "Use it with a structural transient step (`fea.StructuralStep.transient`). Add each acceleration signal to the model's `Signals` option; the motion refers to it by signal id.",
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
    id: "prescribedMotion",
    title: "Drive a support with a translational acceleration history",
    program: "block = geometry.load('block.obj');\nbase = geometry.region(block, 'z_min');\nshake = fea.AccelerationSignal.fromTimeSamples('base_shake', [0; 0.01; 0.02], [0 0 0; 0 0 2.5; 0 0 0], fea.SignalInterpolation.linear(), fea.SignalExtrapolation.reject());\nmotion = fea.StructuralMotion.acceleration(fea.StructuralDof.uz(), shake, fea.Displacement.meters(0), fea.Velocity.metersPerSecond(0));\nconstraint = fea.StructuralConstraint.prescribedMotion('shaker', base, {motion});",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeFilesystem,
    fixture: BLOCK_GEOMETRY,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(class(constraint), 'fea.StructuralConstraint'));\nm = constraint.constraint.kind.prescribed_motion.translations;\nassert(strcmp(m.dof, 'uz'));\nassert(strcmp(m.history.acceleration_driven.acceleration, 'base_shake'));",
    },
}];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Why acceleration instead of displacement?",
        answer: "Shaker tables and ground motion records are measured as acceleration. Reconstructing displacement from the signal and explicit initial conditions keeps the three kinematic quantities consistent.",
    },
    BuiltinDocumentationFaq {
        question: "Can I reuse the object with another geometry?",
        answer: "No. The object is bound to the `geometry_id` and `geometry_revision` of its region. Select the region again from the new asset and rebuild the object; `fea.model` rejects a mismatched binding.",
    },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "Constraints overview",
        target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/constraints"),
    },
    BuiltinDocumentationLink {
        label: "Structural constraints",
        target: BuiltinDocumentationLinkTarget::Documentation(
            "/docs/simulation/constraints/structural",
        ),
    },
];

const IMPLEMENTATION: &[BuiltinDocumentationLink] = &[BuiltinDocumentationLink {
    label: "Implementation",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/constraints/structural.rs"),
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
        title: "fea.StructuralConstraint.prescribedMotion",
        summary: "Drive a support with a translational acceleration history.",
        description: "`fea.StructuralConstraint.prescribedMotion(id, region, motions)` builds a `fea.StructuralConstraint` that moves the selected faces with displacement histories reconstructed from acceleration signals, for base excitation in transient analysis.",
        keywords: &[
            "fea",
            "constraint",
            "structural",
            "support motion",
            "base excitation",
            "acceleration",
            "transient",
        ],
        related: &[
            "fea.StructuralMotion.acceleration",
            "fea.AccelerationSignal.fromTimeSamples",
            "fea.StructuralStep.transient",
            "fea.model",
        ],
        sections: SECTIONS,
        examples: EXAMPLES,
        faqs: FAQS,
        links: LINKS,
        implementation: IMPLEMENTATION,
        verification: VERIFICATION,
        notes: &[
            "Geometry examples load `block.obj`, a 0.2 m x 0.1 m x 0.1 m block whose six faces are named groups `x_min`, `x_max`, `y_min`, `y_max`, `z_min`, and `z_max`.",
        ],
    });
