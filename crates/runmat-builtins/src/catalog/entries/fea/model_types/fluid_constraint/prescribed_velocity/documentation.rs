use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::model_types::fixtures::BLOCK_GEOMETRY;
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "All three components of the velocity are prescribed (an essential condition). The vector is signed, so the direction of flow relative to the face is explicit: for an inlet on a face with outward normal -x, an inflow along +x is a positive x component.",
            "`id` names the constraint. It must be nonempty text (a character vector or string scalar) and should be unique among objects of the same kind in one model; validation messages and run evidence report it. Step activations refer to it with `fea.StepObjectRef.constraint(id)`.",
            "`region` must be a `geometry.Region` returned by `geometry.region(asset, selector)`. The object stores the region id together with the region's `geometry_id` and `geometry_revision` (exposed as read-only properties, the revision as `uint32`), so `fea.model` rejects it when the model is built on a different asset or a different revision of the same asset.",
            "The region is recorded as a boundary-face target (`entity_dimension` `face`, cardinality one or more faces). Whether the selected region really is a set of faces is checked when the model is meshed and validated, not by the constructor.",
            "`velocity` is either a 1-by-3 or 3-by-1 cell array of three `fea.Velocity` values (global x, y, z in m/s), which is stored as a constant vector, or a framed three-component `fea.VelocitySignal`, which is stored by reference to its signal id and is not sampled or collapsed to a magnitude. A scalar speed is rejected: the direction must be explicit. Add a signal to the model's `Signals` option.",
            "All inputs are checked before anything is built: a wrong argument count, a request for more than one output, a value of the wrong class, an empty identifier, or an out-of-range value raises `RunMat:fea:InvalidInput` and no object is returned.",
            "The result is an immutable host value object, not a handle: assigning it to another variable copies the value, and its properties are read-only. It encodes portably, so it can cross an execution boundary unchanged.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Use in a study",
        paragraphs: &[
            "Pass the object to `fea.model` in the `Constraints` option, either alone or in a cell array with others of the same family, for example `model = fea.model('part', block, 'Steps', {step}, 'Constraints', {constraint})`.",
            "Use it with a fluid step and an outlet such as `fea.FluidBoundary.outletPressure`.",
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
    id: "prescribedVelocity",
    title: "Prescribe the full fluid velocity vector on a face region",
    program: "block = geometry.load('block.obj');\ninlet = geometry.region(block, 'x_min');\nv = {fea.Velocity.metersPerSecond(1.5), fea.Velocity.metersPerSecond(0), fea.Velocity.metersPerSecond(0)};\nconstraint = fea.FluidConstraint.prescribedVelocity('inlet', inlet, v);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeFilesystem,
    fixture: BLOCK_GEOMETRY,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(class(constraint), 'fea.FluidConstraint'));\nc = constraint.constraint.kind.prescribed_velocity.value.constant;\nassert(numel(c) == 3);\nassert(c(1).meters_per_second == 1.5);",
    },
}];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Can I pass an inlet speed as one number?",
        answer: "No. A scalar speed does not say which way the fluid moves, so the constructor rejects it. Pass three `fea.Velocity` components or a three-component velocity signal.",
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
        label: "Fluid constraints",
        target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/constraints/fluid"),
    },
];

const IMPLEMENTATION: &[BuiltinDocumentationLink] = &[BuiltinDocumentationLink {
    label: "Implementation",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/constraints/fluid.rs"),
}];

const VERIFICATION: &[BuiltinEvidenceReference] = &[
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Fluid vectors keep three signed typed components and reject a scalar speed",
        location: "crates/runmat-runtime/src/builtins/fea/constraints/fluid/vector_tests.rs::runtime_fluid_vector_keeps_three_signed_typed_components_and_rejects_scalar_speed",
    },
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "A fluid vector signal is kept as a reference, not a magnitude",
        location: "crates/runmat-runtime/src/builtins/fea/constraints/fluid/vector_tests.rs::runtime_fluid_vector_signal_is_an_original_reference_not_a_magnitude",
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    documentation(ConstructorDocumentationDefinition {
        title: "fea.FluidConstraint.prescribedVelocity",
        summary: "Prescribe the full fluid velocity vector on a face region.",
        description: "`fea.FluidConstraint.prescribedVelocity(id, region, velocity)` builds a `fea.FluidConstraint` that holds all three velocity components on the selected faces, typically a velocity inlet.",
        keywords: &[
            "fea",
            "constraint",
            "fluid",
            "velocity inlet",
            "CFD",
        ],
        related: &[
            "fea.FluidExcitation.inletVelocity",
            "fea.FluidBoundary.outletPressure",
            "fea.Velocity.metersPerSecond",
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
