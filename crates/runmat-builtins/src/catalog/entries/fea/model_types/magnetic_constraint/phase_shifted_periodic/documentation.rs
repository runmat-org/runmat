use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::model_types::fixtures::BLOCK_GEOMETRY;
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "This models a machine or transformer sector in a harmonic electromagnetic analysis where adjacent sectors differ by a fixed electrical phase, including the antiperiodic case (180 degrees) for alternating poles. The secondary value equals the primary value multiplied by exp(i phi), where phi is the stored phase in radians. A phase of 0 reduces to the plain periodic relation.",
            "`id` names the constraint. It must be nonempty text (a character vector or string scalar) and should be unique among objects of the same kind in one model; validation messages and run evidence report it. Step activations refer to it with `fea.StepObjectRef.constraint(id)`.",
            "`primary` and `secondary` must be two distinct `geometry.Region` values from the same asset and revision; the constructor rejects regions from different geometries and a region paired with itself. Both regions are recorded as boundary-face targets.",
            "`transform` maps a point on the primary region to its partner on the secondary region: `fea.PeriodicTransform.identity()`, `.translation(translation)`, or `.rotation(origin, axis, angle)`. It must match the real periodicity of the geometry; the constructor does not infer it.",
            "`pairing` decides how entities on the two regions are matched: `fea.PeriodicPairing.matching()` requires a mesh that already matches one-to-one, and `fea.PeriodicPairing.geometric(tolerance)` pairs entities whose mapped positions agree within a `fea.Length` tolerance.",
            "`phase` must be a `fea.ComplexPhase` (`fea.ComplexPhase.degrees(90)` stores pi/2 rad).",
            "All inputs are checked before anything is built: a wrong argument count, a request for more than one output, a value of the wrong class, an empty identifier, or an out-of-range value raises `RunMat:fea:InvalidInput` and no object is returned.",
            "The result is an immutable host value object, not a handle: assigning it to another variable copies the value, and its properties are read-only. It encodes portably, so it can cross an execution boundary unchanged.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Use in a study",
        paragraphs: &[
            "Pass the object to `fea.model` in the `Constraints` option, either alone or in a cell array with others of the same family, for example `model = fea.model('part', block, 'Steps', {step}, 'Constraints', {constraint})`.",
            "Use it with `fea.ElectromagneticStep.harmonic`.",
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
    id: "phaseShiftedPeriodic",
    title: "Make the complex harmonic magnetic vector potential periodic with a phase shift",
    program: "block = geometry.load('block.obj');\nleft = geometry.region(block, 'x_min');\nright = geometry.region(block, 'x_max');\nconstraint = fea.MagneticConstraint.phaseShiftedPeriodic('pole_pair', left, right, fea.PeriodicTransform.translation(fea.Translation.meters([0.2 0 0])), fea.PeriodicPairing.matching(), fea.ComplexPhase.degrees(180));",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeFilesystem,
    fixture: BLOCK_GEOMETRY,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(class(constraint), 'fea.MagneticConstraint'));\nassert(abs(constraint.constraint.kind.phase_shifted_periodic.phase.radians - pi) < 1e-12);",
    },
}];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "How do I choose the phase?",
        answer: "The phase is set by the physics, not the geometry: for a rotating-machine sector it follows from pole count, sector angle, and harmonic; for a periodic acoustic or wave cell it is the wavenumber times the cell period. RunMat does not infer it.",
    },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "Constraints overview",
        target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/constraints"),
    },
    BuiltinDocumentationLink {
        label: "Electromagnetic constraints",
        target: BuiltinDocumentationLinkTarget::Documentation(
            "/docs/simulation/constraints/electromagnetic",
        ),
    },
];

const IMPLEMENTATION: &[BuiltinDocumentationLink] = &[BuiltinDocumentationLink {
    label: "Implementation",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/constraints/magnetic.rs"),
}];

const VERIFICATION: &[BuiltinEvidenceReference] = &[
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Paired tie, periodic, phase-shifted, and affine-jump relations are typed and dimensionally checked",
        location: "crates/runmat-runtime/src/builtins/fea/constraints.rs::tests::paired_relations_are_typed_transformed_and_dimensionally_sound",
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    documentation(ConstructorDocumentationDefinition {
        title: "fea.MagneticConstraint.phaseShiftedPeriodic",
        summary: "Make the complex harmonic magnetic vector potential periodic with a phase shift.",
        description: "`fea.MagneticConstraint.phaseShiftedPeriodic(id, primary, secondary, transform, pairing, phase)` builds a `fea.MagneticConstraint` that relates the harmonic magnetic vector potential on two regions by a Bloch-Floquet phase factor.",
        keywords: &[
            "fea",
            "constraint",
            "periodic",
            "phase",
            "Bloch",
            "Floquet",
            "magnetic",
        ],
        related: &[
            "fea.MagneticConstraint.periodic",
            "fea.ElectromagneticStep.harmonic",
            "fea.ComplexPhase.degrees",
            "fea.PeriodicTransform.rotation",
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
