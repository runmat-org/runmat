use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::model_types::fixtures::BLOCK_GEOMETRY;
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "Magnetic periodicity equates the vector potential at each primary point with that at its transformed partner, which models one sector of a machine whose adjacent sectors carry the same field. For alternating poles use the phase-shifted form.",
            "`id` names the constraint. It must be nonempty text (a character vector or string scalar) and should be unique among objects of the same kind in one model; validation messages and run evidence report it. Step activations refer to it with `fea.StepObjectRef.constraint(id)`.",
            "`primary` and `secondary` must be two distinct `geometry.Region` values from the same asset and revision; the constructor rejects regions from different geometries and a region paired with itself. Both regions are recorded as boundary-face targets.",
            "`transform` maps a point on the primary region to its partner on the secondary region: `fea.PeriodicTransform.identity()`, `.translation(translation)`, or `.rotation(origin, axis, angle)`. It must match the real periodicity of the geometry; the constructor does not infer it.",
            "`pairing` decides how entities on the two regions are matched: `fea.PeriodicPairing.matching()` requires a mesh that already matches one-to-one, and `fea.PeriodicPairing.geometric(tolerance)` pairs entities whose mapped positions agree within a `fea.Length` tolerance.",
            "All inputs are checked before anything is built: a wrong argument count, a request for more than one output, a value of the wrong class, an empty identifier, or an out-of-range value raises `RunMat:fea:InvalidInput` and no object is returned.",
            "The result is an immutable host value object, not a handle: assigning it to another variable copies the value, and its properties are read-only. It encodes portably, so it can cross an execution boundary unchanged.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Use in a study",
        paragraphs: &[
            "Pass the object to `fea.model` in the `Constraints` option, either alone or in a cell array with others of the same family, for example `model = fea.model('part', block, 'Steps', {step}, 'Constraints', {constraint})`.",
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
    id: "periodic",
    title: "Make the magnetic vector potential periodic between two face regions",
    program: "block = geometry.load('block.obj');\nleft = geometry.region(block, 'x_min');\nright = geometry.region(block, 'x_max');\nconstraint = fea.MagneticConstraint.periodic('sector', left, right, fea.PeriodicTransform.translation(fea.Translation.meters([0.2 0 0])), fea.PeriodicPairing.matching());",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeFilesystem,
    fixture: BLOCK_GEOMETRY,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(class(constraint), 'fea.MagneticConstraint'));\nassert(isfield(constraint.constraint.kind, 'periodic'));",
    },
}];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Which region is primary?",
        answer: "Either may be primary, but the transform must map the primary region onto the secondary one. Reversing the regions requires the inverse transform.",
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
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Thermal tie and acoustic periodic relations build their family kind",
        location: "crates/runmat-runtime/src/builtins/fea/constraints/coverage_tests.rs::paired_tie_and_periodic_methods_cover_thermal_and_acoustic_families",
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    documentation(ConstructorDocumentationDefinition {
        title: "fea.MagneticConstraint.periodic",
        summary: "Make the magnetic vector potential periodic between two face regions.",
        description: "`fea.MagneticConstraint.periodic(id, primary, secondary, transform, pairing)` builds a `fea.MagneticConstraint` that equates the magnetic vector potential on a primary region with its transformed partner on a secondary region.",
        keywords: &[
            "fea",
            "constraint",
            "periodic",
            "cyclic symmetry",
            "magnetic",
            "sector",
        ],
        related: &[
            "fea.MagneticConstraint.phaseShiftedPeriodic",
            "fea.PeriodicTransform.translation",
            "fea.PeriodicTransform.rotation",
            "fea.PeriodicPairing.matching",
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
