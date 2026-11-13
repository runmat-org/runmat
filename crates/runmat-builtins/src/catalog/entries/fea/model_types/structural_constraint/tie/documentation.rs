use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::model_types::fixtures::BLOCK_GEOMETRY;
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "A structural tie enforces equal displacement on paired points of the two regions, bonding them as if glued. It is the usual way to join separately meshed parts that do not share nodes. It is expressed in the global frame.",
            "`id` names the constraint. It must be nonempty text (a character vector or string scalar) and should be unique among objects of the same kind in one model; validation messages and run evidence report it. Step activations refer to it with `fea.StepObjectRef.constraint(id)`.",
            "`primary` and `secondary` must be two distinct `geometry.Region` values from the same asset and revision; the constructor rejects regions from different geometries and a region paired with itself. Both regions are recorded as boundary-face targets.",
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
    id: "tie",
    title: "Tie displacement across two nonconforming face regions",
    program: "block = geometry.load('block.obj');\na = geometry.region(block, 'x_min');\nb = geometry.region(block, 'x_max');\nconstraint = fea.StructuralConstraint.tie('glue', a, b, fea.PeriodicPairing.geometric(fea.Length.millimeters(0.01)));",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeFilesystem,
    fixture: BLOCK_GEOMETRY,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(class(constraint), 'fea.StructuralConstraint'));\nt = constraint.constraint.kind.tie;\nassert(abs(t.pairing.geometric_tolerance.meters - 1e-5) < 1e-18);\nassert(strcmp(t.secondary.region_id, 'region_x_max'));",
    },
}];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Does a tie need matching meshes?",
        answer: "No. With `fea.PeriodicPairing.geometric(tolerance)` the secondary entities are paired to the primary surface within the tolerance. Use `fea.PeriodicPairing.matching()` only when the meshes are known to match one-to-one.",
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
        title: "fea.StructuralConstraint.tie",
        summary: "Tie displacement across two nonconforming face regions.",
        description: "`fea.StructuralConstraint.tie(id, primary, secondary, pairing)` builds a `fea.StructuralConstraint` that makes the displacement continuous between two regions whose meshes need not match.",
        keywords: &[
            "fea",
            "constraint",
            "tie",
            "bonded",
            "structural",
        ],
        related: &[
            "fea.ContactInterface.frictionless",
            "fea.PeriodicPairing.geometric",
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
