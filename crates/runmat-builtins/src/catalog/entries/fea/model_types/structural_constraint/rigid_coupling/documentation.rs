use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::model_types::fixtures::BLOCK_GEOMETRY;
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "Every point x on the region moves as u(x) = u_r + theta_r x (x - x_r), where u_r and theta_r are the translation and rotation of the reference point at x_r. The region cannot deform, which adds local stiffness; use it where a much stiffer part (a bolt head, a shaft hub) really drives the face.",
            "`id` names the constraint. It must be nonempty text (a character vector or string scalar) and should be unique among objects of the same kind in one model; validation messages and run evidence report it. Step activations refer to it with `fea.StepObjectRef.constraint(id)`.",
            "`region` must be a `geometry.Region` returned by `geometry.region(asset, selector)`. The object stores the region id together with the region's `geometry_id` and `geometry_revision` (exposed as read-only properties, the revision as `uint32`), so `fea.model` rejects it when the model is built on a different asset or a different revision of the same asset.",
            "The region is recorded as a boundary-face target (`entity_dimension` `face`, cardinality one or more faces). Whether the selected region really is a set of faces is checked when the model is meshed and validated, not by the constructor.",
            "`referencePoint` must be a `fea.ReferencePoint` from `fea.ReferencePoint.fromCoordinates(id, position)`; its position may lie off the geometry.",
            "All inputs are checked before anything is built: a wrong argument count, a request for more than one output, a value of the wrong class, an empty identifier, or an out-of-range value raises `RunMat:fea:InvalidInput` and no object is returned.",
            "The result is an immutable host value object, not a handle: assigning it to another variable copies the value, and its properties are read-only. It encodes portably, so it can cross an execution boundary unchanged.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Use in a study",
        paragraphs: &[
            "Pass the object to `fea.model` in the `Constraints` option, either alone or in a cell array with others of the same family, for example `model = fea.model('part', block, 'Steps', {step}, 'Constraints', {constraint})`.",
            "Loads and supports applied to the reference point are transferred to the region through the coupling.",
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
    id: "rigidCoupling",
    title: "Rigidly couple a face region to a reference point",
    program: "block = geometry.load('block.obj');\nhub = fea.ReferencePoint.fromCoordinates('hub', fea.Position.meters([0.25 0.05 0.05]));\nface = geometry.region(block, 'x_max');\nconstraint = fea.StructuralConstraint.rigidCoupling('hub_link', face, hub);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeFilesystem,
    fixture: BLOCK_GEOMETRY,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(class(constraint), 'fea.StructuralConstraint'));\nc = constraint.constraint.kind.rigid_coupling;\nassert(strcmp(c.reference_point.reference_point_id, 'hub'));\nassert(max(abs(c.reference_point.position.meters - [0.25 0.05 0.05])) < 1e-12);",
    },
}];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Rigid or distributing coupling?",
        answer: "Rigid coupling forces the face to move rigidly and stiffens it. Distributing coupling transfers the reference-point load to the face as a weighted distribution and lets the face deform; prefer it when the driving part is not much stiffer than the face.",
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
        label: "Connectors and couplings",
        target: BuiltinDocumentationLinkTarget::Documentation(
            "/docs/simulation/constraints/connectors",
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
        label: "Prescribed DOFs and reference-point couplings are fully typed",
        location: "crates/runmat-runtime/src/builtins/fea/constraints.rs::tests::prescribed_structural_dofs_and_reference_couplings_are_fully_typed",
    },
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Every region constraint method builds its family kind",
        location: "crates/runmat-runtime/src/builtins/fea/constraints/coverage_tests.rs::every_region_constraint_method_builds_its_family_kind",
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    documentation(ConstructorDocumentationDefinition {
        title: "fea.StructuralConstraint.rigidCoupling",
        summary: "Rigidly couple a face region to a reference point.",
        description: "`fea.StructuralConstraint.rigidCoupling(id, region, referencePoint)` builds a `fea.StructuralConstraint` that makes the selected faces follow the rigid-body motion of a reference point.",
        keywords: &[
            "fea",
            "constraint",
            "structural",
            "rigid coupling",
            "reference point",
            "kinematic coupling",
        ],
        related: &[
            "fea.StructuralConstraint.distributingCoupling",
            "fea.ReferencePoint.fromCoordinates",
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
