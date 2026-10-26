use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::model_types::fixtures::BLOCK_GEOMETRY;
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "Each listed component is held at its value; components that are not listed stay free. This expresses rollers (one translation held at zero), enforced displacements, and supports in a cylindrical or skewed frame. Translations are stored in meters and rotations in radians.",
            "`id` names the constraint. It must be nonempty text (a character vector or string scalar) and should be unique among objects of the same kind in one model; validation messages and run evidence report it. Step activations refer to it with `fea.StepObjectRef.constraint(id)`.",
            "`region` must be a `geometry.Region` returned by `geometry.region(asset, selector)`. The object stores the region id together with the region's `geometry_id` and `geometry_revision` (exposed as read-only properties, the revision as `uint32`), so `fea.model` rejects it when the model is built on a different asset or a different revision of the same asset.",
            "The region is recorded as a boundary-face target (`entity_dimension` `face`, cardinality one or more faces). Whether the selected region really is a set of faces is checked when the model is meshed and validated, not by the constructor.",
            "`values` is one `fea.StructuralDofValue` or a cell array of them, built with `fea.StructuralDofValue.translation(fea.StructuralDof.uz(), fea.Displacement.millimeters(-0.5))` or `fea.StructuralDofValue.rotation(fea.StructuralDof.rx(), fea.Rotation.degrees(0))`. At least one value is required and a DOF may appear only once.",
            "`frame` is a `fea.CoordinateFrame`; the DOF axes (ux, uy, uz, rx, ry, rz) are the axes of that frame. Use `fea.CoordinateFrame.globalFrame()` for global axes or `fea.CoordinateFrame.cylindrical(...)` for radial, tangential, and axial components.",
            "All inputs are checked before anything is built: a wrong argument count, a request for more than one output, a value of the wrong class, an empty identifier, or an out-of-range value raises `RunMat:fea:InvalidInput` and no object is returned.",
            "The result is an immutable host value object, not a handle: assigning it to another variable copies the value, and its properties are read-only. It encodes portably, so it can cross an execution boundary unchanged.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Use in a study",
        paragraphs: &[
            "Pass the object to `fea.model` in the `Constraints` option, either alone or in a cell array with others of the same family, for example `model = fea.model('part', block, 'Steps', {step}, 'Constraints', {constraint})`.",
            "Prescribed rotations only constrain formulations that carry rotational degrees of freedom; a solid (continuum) model has none.",
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
    id: "prescribed",
    title: "Prescribe selected displacement and rotation components on a face region",
    program: "block = geometry.load('block.obj');\nbase = geometry.region(block, 'z_min');\nuz0 = fea.StructuralDofValue.translation(fea.StructuralDof.uz(), fea.Displacement.meters(0));\nconstraint = fea.StructuralConstraint.prescribed('roller', base, {uz0}, fea.CoordinateFrame.globalFrame());",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeFilesystem,
    fixture: BLOCK_GEOMETRY,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(class(constraint), 'fea.StructuralConstraint'));\np = constraint.constraint.kind.prescribed;\nassert(strcmp(p.translations.dof, 'uz'));\nassert(p.translations.value.constant.meters == 0);\nassert(isempty(p.rotations));",
    },
}];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "How do I model a roller support?",
        answer: "Prescribe only the normal translation, for example `uz = 0` on a face whose normal is z, and leave the in-plane components free.",
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
        label: "Prescribed DOFs and reference-point couplings are fully typed",
        location: "crates/runmat-runtime/src/builtins/fea/constraints.rs::tests::prescribed_structural_dofs_and_reference_couplings_are_fully_typed",
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    documentation(ConstructorDocumentationDefinition {
        title: "fea.StructuralConstraint.prescribed",
        summary: "Prescribe selected displacement and rotation components on a face region.",
        description: "`fea.StructuralConstraint.prescribed(id, region, values, frame)` builds a `fea.StructuralConstraint` that prescribes only the listed degrees of freedom, measured in a chosen coordinate frame.",
        keywords: &[
            "fea",
            "constraint",
            "structural",
            "prescribed",
            "roller",
            "enforced displacement",
            "frame",
        ],
        related: &[
            "fea.StructuralDofValue.translation",
            "fea.StructuralDofValue.rotation",
            "fea.StructuralDof.ux",
            "fea.CoordinateFrame.globalFrame",
            "fea.CoordinateFrame.cylindrical",
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
