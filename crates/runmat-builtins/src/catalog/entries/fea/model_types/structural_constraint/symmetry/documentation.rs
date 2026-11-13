use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::model_types::fixtures::BLOCK_GEOMETRY;
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "On a symmetry plane the displacement normal to the plane is zero, and the rotations compatible with mirror symmetry are restrained where the formulation carries rotations. Modeling half or a quarter of a symmetric part with symmetry planes gives the same answer as the full model when geometry, material, loads, and supports are all symmetric.",
            "`id` names the constraint. It must be nonempty text (a character vector or string scalar) and should be unique among objects of the same kind in one model; validation messages and run evidence report it. Step activations refer to it with `fea.StepObjectRef.constraint(id)`.",
            "`region` must be a `geometry.Region` returned by `geometry.region(asset, selector)`. The object stores the region id together with the region's `geometry_id` and `geometry_revision` (exposed as read-only properties, the revision as `uint32`), so `fea.model` rejects it when the model is built on a different asset or a different revision of the same asset.",
            "The region is recorded as a boundary-face target (`entity_dimension` `face`, cardinality one or more faces). Whether the selected region really is a set of faces is checked when the model is meshed and validated, not by the constructor.",
            "All inputs are checked before anything is built: a wrong argument count, a request for more than one output, a value of the wrong class, an empty identifier, or an out-of-range value raises `RunMat:fea:InvalidInput` and no object is returned.",
            "The result is an immutable host value object, not a handle: assigning it to another variable copies the value, and its properties are read-only. It encodes portably, so it can cross an execution boundary unchanged.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Use in a study",
        paragraphs: &[
            "Pass the object to `fea.model` in the `Constraints` option, either alone or in a cell array with others of the same family, for example `model = fea.model('part', block, 'Steps', {step}, 'Constraints', {constraint})`.",
            "Do not use symmetry for modal or buckling studies unless the modes of interest are themselves symmetric: antisymmetric modes are suppressed.",
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
    id: "symmetry",
    title: "Apply a structural symmetry condition on a planar face region",
    program: "block = geometry.load('block.obj');\nmid = geometry.region(block, 'y_min');\nconstraint = fea.StructuralConstraint.symmetry('mid_plane', mid);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeFilesystem,
    fixture: BLOCK_GEOMETRY,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(class(constraint), 'fea.StructuralConstraint'));\nassert(isfield(constraint.constraint.kind, 'symmetry'));",
    },
}];

const FAQS: &[BuiltinDocumentationFaq] = &[
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
        label: "Simple family constraints compose through their typed classes",
        location: "crates/runmat-runtime/src/builtins/fea/constraints.rs::tests::every_simple_family_constraint_composes_through_typed_classes",
    },
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Model composition accepts matching geometry and rejects a foreign region",
        location: "crates/runmat-runtime/src/builtins/fea/constraints.rs::tests::model_composition_accepts_matching_geometry_and_rejects_a_foreign_region",
    },
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Every region constraint method builds its family kind",
        location: "crates/runmat-runtime/src/builtins/fea/constraints/coverage_tests.rs::every_region_constraint_method_builds_its_family_kind",
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    documentation(ConstructorDocumentationDefinition {
        title: "fea.StructuralConstraint.symmetry",
        summary: "Apply a structural symmetry condition on a planar face region.",
        description: "`fea.StructuralConstraint.symmetry(id, region)` builds a `fea.StructuralConstraint` for a mirror-symmetry plane.",
        keywords: &[
            "fea",
            "constraint",
            "structural",
            "symmetry",
            "mirror",
        ],
        related: &[
            "fea.StructuralConstraint.antisymmetry",
            "fea.StructuralConstraint.prescribed",
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
