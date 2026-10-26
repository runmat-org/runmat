use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::model_types::fixtures::BLOCK_GEOMETRY;
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "The target selects every beam domain of the asset rather than one region (`target.beam` is `'all'`), which is the shortest way to give a single-material part its material. It records the asset's `geometry_id` and `geometry_revision` (a read-only `uint32` property), so `fea.model` rejects it for a different asset or revision.",
            "`geometry` must be a `geometry.Asset`, for example from `geometry.load`.",
            "All inputs are checked before anything is built: a wrong argument count, a request for more than one output, a value of the wrong class, an empty identifier, or an out-of-range value raises `RunMat:fea:InvalidInput` and no object is returned.",
            "The result is an immutable host value object, not a handle: assigning it to another variable copies the value, and its properties are read-only. It encodes portably, so it can cross an execution boundary unchanged.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Use in a study",
        paragraphs: &[
            "Pass the target as the first argument of `fea.materialAssignment(target, material)` and pass the assignment to `fea.model` in the `MaterialAssignments` option, with the material in `Materials`.",
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
    id: "allBeams",
    title: "Target every beam domain of a geometry",
    program: "block = geometry.load('block.obj');\ntarget = fea.PropertyRegionTarget.allBeams(block);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeFilesystem,
    fixture: BLOCK_GEOMETRY,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(class(target), 'fea.PropertyRegionTarget'));\nassert(strcmp(target.target.beam, 'all'));\nassert(strcmp(target.geometry_id, block.geometry_id));",
    },
}];

const FAQS: &[BuiltinDocumentationFaq] = &[];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "Geometry, models, and assignments",
        target: BuiltinDocumentationLinkTarget::Documentation(
            "/docs/simulation/language/geometry-models-and-assignments",
        ),
    },
    BuiltinDocumentationLink {
        label: "FEA models",
        target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/models"),
    },
];

const IMPLEMENTATION: &[BuiltinDocumentationLink] = &[BuiltinDocumentationLink {
    label: "Implementation",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/material_assignments.rs"),
}];

const VERIFICATION: &[BuiltinEvidenceReference] = &[
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Typed constructors build full study objects, including property-region targets and material assignments",
        location: "crates/runmat-runtime/src/builtins/fea/mod.rs::tests::typed_constructors_build_full_study_and_sweep_objects",
    },
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Geometry-bound model objects share the uint32 revision class",
        location: "crates/runmat-runtime/src/builtins/fea/constraints/coverage_tests.rs::geometry_bound_model_objects_share_the_uint32_revision_class",
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    documentation(ConstructorDocumentationDefinition {
        title: "fea.PropertyRegionTarget.allBeams",
        summary: "Target every beam domain of a geometry.",
        description: "`fea.PropertyRegionTarget.allBeams(geometry)` returns a `fea.PropertyRegionTarget` that selects all beam domains of one geometry asset.",
        keywords: &[
            "fea",
            "material",
            "property region",
            "beam",
            "all",
        ],
        related: &[
            "fea.PropertyRegionTarget.beam",
            "fea.materialAssignment",
            "fea.material",
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
