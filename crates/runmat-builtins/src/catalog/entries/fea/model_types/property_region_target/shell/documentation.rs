use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::model_types::fixtures::BLOCK_GEOMETRY;
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "A property-region target is a material-bearing domain, deliberately distinct from a boundary selection: the same CAD face can be a shell domain or a boundary face, and the target says which. This target selects one region by id as a shell domain (`target.shell.region.region_id`).",
            "`region` must be a `geometry.Region` returned by `geometry.region(asset, selector)`. The object stores the region id together with the region's `geometry_id` and `geometry_revision` (exposed as read-only properties, the revision as `uint32`), so `fea.model` rejects it when the model is built on a different asset or a different revision of the same asset.",
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
    id: "shell",
    title: "Target one geometry region as a shell material domain",
    program: "block = geometry.load('block.obj');\ntarget = fea.PropertyRegionTarget.shell(geometry.region(block, 'z_min'));",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeFilesystem,
    fixture: BLOCK_GEOMETRY,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(class(target), 'fea.PropertyRegionTarget'));\nassert(strcmp(target.target.shell.region.region_id, 'region_z_min'));\nassert(isa(target.geometry_revision, 'uint32'));",
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
        title: "fea.PropertyRegionTarget.shell",
        summary: "Target one geometry region as a shell material domain.",
        description: "`fea.PropertyRegionTarget.shell(region)` returns a `fea.PropertyRegionTarget` that marks one region as a surface region modeled with shell elements, carrying a thickness.",
        keywords: &[
            "fea",
            "material",
            "property region",
            "shell",
            "domain",
        ],
        related: &[
            "fea.PropertyRegionTarget.allShells",
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
