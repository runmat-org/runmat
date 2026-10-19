use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::model_types::fixtures::BLOCK_GEOMETRY;
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "On the interface the fluid velocity equals the structural velocity (no slip, kinematic continuity) and the fluid traction loads the structure (dynamic continuity). The argument order matters: the first region is the fluid side and the second the structure side.",
            "`id` names the interface. It must be nonempty text (a character vector or string scalar) and should be unique among objects of the same kind in one model; validation messages and run evidence report it. Step activations refer to it with `fea.StepObjectRef.interface(id)`.",
            "`fluidRegion` and `structureRegion` must be two distinct `geometry.Region` values from the same asset and revision; regions from different geometries, or one region given twice, are rejected. Both are recorded as boundary-face targets, the first as the interface's primary side.",
            "Each region must be a `geometry.Region` returned by `geometry.region(asset, selector)`. The object stores the region id together with the region's `geometry_id` and `geometry_revision` (exposed as read-only properties, the revision as `uint32`), so `fea.model` rejects it when the model is built on a different asset or a different revision of the same asset.",
            "All inputs are checked before anything is built: a wrong argument count, a request for more than one output, a value of the wrong class, an empty identifier, or an out-of-range value raises `RunMat:fea:InvalidInput` and no object is returned.",
            "The result is an immutable host value object, not a handle: assigning it to another variable copies the value, and its properties are read-only. It encodes portably, so it can cross an execution boundary unchanged.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Use in a study",
        paragraphs: &[
            "Pass the object to `fea.model` in the `Interfaces` option, either alone or in a cell array with others of the same family, for example `model = fea.model('part', block, 'Steps', {step}, 'Interfaces', {interface})`.",
            "Use it with `fea.FluidStructureStep.transient`, with a fluid domain from `fea.FluidDomain.newtonian` and structural materials on the solid side.",
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
    id: "noSlip",
    title: "Create a no-slip fluid-structure interface between a fluid face and a structural face",
    program: "block = geometry.load('block.obj');\na = geometry.region(block, 'x_min');\nb = geometry.region(block, 'x_max');\ninterface = fea.FluidStructureInterface.noSlip('wetted_wall', a, b);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeFilesystem,
    fixture: BLOCK_GEOMETRY,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(class(interface), 'fea.FluidStructureInterface'));\nassert(strcmp(interface.interface.kind.fluid_structure.kinematics, 'no_slip'));",
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
        label: "Interfaces",
        target: BuiltinDocumentationLinkTarget::Documentation(
            "/docs/simulation/constraints/interfaces",
        ),
    },
    BuiltinDocumentationLink {
        label: "Fluid-structure interaction",
        target: BuiltinDocumentationLinkTarget::Documentation(
            "/docs/simulation/physics/fluid-structure-interaction",
        ),
    },
];

const IMPLEMENTATION: &[BuiltinDocumentationLink] = &[BuiltinDocumentationLink {
    label: "Implementation",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/interfaces.rs"),
}];

const VERIFICATION: &[BuiltinEvidenceReference] = &[
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Concrete interfaces are geometry-bound, portable, and class-checked",
        location: "crates/runmat-runtime/src/builtins/fea/interfaces.rs::tests::concrete_interfaces_are_geometry_bound_portable_and_class_checked",
    },
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Interface construction rejects invalid identity and pairing",
        location: "crates/runmat-runtime/src/builtins/fea/interfaces.rs::tests::interface_construction_rejects_invalid_identity_and_pairing",
    },
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Model composition accepts matching interfaces and rejects foreign geometry",
        location: "crates/runmat-runtime/src/builtins/fea/interfaces.rs::tests::model_composition_accepts_matching_interfaces_and_rejects_foreign_geometry",
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    documentation(ConstructorDocumentationDefinition {
        title: "fea.FluidStructureInterface.noSlip",
        summary: "Create a no-slip fluid-structure interface between a fluid face and a structural face.",
        description: "`fea.FluidStructureInterface.noSlip(id, fluidRegion, structureRegion)` builds a `fea.FluidStructureInterface` that couples a fluid domain to a deforming structure.",
        keywords: &[
            "fea",
            "interface",
            "fluid-structure interaction",
            "FSI",
            "no slip",
        ],
        related: &[
            "fea.FluidStructureStep.transient",
            "fea.FluidDomain.newtonian",
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
