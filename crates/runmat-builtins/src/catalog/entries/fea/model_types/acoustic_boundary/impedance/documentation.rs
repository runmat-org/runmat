use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::model_types::fixtures::BLOCK_GEOMETRY;
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "The boundary relates surface pressure to normal particle velocity, Z = p / v_n, with a real specific acoustic impedance in Pa s/m. Z equal to the characteristic impedance of the fluid (about 413 Pa s/m for air at 20 degrees C) absorbs normally incident waves; larger values approach a rigid wall.",
            "`id` names the boundary. It must be nonempty text (a character vector or string scalar) and should be unique among objects of the same kind in one model; validation messages and run evidence report it.",
            "`region` must be a `geometry.Region` returned by `geometry.region(asset, selector)`. The object stores the region id together with the region's `geometry_id` and `geometry_revision` (exposed as read-only properties, the revision as `uint32`), so `fea.model` rejects it when the model is built on a different asset or a different revision of the same asset.",
            "The region is recorded as a boundary-face target (`entity_dimension` `face`, cardinality one or more faces). Whether the selected region really is a set of faces is checked when the model is meshed and validated, not by the constructor.",
            "`impedance` must be a `fea.SpecificAcousticImpedance` in Pa s/m, for example `fea.SpecificAcousticImpedance.pascalSecondsPerMeter(413)`.",
            "Physical values are passed as typed quantity objects (for example `fea.Temperature.celsius(25)`), never as bare numbers, so the unit is explicit at the call site. The quantity constructors convert to SI once and the object stores SI values.",
            "All inputs are checked before anything is built: a wrong argument count, a request for more than one output, a value of the wrong class, an empty identifier, or an out-of-range value raises `RunMat:fea:InvalidInput` and no object is returned.",
            "The result is an immutable host value object, not a handle: assigning it to another variable copies the value, and its properties are read-only. It encodes portably, so it can cross an execution boundary unchanged.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Use in a study",
        paragraphs: &[
            "Pass the object to `fea.model` in the `Boundaries` option, either alone or in a cell array with others of the same family, for example `model = fea.model('part', block, 'Steps', {step}, 'Boundaries', {boundary})`. A boundary is a natural (operator) condition that stays constant for the analysis; for a value that varies with time or frequency, use the matching `fea.*Excitation` method with a signal instead.",
            "Use it with an acoustic step to model liners, porous treatments, or a measured wall impedance.",
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
    id: "impedance-boundary",
    title: "Create a specific acoustic-impedance boundary on a face region",
    program: "block = geometry.load('block.obj');\nliner = geometry.region(block, 'y_max');\nZ = fea.SpecificAcousticImpedance.pascalSecondsPerMeter(413);\nboundary = fea.AcousticBoundary.impedance('liner', liner, Z);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeFilesystem,
    fixture: BLOCK_GEOMETRY,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(class(boundary), 'fea.AcousticBoundary'));\nassert(boundary.boundary.operator.acoustic_impedance.impedance.pascal_seconds_per_meter == 413);",
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
        label: "FEA models",
        target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/models"),
    },
    BuiltinDocumentationLink {
        label: "Acoustic constraints",
        target: BuiltinDocumentationLinkTarget::Documentation(
            "/docs/simulation/constraints/acoustic",
        ),
    },
];

const IMPLEMENTATION: &[BuiltinDocumentationLink] = &[BuiltinDocumentationLink {
    label: "Implementation",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/fea/boundaries.rs"),
}];

const VERIFICATION: &[BuiltinEvidenceReference] = &[
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Every typed boundary constructor builds a geometry-bound, portable value",
        location: "crates/runmat-runtime/src/builtins/fea/boundaries.rs::tests::family_boundary_constructors_produce_typed_geometry_bound_values",
    },
    BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Model composition rejects boundaries from another geometry or a mislabeled family",
        location: "crates/runmat-runtime/src/builtins/fea/boundaries.rs::tests::model_composition_separates_boundaries_and_rejects_foreign_geometry",
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation =
    documentation(ConstructorDocumentationDefinition {
        title: "fea.AcousticBoundary.impedance",
        summary: "Create a specific acoustic-impedance boundary on a face region.",
        description: "`fea.AcousticBoundary.impedance(id, region, impedance)` builds a `fea.AcousticBoundary` with a locally reacting surface impedance.",
        keywords: &[
            "fea",
            "boundary",
            "acoustic",
            "impedance",
            "absorption",
            "liner",
        ],
        related: &[
            "fea.AcousticBoundary.radiation",
            "fea.SpecificAcousticImpedance.pascalSecondsPerMeter",
            "fea.model",
            "geometry.region",
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
