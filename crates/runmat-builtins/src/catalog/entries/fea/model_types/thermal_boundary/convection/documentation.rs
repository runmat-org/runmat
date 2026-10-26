use crate::fea_contract::catalog_entry::{documentation, ConstructorDocumentationDefinition};
use crate::fea_contract::model_types::fixtures::BLOCK_GEOMETRY;
use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Behavior",
        paragraphs: &[
            "The boundary adds the Robin condition q = h (T - T_amb) to the heat equation, where q is the heat flux leaving the body through the outward normal, h is the film coefficient, and T_amb is the ambient (free-stream) temperature. Assembly contributes both a boundary stiffness term (h on the face mass matrix) and a source term (h T_amb), so a hot surface loses heat and a surface colder than ambient gains heat.",
            "`id` names the boundary. It must be nonempty text (a character vector or string scalar) and should be unique among objects of the same kind in one model; validation messages and run evidence report it.",
            "`region` must be a `geometry.Region` returned by `geometry.region(asset, selector)`. The object stores the region id together with the region's `geometry_id` and `geometry_revision` (exposed as read-only properties, the revision as `uint32`), so `fea.model` rejects it when the model is built on a different asset or a different revision of the same asset.",
            "The region is recorded as a boundary-face target (`entity_dimension` `face`, cardinality one or more faces). Whether the selected region really is a set of faces is checked when the model is meshed and validated, not by the constructor.",
            "`ambientTemperature` must be a `fea.Temperature` (stored in kelvin; `fea.Temperature.celsius(25)` stores 298.15 K). `coefficient` must be a `fea.HeatTransferCoefficient` in W/(m^2 K). Natural convection in air is typically 5 to 25 W/(m^2 K); forced air 25 to 250; liquid cooling reaches several thousand.",
            "Physical values are passed as typed quantity objects (for example `fea.Temperature.celsius(25)`), never as bare numbers, so the unit is explicit at the call site. The quantity constructors convert to SI once and the object stores SI values.",
            "All inputs are checked before anything is built: a wrong argument count, a request for more than one output, a value of the wrong class, an empty identifier, or an out-of-range value raises `RunMat:fea:InvalidInput` and no object is returned.",
            "The result is an immutable host value object, not a handle: assigning it to another variable copies the value, and its properties are read-only. It encodes portably, so it can cross an execution boundary unchanged.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Use in a study",
        paragraphs: &[
            "Pass the object to `fea.model` in the `Boundaries` option, either alone or in a cell array with others of the same family, for example `model = fea.model('part', block, 'Steps', {step}, 'Boundaries', {boundary})`. A boundary is a natural (operator) condition that stays constant for the analysis; for a value that varies with time or frequency, use the matching `fea.*Excitation` method with a signal instead.",
            "Convection needs a thermal or coupled thermal step (`fea.ThermalStep.steady`, `fea.ThermalStep.transient`, or a thermo-mechanical, electro-thermal, or conjugate-heat-transfer step). For an ambient temperature that changes over time, bind a temperature signal with `fea.ThermalExcitation.ambientTemperature`.",
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
    id: "convection-boundary",
    title: "Create a convective (Robin) heat-transfer boundary on a face region",
    program: "block = geometry.load('block.obj');\nlid = geometry.region(block, 'z_max');\nambient = fea.Temperature.celsius(25);\nh = fea.HeatTransferCoefficient.wattsPerSquareMeterKelvin(35);\nboundary = fea.ThermalBoundary.convection('lid_cooling', lid, ambient, h);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::NativeFilesystem,
    fixture: BLOCK_GEOMETRY,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(class(boundary), 'fea.ThermalBoundary'));\nop = boundary.boundary.operator.thermal_convection;\nassert(abs(op.ambient_temperature.constant.kelvin - 298.15) < 1e-9);\nassert(op.coefficient.watts_per_square_meter_kelvin == 35);\nassert(strcmp(boundary.boundary.target.region_id, 'region_z_max'));",
    },
}];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq {
        question: "Can I reuse the object with another geometry?",
        answer: "No. The object is bound to the `geometry_id` and `geometry_revision` of its region. Select the region again from the new asset and rebuild the object; `fea.model` rejects a mismatched binding.",
    },
    BuiltinDocumentationFaq {
        question: "Is the ambient temperature an absolute temperature?",
        answer: "Yes. It is a `fea.Temperature` stored in kelvin. Build it with `fea.Temperature.celsius` or `fea.Temperature.kelvin`; a `fea.TemperatureDifference` is rejected.",
    },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "FEA models",
        target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/models"),
    },
    BuiltinDocumentationLink {
        label: "Thermal loads and boundaries",
        target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/loads/thermal"),
    },
    BuiltinDocumentationLink {
        label: "Thermal physics",
        target: BuiltinDocumentationLinkTarget::Documentation("/docs/simulation/physics/thermal"),
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
        title: "fea.ThermalBoundary.convection",
        summary: "Create a convective (Robin) heat-transfer boundary on a face region.",
        description: "`fea.ThermalBoundary.convection(id, region, ambientTemperature, coefficient)` builds a `fea.ThermalBoundary` that applies Newton cooling to the selected faces.",
        keywords: &[
            "fea",
            "boundary",
            "thermal",
            "convection",
            "film coefficient",
            "Robin",
        ],
        related: &[
            "fea.ThermalBoundary.heatFlux",
            "fea.ThermalConstraint.prescribed",
            "fea.ThermalExcitation.ambientTemperature",
            "fea.Temperature.celsius",
            "fea.HeatTransferCoefficient.wattsPerSquareMeterKelvin",
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
