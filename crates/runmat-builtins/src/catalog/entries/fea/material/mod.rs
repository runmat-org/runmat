use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition, ConstructorEntryDefinition},
    constructor_inference::{infer_constructor, FeaConstructorKind},
    descriptors::FEA_MATERIAL_DESCRIPTOR,
    doc_links,
    integer::FEA_MATERIAL_INTEGER_CAPABILITIES,
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_constructor(request, entry, FeaConstructorKind::Material)
}

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "structural-steel",
        title: "Define structural steel",
        program: "steel = fea.material(\"steel\", \"Name\", \"Structural steel\", ...\n    \"YoungsModulusPa\", 200e9, \"PoissonRatio\", 0.3, \"MassDensityKgPerM3\", 7850);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(steel, 'fea.Material'));\nassert(strcmp(steel.material_id, \"steel\"));\nassert(strcmp(steel.name, \"Structural steel\"));\nassert(steel.mass_density_kg_per_m3 == 7850);",
        },
    },
    BuiltinExample {
        id: "thermal-aluminum",
        title: "Add thermal properties for a coupled analysis",
        program: "aluminum = fea.material(\"al6061\", \"YoungsModulusPa\", 68.9e9, \"PoissonRatio\", 0.33, ...\n    \"MassDensityKgPerM3\", 2700, \"ReferenceTemperatureK\", 293.15, ...\n    \"ModulusTempCoeffPerK\", -4e-4, \"ConductivityWPerMK\", 167, ...\n    \"SpecificHeatJPerKgK\", 896, \"ExpansionCoefficientPerK\", 23.6e-6);",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isa(aluminum, 'fea.Material'));\nassert(aluminum.thermal.conductivity_w_per_mk == 167);",
        },
    },
];

const DOCUMENTATION: BuiltinDocumentation = catalog_entry::documentation(ConstructorDocumentationDefinition {
    title: "fea.material",
    summary: "Define a material's mechanical, thermal, acoustic, electrical, and plastic properties in SI units.",
    description: "`material = fea.material(id, Name, Value, ...)` creates a `fea.Material`. Each name/value pair supplies one SI property, such as `\"YoungsModulusPa\", 200e9`, or a whole capability block such as `\"Thermal\", thermalStruct`. At least one physical capability is required.",
    keywords: &["fea", "material", "elastic", "Young's modulus", "density", "thermal", "conductivity", "damping", "plasticity"],
    related: &["fea.materialAssignment", "fea.model", "fea.MechanicalDamping.rayleigh", "fea.study"],
    sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
            "A material is a set of typed capabilities. Each analysis uses the capabilities it needs: static elasticity needs stiffness, dynamics also needs mass density, heat transfer needs thermal properties, and so on. The returned `fea.Material` is an immutable value identified by `id`; `Name` sets a display name and defaults to the id.",
            "Property names are matched case-insensitively; CamelCase and snake_case spellings are equivalent (`YoungsModulusPa` and `youngs_modulus_pa`). Every value is SI and the unit is part of the name: `MassDensityKgPerM3` (kg/m^3), `YoungsModulusPa` (Pa) with `PoissonRatio`, and `Damping` (a `fea.MechanicalDamping` object such as `fea.MechanicalDamping.rayleigh`).",
            "Thermal capability: `ReferenceTemperatureK` (K), `ModulusTempCoeffPerK` (1/K), `ConductivityWPerMK` (W/(m*K)), `SpecificHeatJPerKgK` (J/(kg*K)), and `ExpansionCoefficientPerK` (1/K). Supplying any one of them requires all five. Elastic capability likewise requires both `YoungsModulusPa` and `PoissonRatio`, and acoustic capability requires both `SpeedOfSoundMPerS` (m/s) and `DampingRatio`.",
            "Electrical capability takes `ConductivitySPerM` (S/m), `ResistiveHeatingCoefficient`, `RelativePermittivity`, `RelativePermeability`, `ConductivityFrequencyResponse`, and an optional `ReferenceTemperatureK`. Whole capability blocks can be passed as structs with snake_case fields through `Mechanical`, `Thermal`, `Acoustic`, `Electrical`, and `Plastic`; plastic behavior is available only as a block.",
            "Construction checks names, required companions, and value types, and fails if no capability is given or a name is unknown. Physical admissibility of the values is checked when a model that uses the material is validated in `fea.study`.",
        ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
            "List the material in the `Materials` option of `fea.model`, and give it to a region with `fea.materialAssignment(target, material)` in `MaterialAssignments`. One material object can be assigned to several regions and reused across models and studies.",
        ] },
        BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
            "`RunMat:fea:InvalidInput` reports a non-text id, an incomplete Name/Value pair, a repeated or unknown property name, a missing companion property, a non-finite or non-numeric value, a resident (device) value, or a material with no capability at all.",
        ] },
    ],
    examples: EXAMPLES,
    faqs: &[
        BuiltinDocumentationFaq { question: "What units does fea.material use?", answer: "SI only, with the unit spelled in each property name: pascals for modulus, kilograms per cubic meter for density, W/(m*K) for conductivity, and so on. Convert values before passing them." },
        BuiltinDocumentationFaq { question: "Can I pass integer values?", answer: "Yes. Integer scalars of any class are converted once to double precision. Very large 64-bit integers can round in that conversion." },
        BuiltinDocumentationFaq { question: "Why was DensityKgPerM3 rejected?", answer: "The density property is named `MassDensityKgPerM3`. The shorter spelling is not accepted, so that the property cannot be mistaken for another density." },
    ],
    links: &[doc_links::MODELS, doc_links::TYPES_AND_UNITS],
    implementation: &[doc_links::FEA_MODELING_SOURCE],
    verification: &[
        doc_links::PORTABLE_SCHEMA_TEST,
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Materials require explicit capabilities and the canonical density name", location: "crates/runmat-runtime/src/builtins/fea/mod.rs::fea_material_requires_explicit_capabilities_and_canonical_density_name" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Materials preserve typed Rayleigh damping", location: "crates/runmat-runtime/src/builtins/fea/mod.rs::fea_material_preserves_typed_rayleigh_damping" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Numeric constructors convert every integer class once to binary64", location: "crates/runmat-runtime/src/builtins/fea/mod.rs::fea_numeric_constructors_cross_all_integer_classes_once_into_binary64" },
    ],
    notes: &[],
});

pub const FEA_MATERIAL_CATALOG_ENTRY: BuiltinCatalogEntry =
    catalog_entry::entry(ConstructorEntryDefinition {
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "fea.material",
        documentation: DOCUMENTATION,
        descriptor: &FEA_MATERIAL_DESCRIPTOR,
        inference: infer,
        integer_capabilities: &FEA_MATERIAL_INTEGER_CAPABILITIES,
        integer_audit: None,
        filesystem_read: false,
        distributed: BuiltinDistributedPolicy::Unsupported,
    });

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&FEA_MATERIAL_CATALOG_ENTRY];
