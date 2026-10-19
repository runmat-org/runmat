use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition, ConstructorEntryDefinition},
    quantity_entry::{
        infer_scalar, quantity_descriptor, QuantityRange, ScalarQuantity, QUANTITY_INPUTS,
        QUANTITY_INTEGER_CAPABILITIES, QUANTITY_OUTPUTS,
    },
};
use crate::*;
use runmat_types::{CallInference, CallRequest, StaticClassIdentity};

const QUANTITY: ScalarQuantity = ScalarQuantity { class: StaticClassIdentity::new("fea.MassDensity"), scale: 1.0, offset: 0.0, range: QuantityRange::Positive };

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_scalar(request, entry, QUANTITY)
}

const MASS_DENSITY_KILOGRAMS_PER_CUBIC_METER_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "value = fea.MassDensity.kilogramsPerCubicMeter(value)",
    inputs: &QUANTITY_INPUTS,
    outputs: &QUANTITY_OUTPUTS,
}];
pub const MASS_DENSITY_KILOGRAMS_PER_CUBIC_METER_DESCRIPTOR: BuiltinDescriptor = quantity_descriptor(&MASS_DENSITY_KILOGRAMS_PER_CUBIC_METER_SIGNATURES);
const MASS_DENSITY_KILOGRAMS_PER_CUBIC_METER_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "mass-density-kilogramspercubicmeter",
    title: "Water at 20 degrees Celsius",
    program: "value = fea.MassDensity.kilogramsPerCubicMeter(998);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(value, 'fea.MassDensity'));\nassert(abs(value.kilograms_per_cubic_meter - 998) <= 1e-9);",
    },
}];
pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.MassDensity.kilogramsPerCubicMeter",
    documentation: catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.MassDensity.kilogramsPerCubicMeter",
        summary: "Create a typed MassDensity value from kilograms per cubic meter.",
        description: "`value = fea.MassDensity.kilogramsPerCubicMeter(x)` returns a `fea.MassDensity` holding a mass density, with `x` given in kilograms per cubic meter and stored in kilograms per cubic meter (kg/m^3).",
        keywords: &[
                "fea",
                "quantity",
                "units",
                "MassDensity",
                "kilograms per cubic meter",
            ],
        related: &["fea.FluidDomain.newtonian"],
        sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
                "The argument is a mass density in kilograms per cubic meter. It is stored unchanged, and the result is an immutable `fea.MassDensity` whose `kilograms_per_cubic_meter` property holds the canonical value in kilograms per cubic meter (kg/m^3).",
                "The input must be a finite real numeric scalar; integer classes are accepted and converted once to double precision. Logical, complex, text, non-scalar, and accelerator-resident values are rejected. The canonical value must be positive; a value outside that range is rejected.",
            ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
                "Pass the value to constructors that take a mass density, such as `fea.FluidDomain.newtonian`. Constructors check the value's class, so a quantity of the wrong dimension is rejected rather than silently reinterpreted.",
            ] },
        BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
                "`RunMat:fea:InvalidInput` reports a missing, non-numeric, non-finite, non-scalar, or out-of-range value.",
            ] },
        ],
        examples: MASS_DENSITY_KILOGRAMS_PER_CUBIC_METER_EXAMPLES,
        faqs: super::FAQS,
        links: super::LINKS,
        implementation: super::IMPLEMENTATION,
        verification: super::VERIFICATION,
        notes: &[],
    }),
    descriptor: &MASS_DENSITY_KILOGRAMS_PER_CUBIC_METER_DESCRIPTOR,
    inference: infer,
    integer_capabilities: &QUANTITY_INTEGER_CAPABILITIES,
    integer_audit: None,
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
