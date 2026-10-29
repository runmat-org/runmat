use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition, ConstructorEntryDefinition},
    identities,
    quantity_entry::{
        infer_scalar, quantity_descriptor, QuantityRange, ScalarQuantity, QUANTITY_INPUTS,
        QUANTITY_INTEGER_CAPABILITIES, QUANTITY_OUTPUTS,
    },
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

const QUANTITY: ScalarQuantity = ScalarQuantity {
    class: identities::FEA_THERMAL_CONTACT_RESISTANCE_CLASS,
    scale: 1.0,
    offset: 0.0,
    range: QuantityRange::NonNegative,
};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_scalar(request, entry, QUANTITY)
}

const THERMAL_CONTACT_RESISTANCE_SQUARE_METER_KELVIN_PER_WATT_SIGNATURES:
    [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "value = fea.ThermalContactResistance.squareMeterKelvinPerWatt(value)",
    inputs: &QUANTITY_INPUTS,
    outputs: &QUANTITY_OUTPUTS,
}];
pub const THERMAL_CONTACT_RESISTANCE_SQUARE_METER_KELVIN_PER_WATT_DESCRIPTOR: BuiltinDescriptor =
    quantity_descriptor(&THERMAL_CONTACT_RESISTANCE_SQUARE_METER_KELVIN_PER_WATT_SIGNATURES);
const THERMAL_CONTACT_RESISTANCE_SQUARE_METER_KELVIN_PER_WATT_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "thermal-contact-resistance-squaremeterkelvinperwatt",
    title: "A thermal-pad contact resistance",
    program: "value = fea.ThermalContactResistance.squareMeterKelvinPerWatt(1e-4);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(value, 'fea.ThermalContactResistance'));\nassert(abs(value.square_meter_kelvin_per_watt - 0.0001) <= 1e-12);",
    },
}];
pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.ThermalContactResistance.squareMeterKelvinPerWatt",
    documentation: catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.ThermalContactResistance.squareMeterKelvinPerWatt",
        summary: "Create a typed ThermalContactResistance value from square meter-kelvin per watt.",
        description: "`value = fea.ThermalContactResistance.squareMeterKelvinPerWatt(x)` returns a `fea.ThermalContactResistance` holding a thermal contact resistance, with `x` given in square meter-kelvin per watt and stored in square meter-kelvin per watt (m^2*K/W).",
        keywords: &[
                "fea",
                "quantity",
                "units",
                "ThermalContactResistance",
                "square meter-kelvin per watt",
            ],
        related: &["fea.ConjugateHeatTransferInterface.resistance"],
        sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
                "The argument is a thermal contact resistance in square meter-kelvin per watt. It is stored unchanged, and the result is an immutable `fea.ThermalContactResistance` whose `square_meter_kelvin_per_watt` property holds the canonical value in square meter-kelvin per watt (m^2*K/W).",
                "The input must be a finite real numeric scalar; integer classes are accepted and converted once to double precision. Logical, complex, text, non-scalar, and accelerator-resident values are rejected. The canonical value cannot be negative; a value outside that range is rejected.",
            ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
                "Pass the value to constructors that take a thermal contact resistance, such as `fea.ConjugateHeatTransferInterface.resistance`. Constructors check the value's class, so a quantity of the wrong dimension is rejected rather than silently reinterpreted.",
            ] },
        BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
                "`RunMat:fea:InvalidInput` reports a missing, non-numeric, non-finite, non-scalar, or out-of-range value.",
            ] },
        ],
        examples: THERMAL_CONTACT_RESISTANCE_SQUARE_METER_KELVIN_PER_WATT_EXAMPLES,
        faqs: super::FAQS,
        links: super::LINKS,
        implementation: super::IMPLEMENTATION,
        verification: super::VERIFICATION,
        notes: &[],
    }),
    descriptor: &THERMAL_CONTACT_RESISTANCE_SQUARE_METER_KELVIN_PER_WATT_DESCRIPTOR,
    inference: infer,
    integer_capabilities: &QUANTITY_INTEGER_CAPABILITIES,
    integer_audit: None,
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
