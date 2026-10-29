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
    class: identities::FEA_HEAT_TRANSFER_COEFFICIENT_CLASS,
    scale: 1.0,
    offset: 0.0,
    range: QuantityRange::NonNegative,
};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_scalar(request, entry, QUANTITY)
}

const HEAT_TRANSFER_COEFFICIENT_WATTS_PER_SQUARE_METER_KELVIN_SIGNATURES:
    [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "value = fea.HeatTransferCoefficient.wattsPerSquareMeterKelvin(value)",
    inputs: &QUANTITY_INPUTS,
    outputs: &QUANTITY_OUTPUTS,
}];
pub const HEAT_TRANSFER_COEFFICIENT_WATTS_PER_SQUARE_METER_KELVIN_DESCRIPTOR: BuiltinDescriptor =
    quantity_descriptor(&HEAT_TRANSFER_COEFFICIENT_WATTS_PER_SQUARE_METER_KELVIN_SIGNATURES);
const HEAT_TRANSFER_COEFFICIENT_WATTS_PER_SQUARE_METER_KELVIN_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "heat-transfer-coefficient-wattspersquaremeterkelvin",
    title: "Natural convection in air",
    program: "value = fea.HeatTransferCoefficient.wattsPerSquareMeterKelvin(25);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(value, 'fea.HeatTransferCoefficient'));\nassert(abs(value.watts_per_square_meter_kelvin - 25) <= 1e-9);",
    },
}];
pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.HeatTransferCoefficient.wattsPerSquareMeterKelvin",
    documentation: catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.HeatTransferCoefficient.wattsPerSquareMeterKelvin",
        summary: "Create a typed HeatTransferCoefficient value from watts per square meter-kelvin.",
        description: "`value = fea.HeatTransferCoefficient.wattsPerSquareMeterKelvin(x)` returns a `fea.HeatTransferCoefficient` holding a heat-transfer coefficient, with `x` given in watts per square meter-kelvin and stored in watts per square meter-kelvin (W/(m^2*K)).",
        keywords: &[
                "fea",
                "quantity",
                "units",
                "HeatTransferCoefficient",
                "watts per square meter-kelvin",
            ],
        related: &[
                "fea.ThermalBoundary.convection",
                "fea.ConjugateHeatTransferInterface.conductance",
            ],
        sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
                "The argument is a heat-transfer coefficient in watts per square meter-kelvin. It is stored unchanged, and the result is an immutable `fea.HeatTransferCoefficient` whose `watts_per_square_meter_kelvin` property holds the canonical value in watts per square meter-kelvin (W/(m^2*K)).",
                "The input must be a finite real numeric scalar; integer classes are accepted and converted once to double precision. Logical, complex, text, non-scalar, and accelerator-resident values are rejected. The canonical value cannot be negative; a value outside that range is rejected.",
            ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
                "Pass the value to constructors that take a heat-transfer coefficient, such as `fea.ThermalBoundary.convection`, `fea.ConjugateHeatTransferInterface.conductance`. Constructors check the value's class, so a quantity of the wrong dimension is rejected rather than silently reinterpreted.",
            ] },
        BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
                "`RunMat:fea:InvalidInput` reports a missing, non-numeric, non-finite, non-scalar, or out-of-range value.",
            ] },
        ],
        examples: HEAT_TRANSFER_COEFFICIENT_WATTS_PER_SQUARE_METER_KELVIN_EXAMPLES,
        faqs: super::FAQS,
        links: super::LINKS,
        implementation: super::IMPLEMENTATION,
        verification: super::VERIFICATION,
        notes: &[],
    }),
    descriptor: &HEAT_TRANSFER_COEFFICIENT_WATTS_PER_SQUARE_METER_KELVIN_DESCRIPTOR,
    inference: infer,
    integer_capabilities: &QUANTITY_INTEGER_CAPABILITIES,
    integer_audit: None,
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
