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
    class: identities::FEA_TEMPERATURE_CLASS,
    scale: 1.0,
    offset: 273.15,
    range: QuantityRange::NonNegative,
};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_scalar(request, entry, QUANTITY)
}

const TEMPERATURE_CELSIUS_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "value = fea.Temperature.celsius(value)",
        inputs: &QUANTITY_INPUTS,
        outputs: &QUANTITY_OUTPUTS,
    }];
pub const TEMPERATURE_CELSIUS_DESCRIPTOR: BuiltinDescriptor =
    quantity_descriptor(&TEMPERATURE_CELSIUS_SIGNATURES);
const TEMPERATURE_CELSIUS_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "temperature-celsius",
    title: "Room temperature in degrees Celsius",
    program: "value = fea.Temperature.celsius(20);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source:
            "assert(isa(value, 'fea.Temperature'));\nassert(abs(value.kelvin - 293.15) <= 1e-9);",
    },
}];
pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.Temperature.celsius",
    documentation: catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.Temperature.celsius",
        summary: "Create a typed Temperature value from degrees Celsius.",
        description: "`value = fea.Temperature.celsius(x)` returns a `fea.Temperature` holding an absolute temperature, with `x` given in degrees Celsius and stored in kelvin (K).",
        keywords: &[
                "fea",
                "quantity",
                "units",
                "Temperature",
                "degrees Celsius",
            ],
        related: &[
                "fea.Temperature.kelvin",
                "fea.ThermalConstraint.prescribed",
                "fea.ThermalBoundary.convection",
                "fea.ThermalDomain.atReferenceTemperature",
            ],
        sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
                "The argument is an absolute temperature in degrees Celsius. It is converted to kelvin (K) by adding 273.15, and the result is an immutable `fea.Temperature` whose `kelvin` property holds the canonical value in kelvin (K).",
                "The input must be a finite real numeric scalar; integer classes are accepted and converted once to double precision. Logical, complex, text, non-scalar, and accelerator-resident values are rejected. The canonical value cannot be below absolute zero (0 K); a value outside that range is rejected.",
                "`fea.Temperature` is an absolute temperature. For a temperature change or jump, use `fea.TemperatureDifference` so the two are never confused.",
            ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
                "Pass the value to constructors that take an absolute temperature, such as `fea.ThermalConstraint.prescribed`, `fea.ThermalBoundary.convection`, `fea.ThermalDomain.atReferenceTemperature`, `fea.ThermoMechanicalDomain.atReferenceTemperature`. Constructors check the value's class, so a quantity of the wrong dimension is rejected rather than silently reinterpreted.",
            ] },
        BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
                "`RunMat:fea:InvalidInput` reports a missing, non-numeric, non-finite, non-scalar, or out-of-range value.",
            ] },
        ],
        examples: TEMPERATURE_CELSIUS_EXAMPLES,
        faqs: super::FAQS,
        links: super::LINKS,
        implementation: super::IMPLEMENTATION,
        verification: super::VERIFICATION,
        notes: &[],
    }),
    descriptor: &TEMPERATURE_CELSIUS_DESCRIPTOR,
    inference: infer,
    integer_capabilities: &QUANTITY_INTEGER_CAPABILITIES,
    integer_audit: None,
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
