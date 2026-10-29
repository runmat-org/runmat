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
    offset: 0.0,
    range: QuantityRange::NonNegative,
};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_scalar(request, entry, QUANTITY)
}

const TEMPERATURE_KELVIN_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "value = fea.Temperature.kelvin(value)",
        inputs: &QUANTITY_INPUTS,
        outputs: &QUANTITY_OUTPUTS,
    }];
pub const TEMPERATURE_KELVIN_DESCRIPTOR: BuiltinDescriptor =
    quantity_descriptor(&TEMPERATURE_KELVIN_SIGNATURES);
const TEMPERATURE_KELVIN_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "temperature-kelvin",
    title: "Room temperature in kelvin",
    program: "value = fea.Temperature.kelvin(293.15);",
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
    name: "fea.Temperature.kelvin",
    documentation: catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.Temperature.kelvin",
        summary: "Create a typed Temperature value from kelvin.",
        description: "`value = fea.Temperature.kelvin(x)` returns a `fea.Temperature` holding an absolute temperature, with `x` given in kelvin and stored in kelvin (K).",
        keywords: &[
                "fea",
                "quantity",
                "units",
                "Temperature",
                "kelvin",
            ],
        related: &[
                "fea.Temperature.celsius",
                "fea.ThermalConstraint.prescribed",
                "fea.ThermalBoundary.convection",
                "fea.ThermalDomain.atReferenceTemperature",
            ],
        sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
                "The argument is an absolute temperature in kelvin. It is stored unchanged, and the result is an immutable `fea.Temperature` whose `kelvin` property holds the canonical value in kelvin (K).",
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
        examples: TEMPERATURE_KELVIN_EXAMPLES,
        faqs: super::FAQS,
        links: super::LINKS,
        implementation: super::IMPLEMENTATION,
        verification: super::VERIFICATION,
        notes: &[],
    }),
    descriptor: &TEMPERATURE_KELVIN_DESCRIPTOR,
    inference: infer,
    integer_capabilities: &QUANTITY_INTEGER_CAPABILITIES,
    integer_audit: None,
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
