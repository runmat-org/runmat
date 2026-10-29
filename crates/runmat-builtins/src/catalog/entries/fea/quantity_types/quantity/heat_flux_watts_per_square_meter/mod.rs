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
    class: identities::FEA_HEAT_FLUX_CLASS,
    scale: 1.0,
    offset: 0.0,
    range: QuantityRange::Finite,
};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_scalar(request, entry, QUANTITY)
}

const HEAT_FLUX_WATTS_PER_SQUARE_METER_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "value = fea.HeatFlux.wattsPerSquareMeter(value)",
        inputs: &QUANTITY_INPUTS,
        outputs: &QUANTITY_OUTPUTS,
    }];
pub const HEAT_FLUX_WATTS_PER_SQUARE_METER_DESCRIPTOR: BuiltinDescriptor =
    quantity_descriptor(&HEAT_FLUX_WATTS_PER_SQUARE_METER_SIGNATURES);
const HEAT_FLUX_WATTS_PER_SQUARE_METER_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "heat-flux-wattspersquaremeter",
    title: "A 1.5 kW/m^2 heat flux",
    program: "value = fea.HeatFlux.wattsPerSquareMeter(1500);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(value, 'fea.HeatFlux'));\nassert(abs(value.watts_per_square_meter - 1500) <= 1e-9);",
    },
}];
pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.HeatFlux.wattsPerSquareMeter",
    documentation: catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.HeatFlux.wattsPerSquareMeter",
        summary: "Create a typed HeatFlux value from watts per square meter.",
        description: "`value = fea.HeatFlux.wattsPerSquareMeter(x)` returns a `fea.HeatFlux` holding a heat flux, with `x` given in watts per square meter and stored in watts per square meter (W/m^2).",
        keywords: &[
                "fea",
                "quantity",
                "units",
                "HeatFlux",
                "watts per square meter",
            ],
        related: &["fea.ThermalBoundary.heatFlux"],
        sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
                "The argument is a heat flux in watts per square meter. It is stored unchanged, and the result is an immutable `fea.HeatFlux` whose `watts_per_square_meter` property holds the canonical value in watts per square meter (W/m^2).",
                "The input must be a finite real numeric scalar; integer classes are accepted and converted once to double precision. Logical, complex, text, non-scalar, and accelerator-resident values are rejected. Any finite real value is accepted, including negative values.",
            ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
                "Pass the value to constructors that take a heat flux, such as `fea.ThermalBoundary.heatFlux`. Constructors check the value's class, so a quantity of the wrong dimension is rejected rather than silently reinterpreted.",
            ] },
        BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
                "`RunMat:fea:InvalidInput` reports a missing, non-numeric, non-finite, non-scalar, or out-of-range value.",
            ] },
        ],
        examples: HEAT_FLUX_WATTS_PER_SQUARE_METER_EXAMPLES,
        faqs: super::FAQS,
        links: super::LINKS,
        implementation: super::IMPLEMENTATION,
        verification: super::VERIFICATION,
        notes: &[],
    }),
    descriptor: &HEAT_FLUX_WATTS_PER_SQUARE_METER_DESCRIPTOR,
    inference: infer,
    integer_capabilities: &QUANTITY_INTEGER_CAPABILITIES,
    integer_audit: None,
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
