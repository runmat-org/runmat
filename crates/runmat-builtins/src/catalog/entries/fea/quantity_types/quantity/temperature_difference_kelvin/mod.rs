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
    class: identities::FEA_TEMPERATURE_DIFFERENCE_CLASS,
    scale: 1.0,
    offset: 0.0,
    range: QuantityRange::Finite,
};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_scalar(request, entry, QUANTITY)
}

const TEMPERATURE_DIFFERENCE_KELVIN_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "value = fea.TemperatureDifference.kelvin(value)",
        inputs: &QUANTITY_INPUTS,
        outputs: &QUANTITY_OUTPUTS,
    }];
pub const TEMPERATURE_DIFFERENCE_KELVIN_DESCRIPTOR: BuiltinDescriptor =
    quantity_descriptor(&TEMPERATURE_DIFFERENCE_KELVIN_SIGNATURES);
const TEMPERATURE_DIFFERENCE_KELVIN_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "temperature-difference-kelvin",
    title: "A 5 K temperature jump",
    program: "value = fea.TemperatureDifference.kelvin(5);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(value, 'fea.TemperatureDifference'));\nassert(abs(value.kelvin - 5) <= 1e-12);",
    },
}];
pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.TemperatureDifference.kelvin",
    documentation: catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.TemperatureDifference.kelvin",
        summary: "Create a typed TemperatureDifference value from kelvin.",
        description: "`value = fea.TemperatureDifference.kelvin(x)` returns a `fea.TemperatureDifference` holding a temperature difference (an interval, not an absolute temperature), with `x` given in kelvin and stored in kelvin (K).",
        keywords: &[
                "fea",
                "quantity",
                "units",
                "TemperatureDifference",
                "kelvin",
            ],
        related: &[
                "fea.TemperatureDifference.celsius",
                "fea.ThermalConstraint.affineJump",
            ],
        sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
                "The argument is a temperature difference (an interval, not an absolute temperature) in kelvin. It is stored unchanged, and the result is an immutable `fea.TemperatureDifference` whose `kelvin` property holds the canonical value in kelvin (K).",
                "The input must be a finite real numeric scalar; integer classes are accepted and converted once to double precision. Logical, complex, text, non-scalar, and accelerator-resident values are rejected. Any finite real value is accepted, including negative values.",
                "Use `fea.TemperatureDifference` for intervals such as a prescribed temperature jump; use `fea.Temperature` for absolute temperatures.",
            ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
                "Pass the value to constructors that take a temperature difference (an interval, not an absolute temperature), such as `fea.ThermalConstraint.affineJump`. Constructors check the value's class, so a quantity of the wrong dimension is rejected rather than silently reinterpreted.",
            ] },
        BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
                "`RunMat:fea:InvalidInput` reports a missing, non-numeric, non-finite, non-scalar, or out-of-range value.",
            ] },
        ],
        examples: TEMPERATURE_DIFFERENCE_KELVIN_EXAMPLES,
        faqs: super::FAQS,
        links: super::LINKS,
        implementation: super::IMPLEMENTATION,
        verification: super::VERIFICATION,
        notes: &[],
    }),
    descriptor: &TEMPERATURE_DIFFERENCE_KELVIN_DESCRIPTOR,
    inference: infer,
    integer_capabilities: &QUANTITY_INTEGER_CAPABILITIES,
    integer_audit: None,
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
