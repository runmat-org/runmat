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
    class: identities::FEA_PRESSURE_CLASS,
    scale: 1.0,
    offset: 0.0,
    range: QuantityRange::Finite,
};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_scalar(request, entry, QUANTITY)
}

const PRESSURE_PASCALS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "value = fea.Pressure.pascals(value)",
    inputs: &QUANTITY_INPUTS,
    outputs: &QUANTITY_OUTPUTS,
}];
pub const PRESSURE_PASCALS_DESCRIPTOR: BuiltinDescriptor =
    quantity_descriptor(&PRESSURE_PASCALS_SIGNATURES);
const PRESSURE_PASCALS_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "pressure-pascals",
    title: "Standard atmospheric pressure",
    program: "value = fea.Pressure.pascals(101325);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(value, 'fea.Pressure'));\nassert(abs(value.pascals - 101325) <= 1e-9);",
    },
}];
pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.Pressure.pascals",
    documentation: catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.Pressure.pascals",
        summary: "Create a typed Pressure value from pascals.",
        description: "`value = fea.Pressure.pascals(x)` returns a `fea.Pressure` holding a pressure, with `x` given in pascals and stored in pascals (Pa).",
        keywords: &[
                "fea",
                "quantity",
                "units",
                "Pressure",
                "pascals",
            ],
        related: &[
                "fea.StructuralBoundary.pressure",
                "fea.AcousticConstraint.prescribedPressure",
                "fea.FluidBoundary.outletPressure",
            ],
        sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
                "The argument is a pressure in pascals. It is stored unchanged, and the result is an immutable `fea.Pressure` whose `pascals` property holds the canonical value in pascals (Pa).",
                "The input must be a finite real numeric scalar; integer classes are accepted and converted once to double precision. Logical, complex, text, non-scalar, and accelerator-resident values are rejected. Any finite real value is accepted, including negative values.",
            ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
                "Pass the value to constructors that take a pressure, such as `fea.StructuralBoundary.pressure`, `fea.AcousticConstraint.prescribedPressure`, `fea.FluidBoundary.outletPressure`, `fea.FluidConstraint.pressureReference`. Constructors check the value's class, so a quantity of the wrong dimension is rejected rather than silently reinterpreted.",
            ] },
        BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
                "`RunMat:fea:InvalidInput` reports a missing, non-numeric, non-finite, non-scalar, or out-of-range value.",
            ] },
        ],
        examples: PRESSURE_PASCALS_EXAMPLES,
        faqs: super::FAQS,
        links: super::LINKS,
        implementation: super::IMPLEMENTATION,
        verification: super::VERIFICATION,
        notes: &[],
    }),
    descriptor: &PRESSURE_PASCALS_DESCRIPTOR,
    inference: infer,
    integer_capabilities: &QUANTITY_INTEGER_CAPABILITIES,
    integer_audit: None,
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
