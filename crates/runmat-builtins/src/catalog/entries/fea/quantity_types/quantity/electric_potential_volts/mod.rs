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
    class: identities::FEA_ELECTRIC_POTENTIAL_CLASS,
    scale: 1.0,
    offset: 0.0,
    range: QuantityRange::Finite,
};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_scalar(request, entry, QUANTITY)
}

const ELECTRIC_POTENTIAL_VOLTS_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "value = fea.ElectricPotential.volts(value)",
        inputs: &QUANTITY_INPUTS,
        outputs: &QUANTITY_OUTPUTS,
    }];
pub const ELECTRIC_POTENTIAL_VOLTS_DESCRIPTOR: BuiltinDescriptor =
    quantity_descriptor(&ELECTRIC_POTENTIAL_VOLTS_SIGNATURES);
const ELECTRIC_POTENTIAL_VOLTS_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "electric-potential-volts",
    title: "A 12 V terminal",
    program: "value = fea.ElectricPotential.volts(12);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source:
            "assert(isa(value, 'fea.ElectricPotential'));\nassert(abs(value.volts - 12) <= 1e-9);",
    },
}];
pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.ElectricPotential.volts",
    documentation: catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.ElectricPotential.volts",
        summary: "Create a typed ElectricPotential value from volts.",
        description: "`value = fea.ElectricPotential.volts(x)` returns a `fea.ElectricPotential` holding an electric potential, with `x` given in volts and stored in volts (V).",
        keywords: &[
                "fea",
                "quantity",
                "units",
                "ElectricPotential",
                "volts",
            ],
        related: &["fea.ElectricalConstraint.prescribedPotential"],
        sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
                "The argument is an electric potential in volts. It is stored unchanged, and the result is an immutable `fea.ElectricPotential` whose `volts` property holds the canonical value in volts (V).",
                "The input must be a finite real numeric scalar; integer classes are accepted and converted once to double precision. Logical, complex, text, non-scalar, and accelerator-resident values are rejected. Any finite real value is accepted, including negative values.",
            ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
                "Pass the value to constructors that take an electric potential, such as `fea.ElectricalConstraint.prescribedPotential`. Constructors check the value's class, so a quantity of the wrong dimension is rejected rather than silently reinterpreted.",
            ] },
        BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
                "`RunMat:fea:InvalidInput` reports a missing, non-numeric, non-finite, non-scalar, or out-of-range value.",
            ] },
        ],
        examples: ELECTRIC_POTENTIAL_VOLTS_EXAMPLES,
        faqs: super::FAQS,
        links: super::LINKS,
        implementation: super::IMPLEMENTATION,
        verification: super::VERIFICATION,
        notes: &[],
    }),
    descriptor: &ELECTRIC_POTENTIAL_VOLTS_DESCRIPTOR,
    inference: infer,
    integer_capabilities: &QUANTITY_INTEGER_CAPABILITIES,
    integer_audit: None,
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
