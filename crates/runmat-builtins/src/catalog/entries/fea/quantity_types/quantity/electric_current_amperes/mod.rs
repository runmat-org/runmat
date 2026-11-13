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
    class: identities::FEA_ELECTRIC_CURRENT_CLASS,
    scale: 1.0,
    offset: 0.0,
    range: QuantityRange::Finite,
};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_scalar(request, entry, QUANTITY)
}

const ELECTRIC_CURRENT_AMPERES_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "value = fea.ElectricCurrent.amperes(value)",
        inputs: &QUANTITY_INPUTS,
        outputs: &QUANTITY_OUTPUTS,
    }];
pub const ELECTRIC_CURRENT_AMPERES_DESCRIPTOR: BuiltinDescriptor =
    quantity_descriptor(&ELECTRIC_CURRENT_AMPERES_SIGNATURES);
const ELECTRIC_CURRENT_AMPERES_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "electric-current-amperes",
    title: "A 3 A terminal current",
    program: "value = fea.ElectricCurrent.amperes(3);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source:
            "assert(isa(value, 'fea.ElectricCurrent'));\nassert(abs(value.amperes - 3) <= 1e-12);",
    },
}];
pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.ElectricCurrent.amperes",
    documentation: catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.ElectricCurrent.amperes",
        summary: "Create a typed ElectricCurrent value from amperes.",
        description: "`value = fea.ElectricCurrent.amperes(x)` returns a `fea.ElectricCurrent` holding an electric current, with `x` given in amperes and stored in amperes (A).",
        keywords: &[
                "fea",
                "quantity",
                "units",
                "ElectricCurrent",
                "amperes",
            ],
        related: &["fea.ElectricalConstraint.integralCurrent"],
        sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
                "The argument is an electric current in amperes. It is stored unchanged, and the result is an immutable `fea.ElectricCurrent` whose `amperes` property holds the canonical value in amperes (A).",
                "The input must be a finite real numeric scalar; integer classes are accepted and converted once to double precision. Logical, complex, text, non-scalar, and accelerator-resident values are rejected. Any finite real value is accepted, including negative values.",
            ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
                "Pass the value to constructors that take an electric current, such as `fea.ElectricalConstraint.integralCurrent`. Constructors check the value's class, so a quantity of the wrong dimension is rejected rather than silently reinterpreted.",
            ] },
        BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
                "`RunMat:fea:InvalidInput` reports a missing, non-numeric, non-finite, non-scalar, or out-of-range value.",
            ] },
        ],
        examples: ELECTRIC_CURRENT_AMPERES_EXAMPLES,
        faqs: super::FAQS,
        links: super::LINKS,
        implementation: super::IMPLEMENTATION,
        verification: super::VERIFICATION,
        notes: &[],
    }),
    descriptor: &ELECTRIC_CURRENT_AMPERES_DESCRIPTOR,
    inference: infer,
    integer_capabilities: &QUANTITY_INTEGER_CAPABILITIES,
    integer_audit: None,
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
