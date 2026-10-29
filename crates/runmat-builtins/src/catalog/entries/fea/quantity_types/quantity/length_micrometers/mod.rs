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
    class: identities::FEA_LENGTH_CLASS,
    scale: 1.0e-6,
    offset: 0.0,
    range: QuantityRange::NonNegative,
};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_scalar(request, entry, QUANTITY)
}

const LENGTH_MICROMETERS_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "value = fea.Length.micrometers(value)",
        inputs: &QUANTITY_INPUTS,
        outputs: &QUANTITY_OUTPUTS,
    }];
pub const LENGTH_MICROMETERS_DESCRIPTOR: BuiltinDescriptor =
    quantity_descriptor(&LENGTH_MICROMETERS_SIGNATURES);
const LENGTH_MICROMETERS_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "length-micrometers",
    title: "A 10 micrometer tolerance",
    program: "value = fea.Length.micrometers(10);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(value, 'fea.Length'));\nassert(abs(value.meters - 1e-05) <= 1e-12);",
    },
}];
pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.Length.micrometers",
    documentation: catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.Length.micrometers",
        summary: "Create a typed Length value from micrometers.",
        description: "`value = fea.Length.micrometers(x)` returns a `fea.Length` holding a length or distance tolerance, with `x` given in micrometers and stored in meters (m).",
        keywords: &[
                "fea",
                "quantity",
                "units",
                "Length",
                "micrometers",
            ],
        related: &[
                "fea.Length.meters",
                "fea.Length.millimeters",
                "fea.PeriodicPairing.geometric",
            ],
        sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
                "The argument is a length or distance tolerance in micrometers. It is multiplied by 1e-06 to give meters (m), and the result is an immutable `fea.Length` whose `meters` property holds the canonical value in meters (m).",
                "The input must be a finite real numeric scalar; integer classes are accepted and converted once to double precision. Logical, complex, text, non-scalar, and accelerator-resident values are rejected. The canonical value cannot be negative; a value outside that range is rejected.",
            ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
                "Pass the value to constructors that take a length or distance tolerance, such as `fea.PeriodicPairing.geometric`. Constructors check the value's class, so a quantity of the wrong dimension is rejected rather than silently reinterpreted.",
            ] },
        BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
                "`RunMat:fea:InvalidInput` reports a missing, non-numeric, non-finite, non-scalar, or out-of-range value.",
            ] },
        ],
        examples: LENGTH_MICROMETERS_EXAMPLES,
        faqs: super::FAQS,
        links: super::LINKS,
        implementation: super::IMPLEMENTATION,
        verification: super::VERIFICATION,
        notes: &[],
    }),
    descriptor: &LENGTH_MICROMETERS_DESCRIPTOR,
    inference: infer,
    integer_capabilities: &QUANTITY_INTEGER_CAPABILITIES,
    integer_audit: None,
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
