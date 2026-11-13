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
    class: identities::FEA_TIME_CLASS,
    scale: 1.0,
    offset: 0.0,
    range: QuantityRange::Finite,
};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_scalar(request, entry, QUANTITY)
}

const TIME_SECONDS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "value = fea.Time.seconds(value)",
    inputs: &QUANTITY_INPUTS,
    outputs: &QUANTITY_OUTPUTS,
}];
pub const TIME_SECONDS_DESCRIPTOR: BuiltinDescriptor =
    quantity_descriptor(&TIME_SECONDS_SIGNATURES);
const TIME_SECONDS_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "time-seconds",
    title: "The 12.2 ms state of a transient run",
    program: "value = fea.Time.seconds(0.0122);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(value, 'fea.Time'));\nassert(abs(value.seconds - 0.0122) <= 1e-15);",
    },
}];
pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.Time.seconds",
    documentation: catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.Time.seconds",
        summary: "Create a typed Time value from seconds.",
        description: "`value = fea.Time.seconds(t)` returns a `fea.Time` holding a time instant, with `t` given in seconds and stored in seconds (s).",
        keywords: &[
                "fea",
                "quantity",
                "units",
                "Time",
                "seconds",
                "transient",
            ],
        related: &[
                "fea.field",
                "fea.fieldPage",
            ],
        sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
                "The argument is a time in seconds. It is stored unchanged, and the result is an immutable `fea.Time` whose `seconds` property holds the canonical value in seconds (s).",
                "The input must be a finite real numeric scalar; integer classes are accepted and converted once to double precision. Logical, complex, text, non-scalar, and accelerator-resident values are rejected.",
            ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
                "It is the time sample argument of `fea.field` for transient results: `fea.field(results, fea.StructuralField.displacement(), fea.Time.seconds(0.0122))` selects the retained state at that time, and `fea.fieldPage` then reads its values in bounded pages.",
            ] },
        BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
                "`RunMat:fea:InvalidInput` reports a missing, non-numeric, non-finite, or non-scalar value.",
            ] },
        ],
        examples: TIME_SECONDS_EXAMPLES,
        faqs: super::FAQS,
        links: super::LINKS,
        implementation: super::IMPLEMENTATION,
        verification: super::VERIFICATION,
        notes: &[],
    }),
    descriptor: &TIME_SECONDS_DESCRIPTOR,
    inference: infer,
    integer_capabilities: &QUANTITY_INTEGER_CAPABILITIES,
    integer_audit: None,
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
