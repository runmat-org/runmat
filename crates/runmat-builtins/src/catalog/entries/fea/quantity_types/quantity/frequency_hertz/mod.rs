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

const QUANTITY: ScalarQuantity = ScalarQuantity { class: identities::FEA_FREQUENCY_CLASS, scale: 1.0, offset: 0.0, range: QuantityRange::Positive };

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_scalar(request, entry, QUANTITY)
}

const FREQUENCY_HERTZ_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "value = fea.Frequency.hertz(value)",
    inputs: &QUANTITY_INPUTS,
    outputs: &QUANTITY_OUTPUTS,
}];
pub const FREQUENCY_HERTZ_DESCRIPTOR: BuiltinDescriptor = quantity_descriptor(&FREQUENCY_HERTZ_SIGNATURES);
const FREQUENCY_HERTZ_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "frequency-hertz",
    title: "A 120 Hz excitation frequency",
    program: "value = fea.Frequency.hertz(120);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(value, 'fea.Frequency'));\nassert(abs(value.hertz - 120) <= 1e-9);",
    },
}];
pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.Frequency.hertz",
    documentation: catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.Frequency.hertz",
        summary: "Create a typed Frequency value from hertz.",
        description: "`value = fea.Frequency.hertz(x)` returns a `fea.Frequency` holding a frequency, with `x` given in hertz and stored in hertz (Hz).",
        keywords: &[
                "fea",
                "quantity",
                "units",
                "Frequency",
                "hertz",
            ],
        related: &[
                "fea.ElectromagneticDomain.atFrequency",
                "fea.field",
            ],
        sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
                "The argument is a frequency in hertz. It is stored unchanged, and the result is an immutable `fea.Frequency` whose `hertz` property holds the canonical value in hertz (Hz).",
                "The input must be a finite real numeric scalar; integer classes are accepted and converted once to double precision. Logical, complex, text, non-scalar, and accelerator-resident values are rejected. The canonical value must be positive; a value outside that range is rejected.",
            ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
                "Pass the value to constructors that take a frequency, such as `fea.ElectromagneticDomain.atFrequency`, `fea.field`. Constructors check the value's class, so a quantity of the wrong dimension is rejected rather than silently reinterpreted.",
                "It is also the sample argument of `fea.field` for frequency-domain results: `fea.field(results, selector, fea.Frequency.hertz(120))`.",
            ] },
        BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
                "`RunMat:fea:InvalidInput` reports a missing, non-numeric, non-finite, non-scalar, or out-of-range value.",
            ] },
        ],
        examples: FREQUENCY_HERTZ_EXAMPLES,
        faqs: super::FAQS,
        links: super::LINKS,
        implementation: super::IMPLEMENTATION,
        verification: super::VERIFICATION,
        notes: &[],
    }),
    descriptor: &FREQUENCY_HERTZ_DESCRIPTOR,
    inference: infer,
    integer_capabilities: &QUANTITY_INTEGER_CAPABILITIES,
    integer_audit: None,
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
