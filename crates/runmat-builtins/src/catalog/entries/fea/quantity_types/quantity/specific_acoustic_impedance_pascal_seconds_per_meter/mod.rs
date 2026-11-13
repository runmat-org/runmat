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
    class: identities::FEA_SPECIFIC_ACOUSTIC_IMPEDANCE_CLASS,
    scale: 1.0,
    offset: 0.0,
    range: QuantityRange::Positive,
};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_scalar(request, entry, QUANTITY)
}

const SPECIFIC_ACOUSTIC_IMPEDANCE_PASCAL_SECONDS_PER_METER_SIGNATURES:
    [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "value = fea.SpecificAcousticImpedance.pascalSecondsPerMeter(value)",
    inputs: &QUANTITY_INPUTS,
    outputs: &QUANTITY_OUTPUTS,
}];
pub const SPECIFIC_ACOUSTIC_IMPEDANCE_PASCAL_SECONDS_PER_METER_DESCRIPTOR: BuiltinDescriptor =
    quantity_descriptor(&SPECIFIC_ACOUSTIC_IMPEDANCE_PASCAL_SECONDS_PER_METER_SIGNATURES);
const SPECIFIC_ACOUSTIC_IMPEDANCE_PASCAL_SECONDS_PER_METER_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "specific-acoustic-impedance-pascalsecondspermeter",
    title: "The characteristic impedance of air",
    program: "value = fea.SpecificAcousticImpedance.pascalSecondsPerMeter(413);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(value, 'fea.SpecificAcousticImpedance'));\nassert(abs(value.pascal_seconds_per_meter - 413) <= 1e-9);",
    },
}];
pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.SpecificAcousticImpedance.pascalSecondsPerMeter",
    documentation: catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.SpecificAcousticImpedance.pascalSecondsPerMeter",
        summary: "Create a typed SpecificAcousticImpedance value from pascal-seconds per meter.",
        description: "`value = fea.SpecificAcousticImpedance.pascalSecondsPerMeter(x)` returns a `fea.SpecificAcousticImpedance` holding a specific acoustic impedance, with `x` given in pascal-seconds per meter and stored in pascal-seconds per meter (Pa*s/m).",
        keywords: &[
                "fea",
                "quantity",
                "units",
                "SpecificAcousticImpedance",
                "pascal-seconds per meter",
            ],
        related: &["fea.AcousticBoundary.impedance"],
        sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
                "The argument is a specific acoustic impedance in pascal-seconds per meter. It is stored unchanged, and the result is an immutable `fea.SpecificAcousticImpedance` whose `pascal_seconds_per_meter` property holds the canonical value in pascal-seconds per meter (Pa*s/m).",
                "The input must be a finite real numeric scalar; integer classes are accepted and converted once to double precision. Logical, complex, text, non-scalar, and accelerator-resident values are rejected. The canonical value must be positive; a value outside that range is rejected.",
            ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
                "Pass the value to constructors that take a specific acoustic impedance, such as `fea.AcousticBoundary.impedance`. Constructors check the value's class, so a quantity of the wrong dimension is rejected rather than silently reinterpreted.",
            ] },
        BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
                "`RunMat:fea:InvalidInput` reports a missing, non-numeric, non-finite, non-scalar, or out-of-range value.",
            ] },
        ],
        examples: SPECIFIC_ACOUSTIC_IMPEDANCE_PASCAL_SECONDS_PER_METER_EXAMPLES,
        faqs: super::FAQS,
        links: super::LINKS,
        implementation: super::IMPLEMENTATION,
        verification: super::VERIFICATION,
        notes: &[],
    }),
    descriptor: &SPECIFIC_ACOUSTIC_IMPEDANCE_PASCAL_SECONDS_PER_METER_DESCRIPTOR,
    inference: infer,
    integer_capabilities: &QUANTITY_INTEGER_CAPABILITIES,
    integer_audit: None,
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
