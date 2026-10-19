use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition, ConstructorEntryDefinition},
    quantity_entry::{
        infer_scalar, quantity_descriptor, QuantityRange, ScalarQuantity, QUANTITY_INPUTS,
        QUANTITY_INTEGER_CAPABILITIES, QUANTITY_OUTPUTS,
    },
};
use crate::*;
use runmat_types::{CallInference, CallRequest, StaticClassIdentity};

const QUANTITY: ScalarQuantity = ScalarQuantity { class: StaticClassIdentity::new("fea.DynamicViscosity"), scale: 1.0, offset: 0.0, range: QuantityRange::Positive };

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_scalar(request, entry, QUANTITY)
}

const DYNAMIC_VISCOSITY_PASCAL_SECONDS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "value = fea.DynamicViscosity.pascalSeconds(value)",
    inputs: &QUANTITY_INPUTS,
    outputs: &QUANTITY_OUTPUTS,
}];
pub const DYNAMIC_VISCOSITY_PASCAL_SECONDS_DESCRIPTOR: BuiltinDescriptor = quantity_descriptor(&DYNAMIC_VISCOSITY_PASCAL_SECONDS_SIGNATURES);
const DYNAMIC_VISCOSITY_PASCAL_SECONDS_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "dynamic-viscosity-pascalseconds",
    title: "Water at 20 degrees Celsius",
    program: "value = fea.DynamicViscosity.pascalSeconds(1.0e-3);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(value, 'fea.DynamicViscosity'));\nassert(abs(value.pascal_seconds - 0.001) <= 1e-12);",
    },
}];
pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.DynamicViscosity.pascalSeconds",
    documentation: catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.DynamicViscosity.pascalSeconds",
        summary: "Create a typed DynamicViscosity value from pascal-seconds.",
        description: "`value = fea.DynamicViscosity.pascalSeconds(x)` returns a `fea.DynamicViscosity` holding a dynamic viscosity, with `x` given in pascal-seconds and stored in pascal-seconds (Pa*s).",
        keywords: &[
                "fea",
                "quantity",
                "units",
                "DynamicViscosity",
                "pascal-seconds",
            ],
        related: &["fea.FluidDomain.newtonian"],
        sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
                "The argument is a dynamic viscosity in pascal-seconds. It is stored unchanged, and the result is an immutable `fea.DynamicViscosity` whose `pascal_seconds` property holds the canonical value in pascal-seconds (Pa*s).",
                "The input must be a finite real numeric scalar; integer classes are accepted and converted once to double precision. Logical, complex, text, non-scalar, and accelerator-resident values are rejected. The canonical value must be positive; a value outside that range is rejected.",
            ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
                "Pass the value to constructors that take a dynamic viscosity, such as `fea.FluidDomain.newtonian`. Constructors check the value's class, so a quantity of the wrong dimension is rejected rather than silently reinterpreted.",
            ] },
        BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
                "`RunMat:fea:InvalidInput` reports a missing, non-numeric, non-finite, non-scalar, or out-of-range value.",
            ] },
        ],
        examples: DYNAMIC_VISCOSITY_PASCAL_SECONDS_EXAMPLES,
        faqs: super::FAQS,
        links: super::LINKS,
        implementation: super::IMPLEMENTATION,
        verification: super::VERIFICATION,
        notes: &[],
    }),
    descriptor: &DYNAMIC_VISCOSITY_PASCAL_SECONDS_DESCRIPTOR,
    inference: infer,
    integer_capabilities: &QUANTITY_INTEGER_CAPABILITIES,
    integer_audit: None,
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
