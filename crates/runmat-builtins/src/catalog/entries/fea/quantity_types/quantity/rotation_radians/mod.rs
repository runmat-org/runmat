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
    class: identities::FEA_ROTATION_CLASS,
    scale: 1.0,
    offset: 0.0,
    range: QuantityRange::Finite,
};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_scalar(request, entry, QUANTITY)
}

const ROTATION_RADIANS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "value = fea.Rotation.radians(value)",
    inputs: &QUANTITY_INPUTS,
    outputs: &QUANTITY_OUTPUTS,
}];
pub const ROTATION_RADIANS_DESCRIPTOR: BuiltinDescriptor =
    quantity_descriptor(&ROTATION_RADIANS_SIGNATURES);
const ROTATION_RADIANS_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "rotation-radians",
    title: "A 0.01 rad rotation",
    program: "value = fea.Rotation.radians(0.01);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(value, 'fea.Rotation'));\nassert(abs(value.radians - 0.01) <= 1e-12);",
    },
}];
pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.Rotation.radians",
    documentation: catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.Rotation.radians",
        summary: "Create a typed Rotation value from radians.",
        description: "`value = fea.Rotation.radians(x)` returns a `fea.Rotation` holding a rotation angle, with `x` given in radians and stored in radians (rad).",
        keywords: &[
                "fea",
                "quantity",
                "units",
                "Rotation",
                "radians",
            ],
        related: &[
                "fea.Rotation.degrees",
                "fea.StructuralDofValue.rotation",
                "fea.PeriodicTransform.rotation",
            ],
        sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
                "The argument is a rotation angle in radians. It is stored unchanged, and the result is an immutable `fea.Rotation` whose `radians` property holds the canonical value in radians (rad).",
                "The input must be a finite real numeric scalar; integer classes are accepted and converted once to double precision. Logical, complex, text, non-scalar, and accelerator-resident values are rejected. Any finite real value is accepted, including negative values.",
            ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
                "Pass the value to constructors that take a rotation angle, such as `fea.StructuralDofValue.rotation`, `fea.PeriodicTransform.rotation`. Constructors check the value's class, so a quantity of the wrong dimension is rejected rather than silently reinterpreted.",
            ] },
        BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
                "`RunMat:fea:InvalidInput` reports a missing, non-numeric, non-finite, non-scalar, or out-of-range value.",
            ] },
        ],
        examples: ROTATION_RADIANS_EXAMPLES,
        faqs: super::FAQS,
        links: super::LINKS,
        implementation: super::IMPLEMENTATION,
        verification: super::VERIFICATION,
        notes: &[],
    }),
    descriptor: &ROTATION_RADIANS_DESCRIPTOR,
    inference: infer,
    integer_capabilities: &QUANTITY_INTEGER_CAPABILITIES,
    integer_audit: None,
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
