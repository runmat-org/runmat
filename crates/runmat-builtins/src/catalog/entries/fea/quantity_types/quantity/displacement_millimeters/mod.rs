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
    class: identities::FEA_DISPLACEMENT_CLASS,
    scale: 1.0e-3,
    offset: 0.0,
    range: QuantityRange::Finite,
};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_scalar(request, entry, QUANTITY)
}

const DISPLACEMENT_MILLIMETERS_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "value = fea.Displacement.millimeters(value)",
        inputs: &QUANTITY_INPUTS,
        outputs: &QUANTITY_OUTPUTS,
    }];
pub const DISPLACEMENT_MILLIMETERS_DESCRIPTOR: BuiltinDescriptor =
    quantity_descriptor(&DISPLACEMENT_MILLIMETERS_SIGNATURES);
const DISPLACEMENT_MILLIMETERS_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "displacement-millimeters",
    title: "A 2.5 mm displacement",
    program: "value = fea.Displacement.millimeters(2.5);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source:
            "assert(isa(value, 'fea.Displacement'));\nassert(abs(value.meters - 0.0025) <= 1e-12);",
    },
}];
pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.Displacement.millimeters",
    documentation: catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.Displacement.millimeters",
        summary: "Create a typed Displacement value from millimeters.",
        description: "`value = fea.Displacement.millimeters(x)` returns a `fea.Displacement` holding a translational displacement, with `x` given in millimeters and stored in meters (m).",
        keywords: &[
                "fea",
                "quantity",
                "units",
                "Displacement",
                "millimeters",
            ],
        related: &[
                "fea.Displacement.meters",
                "fea.Displacement.micrometers",
                "fea.StructuralDofValue.translation",
                "fea.StructuralConstraint.prescribed",
            ],
        sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
                "The argument is a translational displacement in millimeters. It is multiplied by 0.001 to give meters (m), and the result is an immutable `fea.Displacement` whose `meters` property holds the canonical value in meters (m).",
                "The input must be a finite real numeric scalar; integer classes are accepted and converted once to double precision. Logical, complex, text, non-scalar, and accelerator-resident values are rejected. Any finite real value is accepted, including negative values.",
            ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
                "Pass the value to constructors that take a translational displacement, such as `fea.StructuralDofValue.translation`, `fea.StructuralConstraint.prescribed`. Constructors check the value's class, so a quantity of the wrong dimension is rejected rather than silently reinterpreted.",
            ] },
        BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
                "`RunMat:fea:InvalidInput` reports a missing, non-numeric, non-finite, non-scalar, or out-of-range value.",
            ] },
        ],
        examples: DISPLACEMENT_MILLIMETERS_EXAMPLES,
        faqs: super::FAQS,
        links: super::LINKS,
        implementation: super::IMPLEMENTATION,
        verification: super::VERIFICATION,
        notes: &[],
    }),
    descriptor: &DISPLACEMENT_MILLIMETERS_DESCRIPTOR,
    inference: infer,
    integer_capabilities: &QUANTITY_INTEGER_CAPABILITIES,
    integer_audit: None,
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
