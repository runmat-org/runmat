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
    class: identities::FEA_FRICTION_COEFFICIENT_CLASS,
    scale: 1.0,
    offset: 0.0,
    range: QuantityRange::NonNegative,
};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_scalar(request, entry, QUANTITY)
}

const FRICTION_COEFFICIENT_RATIO_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "value = fea.FrictionCoefficient.ratio(value)",
        inputs: &QUANTITY_INPUTS,
        outputs: &QUANTITY_OUTPUTS,
    }];
pub const FRICTION_COEFFICIENT_RATIO_DESCRIPTOR: BuiltinDescriptor =
    quantity_descriptor(&FRICTION_COEFFICIENT_RATIO_SIGNATURES);
const FRICTION_COEFFICIENT_RATIO_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "friction-coefficient-ratio",
    title: "Steel on steel, lubricated",
    program: "value = fea.FrictionCoefficient.ratio(0.3);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(value, 'fea.FrictionCoefficient'));\nassert(abs(value.ratio - 0.3) <= 1e-12);",
    },
}];
pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.FrictionCoefficient.ratio",
    documentation: catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.FrictionCoefficient.ratio",
        summary: "Create a typed FrictionCoefficient value from a dimensionless ratio.",
        description: "`value = fea.FrictionCoefficient.ratio(x)` returns a `fea.FrictionCoefficient` holding a Coulomb friction coefficient, with `x` given in a dimensionless ratio and stored in a dimensionless ratio.",
        keywords: &[
                "fea",
                "quantity",
                "units",
                "FrictionCoefficient",
                "a dimensionless ratio",
            ],
        related: &["fea.ContactInterface.coulomb"],
        sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
                "The argument is a Coulomb friction coefficient in a dimensionless ratio. It is stored unchanged, and the result is an immutable `fea.FrictionCoefficient` whose `ratio` property holds the canonical value in a dimensionless ratio.",
                "The input must be a finite real numeric scalar; integer classes are accepted and converted once to double precision. Logical, complex, text, non-scalar, and accelerator-resident values are rejected. The canonical value cannot be negative; a value outside that range is rejected.",
            ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
                "Pass the value to constructors that take a Coulomb friction coefficient, such as `fea.ContactInterface.coulomb`. Constructors check the value's class, so a quantity of the wrong dimension is rejected rather than silently reinterpreted.",
            ] },
        BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
                "`RunMat:fea:InvalidInput` reports a missing, non-numeric, non-finite, non-scalar, or out-of-range value.",
            ] },
        ],
        examples: FRICTION_COEFFICIENT_RATIO_EXAMPLES,
        faqs: super::FAQS,
        links: super::LINKS,
        implementation: super::IMPLEMENTATION,
        verification: super::VERIFICATION,
        notes: &[],
    }),
    descriptor: &FRICTION_COEFFICIENT_RATIO_DESCRIPTOR,
    inference: infer,
    integer_capabilities: &QUANTITY_INTEGER_CAPABILITIES,
    integer_audit: None,
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
