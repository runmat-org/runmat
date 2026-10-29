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
    class: identities::FEA_RAYLEIGH_STIFFNESS_COEFFICIENT_CLASS,
    scale: 1.0,
    offset: 0.0,
    range: QuantityRange::NonNegative,
};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_scalar(request, entry, QUANTITY)
}

const RAYLEIGH_STIFFNESS_COEFFICIENT_SECONDS_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "value = fea.RayleighStiffnessCoefficient.seconds(value)",
        inputs: &QUANTITY_INPUTS,
        outputs: &QUANTITY_OUTPUTS,
    }];
pub const RAYLEIGH_STIFFNESS_COEFFICIENT_SECONDS_DESCRIPTOR: BuiltinDescriptor =
    quantity_descriptor(&RAYLEIGH_STIFFNESS_COEFFICIENT_SECONDS_SIGNATURES);
const RAYLEIGH_STIFFNESS_COEFFICIENT_SECONDS_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "rayleigh-stiffness-coefficient-seconds",
    title: "A Rayleigh beta of 1e-5 s",
    program: "value = fea.RayleighStiffnessCoefficient.seconds(1e-5);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(value, 'fea.RayleighStiffnessCoefficient'));\nassert(abs(value.seconds - 1e-05) <= 1e-12);",
    },
}];
pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.RayleighStiffnessCoefficient.seconds",
    documentation: catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.RayleighStiffnessCoefficient.seconds",
        summary: "Create a typed RayleighStiffnessCoefficient value from seconds.",
        description: "`value = fea.RayleighStiffnessCoefficient.seconds(x)` returns a `fea.RayleighStiffnessCoefficient` holding the stiffness-proportional Rayleigh damping coefficient (beta), with `x` given in seconds and stored in seconds (s).",
        keywords: &[
                "fea",
                "quantity",
                "units",
                "RayleighStiffnessCoefficient",
                "seconds",
            ],
        related: &["fea.MechanicalDamping.rayleigh"],
        sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
                "The argument is the stiffness-proportional Rayleigh damping coefficient (beta) in seconds. It is stored unchanged, and the result is an immutable `fea.RayleighStiffnessCoefficient` whose `seconds` property holds the canonical value in seconds (s).",
                "The input must be a finite real numeric scalar; integer classes are accepted and converted once to double precision. Logical, complex, text, non-scalar, and accelerator-resident values are rejected. The canonical value cannot be negative; a value outside that range is rejected.",
            ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
                "Pass the value to constructors that take the stiffness-proportional Rayleigh damping coefficient (beta), such as `fea.MechanicalDamping.rayleigh`. Constructors check the value's class, so a quantity of the wrong dimension is rejected rather than silently reinterpreted.",
            ] },
        BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
                "`RunMat:fea:InvalidInput` reports a missing, non-numeric, non-finite, non-scalar, or out-of-range value.",
            ] },
        ],
        examples: RAYLEIGH_STIFFNESS_COEFFICIENT_SECONDS_EXAMPLES,
        faqs: super::FAQS,
        links: super::LINKS,
        implementation: super::IMPLEMENTATION,
        verification: super::VERIFICATION,
        notes: &[],
    }),
    descriptor: &RAYLEIGH_STIFFNESS_COEFFICIENT_SECONDS_DESCRIPTOR,
    inference: infer,
    integer_capabilities: &QUANTITY_INTEGER_CAPABILITIES,
    integer_audit: None,
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
