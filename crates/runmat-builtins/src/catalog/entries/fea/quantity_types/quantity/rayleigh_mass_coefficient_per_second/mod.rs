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
    class: identities::FEA_RAYLEIGH_MASS_COEFFICIENT_CLASS,
    scale: 1.0,
    offset: 0.0,
    range: QuantityRange::NonNegative,
};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_scalar(request, entry, QUANTITY)
}

const RAYLEIGH_MASS_COEFFICIENT_PER_SECOND_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "value = fea.RayleighMassCoefficient.perSecond(value)",
        inputs: &QUANTITY_INPUTS,
        outputs: &QUANTITY_OUTPUTS,
    }];
pub const RAYLEIGH_MASS_COEFFICIENT_PER_SECOND_DESCRIPTOR: BuiltinDescriptor =
    quantity_descriptor(&RAYLEIGH_MASS_COEFFICIENT_PER_SECOND_SIGNATURES);
const RAYLEIGH_MASS_COEFFICIENT_PER_SECOND_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "rayleigh-mass-coefficient-persecond",
    title: "A Rayleigh alpha of 0.5 1/s",
    program: "value = fea.RayleighMassCoefficient.perSecond(0.5);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(value, 'fea.RayleighMassCoefficient'));\nassert(abs(value.per_second - 0.5) <= 1e-12);",
    },
}];
pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.RayleighMassCoefficient.perSecond",
    documentation: catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.RayleighMassCoefficient.perSecond",
        summary: "Create a typed RayleighMassCoefficient value from per second.",
        description: "`value = fea.RayleighMassCoefficient.perSecond(x)` returns a `fea.RayleighMassCoefficient` holding the mass-proportional Rayleigh damping coefficient (alpha), with `x` given in per second and stored in per second (1/s).",
        keywords: &[
                "fea",
                "quantity",
                "units",
                "RayleighMassCoefficient",
                "per second",
            ],
        related: &["fea.MechanicalDamping.rayleigh"],
        sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
                "The argument is the mass-proportional Rayleigh damping coefficient (alpha) in per second. It is stored unchanged, and the result is an immutable `fea.RayleighMassCoefficient` whose `per_second` property holds the canonical value in per second (1/s).",
                "The input must be a finite real numeric scalar; integer classes are accepted and converted once to double precision. Logical, complex, text, non-scalar, and accelerator-resident values are rejected. The canonical value cannot be negative; a value outside that range is rejected.",
            ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
                "Pass the value to constructors that take the mass-proportional Rayleigh damping coefficient (alpha), such as `fea.MechanicalDamping.rayleigh`. Constructors check the value's class, so a quantity of the wrong dimension is rejected rather than silently reinterpreted.",
            ] },
        BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
                "`RunMat:fea:InvalidInput` reports a missing, non-numeric, non-finite, non-scalar, or out-of-range value.",
            ] },
        ],
        examples: RAYLEIGH_MASS_COEFFICIENT_PER_SECOND_EXAMPLES,
        faqs: super::FAQS,
        links: super::LINKS,
        implementation: super::IMPLEMENTATION,
        verification: super::VERIFICATION,
        notes: &[],
    }),
    descriptor: &RAYLEIGH_MASS_COEFFICIENT_PER_SECOND_DESCRIPTOR,
    inference: infer,
    integer_capabilities: &QUANTITY_INTEGER_CAPABILITIES,
    integer_audit: None,
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
