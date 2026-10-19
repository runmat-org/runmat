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

const QUANTITY: ScalarQuantity = ScalarQuantity { class: identities::FEA_TURBULENCE_INTENSITY_CLASS, scale: 1.0, offset: 0.0, range: QuantityRange::UnitInterval };

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_scalar(request, entry, QUANTITY)
}

const TURBULENCE_INTENSITY_RATIO_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "value = fea.TurbulenceIntensity.ratio(value)",
    inputs: &QUANTITY_INPUTS,
    outputs: &QUANTITY_OUTPUTS,
}];
pub const TURBULENCE_INTENSITY_RATIO_DESCRIPTOR: BuiltinDescriptor = quantity_descriptor(&TURBULENCE_INTENSITY_RATIO_SIGNATURES);
const TURBULENCE_INTENSITY_RATIO_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "turbulence-intensity-ratio",
    title: "A 5 percent turbulence intensity",
    program: "value = fea.TurbulenceIntensity.ratio(0.05);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(value, 'fea.TurbulenceIntensity'));\nassert(abs(value.ratio - 0.05) <= 1e-12);",
    },
}];
pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.TurbulenceIntensity.ratio",
    documentation: catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.TurbulenceIntensity.ratio",
        summary: "Create a typed TurbulenceIntensity value from a dimensionless ratio.",
        description: "`value = fea.TurbulenceIntensity.ratio(x)` returns a `fea.TurbulenceIntensity` holding a turbulence intensity, with `x` given in a dimensionless ratio and stored in a dimensionless ratio.",
        keywords: &[
                "fea",
                "quantity",
                "units",
                "TurbulenceIntensity",
                "a dimensionless ratio",
            ],
        related: &["fea.FluidDomain.newtonian"],
        sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
                "The argument is a turbulence intensity in a dimensionless ratio. It is stored unchanged, and the result is an immutable `fea.TurbulenceIntensity` whose `ratio` property holds the canonical value in a dimensionless ratio.",
                "The input must be a finite real numeric scalar; integer classes are accepted and converted once to double precision. Logical, complex, text, non-scalar, and accelerator-resident values are rejected. The canonical value must lie in the closed interval [0, 1]; a value outside that range is rejected.",
            ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
                "Pass the value to constructors that take a turbulence intensity, such as `fea.FluidDomain.newtonian`. Constructors check the value's class, so a quantity of the wrong dimension is rejected rather than silently reinterpreted.",
            ] },
        BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
                "`RunMat:fea:InvalidInput` reports a missing, non-numeric, non-finite, non-scalar, or out-of-range value.",
            ] },
        ],
        examples: TURBULENCE_INTENSITY_RATIO_EXAMPLES,
        faqs: super::FAQS,
        links: super::LINKS,
        implementation: super::IMPLEMENTATION,
        verification: super::VERIFICATION,
        notes: &[],
    }),
    descriptor: &TURBULENCE_INTENSITY_RATIO_DESCRIPTOR,
    inference: infer,
    integer_capabilities: &QUANTITY_INTEGER_CAPABILITIES,
    integer_audit: None,
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
