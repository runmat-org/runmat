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
    class: identities::FEA_VELOCITY_CLASS,
    scale: 1.0,
    offset: 0.0,
    range: QuantityRange::Finite,
};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_scalar(request, entry, QUANTITY)
}

const VELOCITY_METERS_PER_SECOND_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "value = fea.Velocity.metersPerSecond(value)",
        inputs: &QUANTITY_INPUTS,
        outputs: &QUANTITY_OUTPUTS,
    }];
pub const VELOCITY_METERS_PER_SECOND_DESCRIPTOR: BuiltinDescriptor =
    quantity_descriptor(&VELOCITY_METERS_PER_SECOND_SIGNATURES);
const VELOCITY_METERS_PER_SECOND_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "velocity-meterspersecond",
    title: "An inlet speed of 2.5 m/s",
    program: "value = fea.Velocity.metersPerSecond(2.5);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(value, 'fea.Velocity'));\nassert(abs(value.meters_per_second - 2.5) <= 1e-12);",
    },
}];
pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.Velocity.metersPerSecond",
    documentation: catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.Velocity.metersPerSecond",
        summary: "Create a typed Velocity value from meters per second.",
        description: "`value = fea.Velocity.metersPerSecond(x)` returns a `fea.Velocity` holding a speed, with `x` given in meters per second and stored in meters per second (m/s).",
        keywords: &[
                "fea",
                "quantity",
                "units",
                "Velocity",
                "meters per second",
            ],
        related: &[
                "fea.FluidConstraint.prescribedVelocity",
                "fea.FluidConstraint.movingWall",
            ],
        sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
                "The argument is a speed in meters per second. It is stored unchanged, and the result is an immutable `fea.Velocity` whose `meters_per_second` property holds the canonical value in meters per second (m/s).",
                "The input must be a finite real numeric scalar; integer classes are accepted and converted once to double precision. Logical, complex, text, non-scalar, and accelerator-resident values are rejected. Any finite real value is accepted, including negative values.",
            ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
                "Pass the value to constructors that take a speed, such as `fea.FluidConstraint.prescribedVelocity`, `fea.FluidConstraint.movingWall`. Constructors check the value's class, so a quantity of the wrong dimension is rejected rather than silently reinterpreted.",
            ] },
        BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
                "`RunMat:fea:InvalidInput` reports a missing, non-numeric, non-finite, non-scalar, or out-of-range value.",
            ] },
        ],
        examples: VELOCITY_METERS_PER_SECOND_EXAMPLES,
        faqs: super::FAQS,
        links: super::LINKS,
        implementation: super::IMPLEMENTATION,
        verification: super::VERIFICATION,
        notes: &[],
    }),
    descriptor: &VELOCITY_METERS_PER_SECOND_DESCRIPTOR,
    inference: infer,
    integer_capabilities: &QUANTITY_INTEGER_CAPABILITIES,
    integer_audit: None,
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
