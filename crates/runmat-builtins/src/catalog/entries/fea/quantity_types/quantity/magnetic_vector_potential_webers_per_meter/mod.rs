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
    class: identities::FEA_MAGNETIC_VECTOR_POTENTIAL_CLASS,
    scale: 1.0,
    offset: 0.0,
    range: QuantityRange::Finite,
};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_scalar(request, entry, QUANTITY)
}

const MAGNETIC_VECTOR_POTENTIAL_WEBERS_PER_METER_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "value = fea.MagneticVectorPotential.webersPerMeter(value)",
        inputs: &QUANTITY_INPUTS,
        outputs: &QUANTITY_OUTPUTS,
    }];
pub const MAGNETIC_VECTOR_POTENTIAL_WEBERS_PER_METER_DESCRIPTOR: BuiltinDescriptor =
    quantity_descriptor(&MAGNETIC_VECTOR_POTENTIAL_WEBERS_PER_METER_SIGNATURES);
const MAGNETIC_VECTOR_POTENTIAL_WEBERS_PER_METER_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "magnetic-vector-potential-weberspermeter",
    title: "A zero vector-potential boundary value",
    program: "value = fea.MagneticVectorPotential.webersPerMeter(0);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(value, 'fea.MagneticVectorPotential'));\nassert(abs(value.webers_per_meter - 0) <= 1e-12);",
    },
}];
pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.MagneticVectorPotential.webersPerMeter",
    documentation: catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.MagneticVectorPotential.webersPerMeter",
        summary: "Create a typed MagneticVectorPotential value from webers per meter.",
        description: "`value = fea.MagneticVectorPotential.webersPerMeter(x)` returns a `fea.MagneticVectorPotential` holding a magnetic vector potential component, with `x` given in webers per meter and stored in webers per meter (Wb/m).",
        keywords: &[
                "fea",
                "quantity",
                "units",
                "MagneticVectorPotential",
                "webers per meter",
            ],
        related: &["fea.MagneticConstraint.prescribedVectorPotential"],
        sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
                "The argument is a magnetic vector potential component in webers per meter. It is stored unchanged, and the result is an immutable `fea.MagneticVectorPotential` whose `webers_per_meter` property holds the canonical value in webers per meter (Wb/m).",
                "The input must be a finite real numeric scalar; integer classes are accepted and converted once to double precision. Logical, complex, text, non-scalar, and accelerator-resident values are rejected. Any finite real value is accepted, including negative values.",
            ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
                "Pass the value to constructors that take a magnetic vector potential component, such as `fea.MagneticConstraint.prescribedVectorPotential`. Constructors check the value's class, so a quantity of the wrong dimension is rejected rather than silently reinterpreted.",
            ] },
        BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
                "`RunMat:fea:InvalidInput` reports a missing, non-numeric, non-finite, non-scalar, or out-of-range value.",
            ] },
        ],
        examples: MAGNETIC_VECTOR_POTENTIAL_WEBERS_PER_METER_EXAMPLES,
        faqs: super::FAQS,
        links: super::LINKS,
        implementation: super::IMPLEMENTATION,
        verification: super::VERIFICATION,
        notes: &[],
    }),
    descriptor: &MAGNETIC_VECTOR_POTENTIAL_WEBERS_PER_METER_DESCRIPTOR,
    inference: infer,
    integer_capabilities: &QUANTITY_INTEGER_CAPABILITIES,
    integer_audit: None,
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
