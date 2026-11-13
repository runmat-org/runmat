use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition, ConstructorEntryDefinition},
    identities,
    quantity_entry::{
        infer_vector, quantity_descriptor, VECTOR_QUANTITY_INPUTS,
        VECTOR_QUANTITY_INTEGER_CAPABILITIES, VECTOR_QUANTITY_OUTPUTS,
    },
};
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    infer_vector(request, entry, identities::FEA_POSITION_CLASS)
}

const POSITION_METERS_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "position = fea.Position.meters(coordinates)",
    inputs: &VECTOR_QUANTITY_INPUTS,
    outputs: &VECTOR_QUANTITY_OUTPUTS,
}];
pub const POSITION_METERS_DESCRIPTOR: BuiltinDescriptor =
    quantity_descriptor(&POSITION_METERS_SIGNATURES);
const POSITION_METERS_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "position-meters",
    title: "A point in meters",
    program: "value = fea.Position.meters([0.1 0.2 0.3]);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(value, 'fea.Position'));\nassert(max(abs(value.meters - [0.1 0.2 0.3])) <= 1e-12);",
    },
}];
pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.Position.meters",
    documentation: catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.Position.meters",
        summary: "Create a typed three-component Position from meters.",
        description: "`value = fea.Position.meters([x y z])` returns a `fea.Position` holding a point in the global frame, with components in meters and stored in meters as a 1-by-3 row.",
        keywords: &[
                "fea",
                "quantity",
                "units",
                "vector",
                "Position",
                "meters",
            ],
        related: &[
                "fea.Position.millimeters",
                "fea.ReferencePoint.fromCoordinates",
                "fea.PeriodicTransform.rotation",
            ],
        sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
                "The argument is a real numeric vector with exactly three elements (x, y, z) in meters, in the global frame. Each component is stored unchanged, and the result is an immutable `fea.Position` whose `meters` property is a 1-by-3 row in meters.",
                "Every component must be finite; integer classes are accepted and converted once to double precision. Vectors of any other length, matrices, complex, logical, text, and accelerator-resident values are rejected.",
            ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
                "Pass the value to constructors that take a point in the global frame, such as `fea.ReferencePoint.fromCoordinates`, `fea.PeriodicTransform.rotation`.",
            ] },
        BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
                "`RunMat:fea:InvalidInput` reports a missing argument, a non-numeric value, a vector without exactly three elements, or a non-finite component.",
            ] },
        ],
        examples: POSITION_METERS_EXAMPLES,
        faqs: super::FAQS,
        links: super::LINKS,
        implementation: super::IMPLEMENTATION,
        verification: super::VERIFICATION,
        notes: &[],
    }),
    descriptor: &POSITION_METERS_DESCRIPTOR,
    inference: infer,
    integer_capabilities: &VECTOR_QUANTITY_INTEGER_CAPABILITIES,
    integer_audit: None,
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
