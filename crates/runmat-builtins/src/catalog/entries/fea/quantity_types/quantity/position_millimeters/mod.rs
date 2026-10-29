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

const POSITION_MILLIMETERS_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "position = fea.Position.millimeters(coordinates)",
        inputs: &VECTOR_QUANTITY_INPUTS,
        outputs: &VECTOR_QUANTITY_OUTPUTS,
    }];
pub const POSITION_MILLIMETERS_DESCRIPTOR: BuiltinDescriptor =
    quantity_descriptor(&POSITION_MILLIMETERS_SIGNATURES);
const POSITION_MILLIMETERS_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "position-millimeters",
    title: "A point in millimeters",
    program: "value = fea.Position.millimeters([10 20 30]);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(value, 'fea.Position'));\nassert(max(abs(value.meters - [0.01 0.02 0.03])) <= 1e-12);",
    },
}];
pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.Position.millimeters",
    documentation: catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.Position.millimeters",
        summary: "Create a typed three-component Position from millimeters.",
        description: "`value = fea.Position.millimeters([x y z])` returns a `fea.Position` holding a point in the global frame, with components in millimeters and stored in meters as a 1-by-3 row.",
        keywords: &[
                "fea",
                "quantity",
                "units",
                "vector",
                "Position",
                "millimeters",
            ],
        related: &[
                "fea.Position.meters",
                "fea.ReferencePoint.fromCoordinates",
                "fea.PeriodicTransform.rotation",
            ],
        sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
                "The argument is a real numeric vector with exactly three elements (x, y, z) in millimeters, in the global frame. Each component is multiplied by 0.001 to give meters, and the result is an immutable `fea.Position` whose `meters` property is a 1-by-3 row in meters.",
                "Every component must be finite; integer classes are accepted and converted once to double precision. Vectors of any other length, matrices, complex, logical, text, and accelerator-resident values are rejected.",
            ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
                "Pass the value to constructors that take a point in the global frame, such as `fea.ReferencePoint.fromCoordinates`, `fea.PeriodicTransform.rotation`.",
            ] },
        BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
                "`RunMat:fea:InvalidInput` reports a missing argument, a non-numeric value, a vector without exactly three elements, or a non-finite component.",
            ] },
        ],
        examples: POSITION_MILLIMETERS_EXAMPLES,
        faqs: super::FAQS,
        links: super::LINKS,
        implementation: super::IMPLEMENTATION,
        verification: super::VERIFICATION,
        notes: &[],
    }),
    descriptor: &POSITION_MILLIMETERS_DESCRIPTOR,
    inference: infer,
    integer_capabilities: &VECTOR_QUANTITY_INTEGER_CAPABILITIES,
    integer_audit: None,
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
