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
    infer_vector(request, entry, identities::FEA_TRANSLATION_CLASS)
}

const TRANSLATION_MILLIMETERS_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "translation = fea.Translation.millimeters(components)",
        inputs: &VECTOR_QUANTITY_INPUTS,
        outputs: &VECTOR_QUANTITY_OUTPUTS,
    }];
pub const TRANSLATION_MILLIMETERS_DESCRIPTOR: BuiltinDescriptor =
    quantity_descriptor(&TRANSLATION_MILLIMETERS_SIGNATURES);
const TRANSLATION_MILLIMETERS_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "translation-millimeters",
    title: "A 50 mm offset along y",
    program: "value = fea.Translation.millimeters([0 50 0]);",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(value, 'fea.Translation'));\nassert(max(abs(value.meters - [0 0.05 0])) <= 1e-12);",
    },
}];
pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.Translation.millimeters",
    documentation: catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.Translation.millimeters",
        summary: "Create a typed three-component Translation from millimeters.",
        description: "`value = fea.Translation.millimeters([x y z])` returns a `fea.Translation` holding a translation vector, with components in millimeters and stored in meters as a 1-by-3 row.",
        keywords: &[
                "fea",
                "quantity",
                "units",
                "vector",
                "Translation",
                "millimeters",
            ],
        related: &[
                "fea.Translation.meters",
                "fea.PeriodicTransform.translation",
            ],
        sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
                "The argument is a real numeric vector with exactly three elements (x, y, z) in millimeters, in the global frame. Each component is multiplied by 0.001 to give meters, and the result is an immutable `fea.Translation` whose `meters` property is a 1-by-3 row in meters.",
                "Every component must be finite; integer classes are accepted and converted once to double precision. Vectors of any other length, matrices, complex, logical, text, and accelerator-resident values are rejected.",
            ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
                "Pass the value to constructors that take a translation vector, such as `fea.PeriodicTransform.translation`.",
            ] },
        BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
                "`RunMat:fea:InvalidInput` reports a missing argument, a non-numeric value, a vector without exactly three elements, or a non-finite component.",
            ] },
        ],
        examples: TRANSLATION_MILLIMETERS_EXAMPLES,
        faqs: super::FAQS,
        links: super::LINKS,
        implementation: super::IMPLEMENTATION,
        verification: super::VERIFICATION,
        notes: &[],
    }),
    descriptor: &TRANSLATION_MILLIMETERS_DESCRIPTOR,
    inference: infer,
    integer_capabilities: &VECTOR_QUANTITY_INTEGER_CAPABILITIES,
    integer_audit: None,
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
