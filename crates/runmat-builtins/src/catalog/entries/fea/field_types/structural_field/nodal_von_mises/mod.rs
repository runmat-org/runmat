use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition, ConstructorEntryDefinition},
    selector_entry::{self, NO_INPUTS, SELECTOR_OUTPUT},
};
use crate::*;

const NODAL_VON_MISES_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "field = fea.StructuralField.nodalVonMises()",
    inputs: &NO_INPUTS,
    outputs: &SELECTOR_OUTPUT,
}];
pub const NODAL_VON_MISES_DESCRIPTOR: BuiltinDescriptor =
    selector_entry::selector_descriptor(&NODAL_VON_MISES_SIGNATURES);
const NODAL_VON_MISES_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "select-nodal-von-mises",
    title: "Select nodal von Mises stress",
    program: "selector = fea.StructuralField.nodalVonMises();",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(selector, 'fea.StructuralField'));\nassert(strcmp(selector.value, \"nodal_von_mises\"));",
    },
}];
pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.StructuralField.nodalVonMises",
    documentation: catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.StructuralField.nodalVonMises",
        summary: "Select the nodal von Mises stress result field for fea.field.",
        description: "`field = fea.StructuralField.nodalVonMises()` returns a `fea.StructuralField` selector for `structural.nodal_von_mises`: von Mises stress transferred from elements to nodes by volume-weighted averaging, which gives a smooth contour for plots. Pass it to `fea.field` to read the field from a run.",
        keywords: &[
                "fea",
                "field",
                "structural",
                "nodal von Mises stress",
                "post-processing",
            ],
        related: &[
                "fea.field",
                "fea.results",
                "fea.fieldPage",
                "fea.run",
            ],
        sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
                "The selector is a small immutable value; it does no I/O. Its `value` property is `\"nodal_von_mises\"`, and `fea.field` resolves it to the stored field id `structural.nodal_von_mises`.",
                "Physical quantity: von Mises stress transferred from elements to nodes by volume-weighted averaging, which gives a smooth contour for plots. Units: pascals (Pa). Shape: a scalar per node. Location: node.",
                "Produced by structural static steps (`fea.StructuralStep.static`). A run that did not produce the field makes `fea.field` fail with `RunMat:fea:OperationFailed`. Averaging smooths peaks; use `fea.StructuralField.vonMises()` for the element values a yield check should use.",
            ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
                "After `run = fea.run(study)`, call `fea.field(run, fea.StructuralField.nodalVonMises())` (or pass `fea.results(run)` first), then read values with `fea.fieldPage` or draw them with `plot(field)`.",
            ] },
        ],
        examples: NODAL_VON_MISES_EXAMPLES,
        faqs: super::FAQS,
        links: super::LINKS,
        implementation: super::IMPLEMENTATION,
        verification: super::VERIFICATION,
        notes: &[],
    }),
    descriptor: &NODAL_VON_MISES_DESCRIPTOR,
    inference: selector_entry::infer_structural,
    integer_capabilities: &[],
    integer_audit: Some(&selector_entry::STRUCTURAL_SELECTOR_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
