use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition, ConstructorEntryDefinition},
    selector_entry::{self, NO_INPUTS, SELECTOR_OUTPUT},
};
use crate::*;

const SHELL_VON_MISES_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "field = fea.StructuralField.shellVonMises()",
    inputs: &NO_INPUTS,
    outputs: &SELECTOR_OUTPUT,
}];
pub const SHELL_VON_MISES_DESCRIPTOR: BuiltinDescriptor =
    selector_entry::selector_descriptor(&SHELL_VON_MISES_SIGNATURES);
const SHELL_VON_MISES_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "select-shell-von-mises",
    title: "Select shell von Mises stress",
    program: "selector = fea.StructuralField.shellVonMises();",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(selector, 'fea.StructuralField'));\nassert(strcmp(selector.value, \"shell_von_mises\"));",
    },
}];
pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.StructuralField.shellVonMises",
    documentation: catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.StructuralField.shellVonMises",
        summary: "Select the shell von Mises stress result field for fea.field.",
        description: "`field = fea.StructuralField.shellVonMises()` returns a `fea.StructuralField` selector for `structural.shell_von_mises`: von Mises equivalent stress in each shell element, from the membrane and bending stress state. Pass it to `fea.field` to read the field from a run.",
        keywords: &[
                "fea",
                "field",
                "structural",
                "shell von Mises stress",
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
                "The selector is a small immutable value; it does no I/O. Its `value` property is `\"shell_von_mises\"`, and `fea.field` resolves it to the stored field id `structural.shell_von_mises`.",
                "Physical quantity: von Mises equivalent stress in each shell element, from the membrane and bending stress state. Units: pascals (Pa). Shape: a scalar per shell element. Location: element.",
                "Produced by structural static steps on models with shell elements. A run that did not produce the field makes `fea.field` fail with `RunMat:fea:OperationFailed`. Use it for yield checks of thin-walled parts.",
            ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
                "After `run = fea.run(study)`, call `fea.field(run, fea.StructuralField.shellVonMises())` (or pass `fea.results(run)` first), then read values with `fea.fieldPage` or draw them with `plot(field)`.",
            ] },
        ],
        examples: SHELL_VON_MISES_EXAMPLES,
        faqs: super::FAQS,
        links: super::LINKS,
        implementation: super::IMPLEMENTATION,
        verification: super::VERIFICATION,
        notes: &[],
    }),
    descriptor: &SHELL_VON_MISES_DESCRIPTOR,
    inference: selector_entry::infer_structural,
    integer_capabilities: &[],
    integer_audit: Some(&selector_entry::STRUCTURAL_SELECTOR_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
