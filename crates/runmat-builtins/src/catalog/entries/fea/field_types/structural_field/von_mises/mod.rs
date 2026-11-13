use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition, ConstructorEntryDefinition},
    selector_entry::{self, NO_INPUTS, SELECTOR_OUTPUT},
};
use crate::*;

const VON_MISES_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "field = fea.StructuralField.vonMises()",
    inputs: &NO_INPUTS,
    outputs: &SELECTOR_OUTPUT,
}];
pub const VON_MISES_DESCRIPTOR: BuiltinDescriptor =
    selector_entry::selector_descriptor(&VON_MISES_SIGNATURES);
const VON_MISES_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "select-von-mises",
    title: "Select element von Mises stress",
    program: "selector = fea.StructuralField.vonMises();",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(selector, 'fea.StructuralField'));\nassert(strcmp(selector.value, \"von_mises\"));",
    },
}];
pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.StructuralField.vonMises",
    documentation: catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.StructuralField.vonMises",
        summary: "Select the element von Mises stress result field for fea.field.",
        description: "`field = fea.StructuralField.vonMises()` returns a `fea.StructuralField` selector for `structural.von_mises`: von Mises equivalent stress, one value per element, computed from the element stress tensor. Pass it to `fea.field` to read the field from a run.",
        keywords: &[
                "fea",
                "field",
                "structural",
                "element von Mises stress",
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
                "The selector is a small immutable value; it does no I/O. Its `value` property is `\"von_mises\"`, and `fea.field` resolves it to the stored field id `structural.von_mises`.",
                "Physical quantity: von Mises equivalent stress, one value per element, computed from the element stress tensor. Units: pascals (Pa). Shape: a scalar per element. Location: element.",
                "Produced by structural static steps (`fea.StructuralStep.static`), and nonlinear structural steps. A run that did not produce the field makes `fea.field` fail with `RunMat:fea:OperationFailed`. Use it for yield checks against the material's yield stress. It is a nonlinear function of stress, so statistical selectors do not accept it; use `fea.StructuralField.stress()` with them instead.",
            ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
                "After `run = fea.run(study)`, call `fea.field(run, fea.StructuralField.vonMises())` (or pass `fea.results(run)` first), then read values with `fea.fieldPage` or draw them with `plot(field)`.",
            ] },
        ],
        examples: VON_MISES_EXAMPLES,
        faqs: super::FAQS,
        links: super::LINKS,
        implementation: super::IMPLEMENTATION,
        verification: super::VERIFICATION,
        notes: &[],
    }),
    descriptor: &VON_MISES_DESCRIPTOR,
    inference: selector_entry::infer_structural,
    integer_capabilities: &[],
    integer_audit: Some(&selector_entry::STRUCTURAL_SELECTOR_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
