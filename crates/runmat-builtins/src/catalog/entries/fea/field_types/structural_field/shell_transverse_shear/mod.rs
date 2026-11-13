use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition, ConstructorEntryDefinition},
    selector_entry::{self, NO_INPUTS, SELECTOR_OUTPUT},
};
use crate::*;

const SHELL_TRANSVERSE_SHEAR_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "field = fea.StructuralField.shellTransverseShear()",
        inputs: &NO_INPUTS,
        outputs: &SELECTOR_OUTPUT,
    }];
pub const SHELL_TRANSVERSE_SHEAR_DESCRIPTOR: BuiltinDescriptor =
    selector_entry::selector_descriptor(&SHELL_TRANSVERSE_SHEAR_SIGNATURES);
const SHELL_TRANSVERSE_SHEAR_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "select-shell-transverse-shear",
    title: "Select shell transverse shear",
    program: "selector = fea.StructuralField.shellTransverseShear();",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(selector, 'fea.StructuralField'));\nassert(strcmp(selector.value, \"shell_transverse_shear\"));",
    },
}];
pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.StructuralField.shellTransverseShear",
    documentation: catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.StructuralField.shellTransverseShear",
        summary: "Select the shell transverse shear result field for fea.field.",
        description: "`field = fea.StructuralField.shellTransverseShear()` returns a `fea.StructuralField` selector for `structural.shell_transverse_shear`: transverse shear force resultants per unit width in each shell element. Pass it to `fea.field` to read the field from a run.",
        keywords: &[
                "fea",
                "field",
                "structural",
                "shell transverse shear",
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
                "The selector is a small immutable value; it does no I/O. Its `value` property is `\"shell_transverse_shear\"`, and `fea.field` resolves it to the stored field id `structural.shell_transverse_shear`.",
                "Physical quantity: transverse shear force resultants per unit width in each shell element. Units: newtons per meter (N/m). Shape: two components per shell element. Location: element.",
                "Produced by structural static steps on models with shell elements. A run that did not produce the field makes `fea.field` fail with `RunMat:fea:OperationFailed`. It is a linear recovery quantity, so it can also be passed to `fea.ResponseSpectrumField.envelope`, `fea.RandomResponseField.rms`, and `fea.RandomResponseField.variance`.",
            ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
                "After `run = fea.run(study)`, call `fea.field(run, fea.StructuralField.shellTransverseShear())` (or pass `fea.results(run)` first), then read values with `fea.fieldPage` or draw them with `plot(field)`.",
            ] },
        ],
        examples: SHELL_TRANSVERSE_SHEAR_EXAMPLES,
        faqs: super::FAQS,
        links: super::LINKS,
        implementation: super::IMPLEMENTATION,
        verification: super::VERIFICATION,
        notes: &[],
    }),
    descriptor: &SHELL_TRANSVERSE_SHEAR_DESCRIPTOR,
    inference: selector_entry::infer_structural,
    integer_capabilities: &[],
    integer_audit: Some(&selector_entry::STRUCTURAL_SELECTOR_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
