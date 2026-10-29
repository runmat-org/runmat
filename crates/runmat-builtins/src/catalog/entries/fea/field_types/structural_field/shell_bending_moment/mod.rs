use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition, ConstructorEntryDefinition},
    selector_entry::{self, NO_INPUTS, SELECTOR_OUTPUT},
};
use crate::*;

const SHELL_BENDING_MOMENT_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "field = fea.StructuralField.shellBendingMoment()",
        inputs: &NO_INPUTS,
        outputs: &SELECTOR_OUTPUT,
    }];
pub const SHELL_BENDING_MOMENT_DESCRIPTOR: BuiltinDescriptor =
    selector_entry::selector_descriptor(&SHELL_BENDING_MOMENT_SIGNATURES);
const SHELL_BENDING_MOMENT_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "select-shell-bending-moment",
    title: "Select shell bending moment",
    program: "selector = fea.StructuralField.shellBendingMoment();",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(selector, 'fea.StructuralField'));\nassert(strcmp(selector.value, \"shell_bending_moment\"));",
    },
}];
pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.StructuralField.shellBendingMoment",
    documentation: catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.StructuralField.shellBendingMoment",
        summary: "Select the shell bending moment result field for fea.field.",
        description: "`field = fea.StructuralField.shellBendingMoment()` returns a `fea.StructuralField` selector for `structural.shell_bending_moment`: bending and twisting moment resultants per unit width in each shell element. Pass it to `fea.field` to read the field from a run.",
        keywords: &[
                "fea",
                "field",
                "structural",
                "shell bending moment",
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
                "The selector is a small immutable value; it does no I/O. Its `value` property is `\"shell_bending_moment\"`, and `fea.field` resolves it to the stored field id `structural.shell_bending_moment`.",
                "Physical quantity: bending and twisting moment resultants per unit width in each shell element. Units: newton-meters per meter (N*m/m). Shape: three in-plane components per shell element. Location: element.",
                "Produced by structural static steps on models with shell elements. A run that did not produce the field makes `fea.field` fail with `RunMat:fea:OperationFailed`. It is a linear recovery quantity, so it can also be passed to `fea.ResponseSpectrumField.envelope`, `fea.RandomResponseField.rms`, and `fea.RandomResponseField.variance`.",
            ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
                "After `run = fea.run(study)`, call `fea.field(run, fea.StructuralField.shellBendingMoment())` (or pass `fea.results(run)` first), then read values with `fea.fieldPage` or draw them with `plot(field)`.",
            ] },
        ],
        examples: SHELL_BENDING_MOMENT_EXAMPLES,
        faqs: super::FAQS,
        links: super::LINKS,
        implementation: super::IMPLEMENTATION,
        verification: super::VERIFICATION,
        notes: &[],
    }),
    descriptor: &SHELL_BENDING_MOMENT_DESCRIPTOR,
    inference: selector_entry::infer_structural,
    integer_capabilities: &[],
    integer_audit: Some(&selector_entry::STRUCTURAL_SELECTOR_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
