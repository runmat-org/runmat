use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition, ConstructorEntryDefinition},
    selector_entry::{self, OUTPUT_NAME_INPUT, SELECTOR_OUTPUT},
};
use crate::*;

const NAMED_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "field = fea.OutputField.named(name)",
    inputs: &OUTPUT_NAME_INPUT,
    outputs: &SELECTOR_OUTPUT,
}];
pub const NAMED_DESCRIPTOR: BuiltinDescriptor =
    selector_entry::selector_descriptor(&NAMED_SIGNATURES);
const NAMED_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "named-output",
    title: "Select a declared output by name",
    program: "selector = fea.OutputField.named(\"von_mises\");",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(selector, 'fea.OutputField'));\nassert(strcmp(selector.value, \"von_mises\"));",
    },
}];
pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.OutputField.named",
    documentation: catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.OutputField.named",
        summary: "Select an output that a .fea study declares by name.",
        description: "`field = fea.OutputField.named(name)` returns a `fea.OutputField` selector for the output declared as `output <name> = ...` in a `.fea` study, for example `fea.OutputField.named(\"von_mises\")`.",
        keywords: &[
                "fea",
                "output",
                "field",
                "named output",
                "post-processing",
            ],
        related: &[
                "fea.field",
                "fea.results",
                "fea.StructuralField.vonMises",
            ],
        sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
                "A `.fea` study can name the results it publishes, for example `output von_mises = analysis.static.field.element.von_mises_stress`. This selector refers to such an output by its name, independent of the underlying field id.",
                "The name must be a valid FEA identifier: a letter or underscore followed by letters, digits, or underscores, at most 512 characters. Any other text is rejected with `RunMat:fea:InvalidInput`.",
                "The selector is an immutable value with the name in its `value` property. Whether the output exists is checked when `fea.field` reads it from a run; an undeclared name makes that call fail with `RunMat:fea:OperationFailed`.",
            ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
                "Use it with runs of `.fea` studies that declare outputs, for example `fea.field(results, fea.OutputField.named(\"displacement\"))`. Studies built in RunMat code have no named outputs; select their fields with `fea.StructuralField.*` instead.",
            ] },
        BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
                "`RunMat:fea:InvalidInput` reports a missing or non-text name or a name that is not a valid FEA identifier.",
            ] },
        ],
        examples: NAMED_EXAMPLES,
        faqs: super::OUTPUT_FAQS,
        links: super::LINKS,
        implementation: super::IMPLEMENTATION,
        verification: super::VERIFICATION,
        notes: &[],
    }),
    descriptor: &NAMED_DESCRIPTOR,
    inference: selector_entry::infer_output_named,
    integer_capabilities: &[],
    integer_audit: Some(&selector_entry::STRUCTURAL_SELECTOR_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
