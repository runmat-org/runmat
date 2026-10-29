use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition, ConstructorEntryDefinition},
    selector_entry::{self, NO_INPUTS, SELECTOR_OUTPUT},
};
use crate::*;

const CONSTRAINT_REACTION_FORCE_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "field = fea.StructuralField.constraintReactionForce()",
        inputs: &NO_INPUTS,
        outputs: &SELECTOR_OUTPUT,
    }];
pub const CONSTRAINT_REACTION_FORCE_DESCRIPTOR: BuiltinDescriptor =
    selector_entry::selector_descriptor(&CONSTRAINT_REACTION_FORCE_SIGNATURES);
const CONSTRAINT_REACTION_FORCE_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "select-constraint-reaction-force",
    title: "Select total reaction force per constraint",
    program: "selector = fea.StructuralField.constraintReactionForce();",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(selector, 'fea.StructuralField'));\nassert(strcmp(selector.value, \"constraint_reaction_force\"));",
    },
}];
pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.StructuralField.constraintReactionForce",
    documentation: catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.StructuralField.constraintReactionForce",
        summary: "Select the total reaction force per constraint result field for fea.field.",
        description: "`field = fea.StructuralField.constraintReactionForce()` returns a `fea.StructuralField` selector for `structural.constraint_reaction_force`: resultant reaction force of each authored constraint, summed over the nodes it constrains. Pass it to `fea.field` to read the field from a run.",
        keywords: &[
                "fea",
                "field",
                "structural",
                "total reaction force per constraint",
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
                "The selector is a small immutable value; it does no I/O. Its `value` property is `\"constraint_reaction_force\"`, and `fea.field` resolves it to the stored field id `structural.constraint_reaction_force`.",
                "Physical quantity: resultant reaction force of each authored constraint, summed over the nodes it constrains. Units: newtons (N). Shape: a three-component vector per constraint. Location: constraint.",
                "Produced by structural static steps (`fea.StructuralStep.static`), and harmonic structural steps, as a complex phasor at each frequency sample (select the sample with `fea.Frequency.hertz`). A run that did not produce the field makes `fea.field` fail with `RunMat:fea:OperationFailed`. Rows follow the model's constraint groups, so the value for a support named `support` is that support's total load path.",
            ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
                "After `run = fea.run(study)`, call `fea.field(run, fea.StructuralField.constraintReactionForce())` (or pass `fea.results(run)` first), then read values with `fea.fieldPage` or draw them with `plot(field)`.",
            ] },
        ],
        examples: CONSTRAINT_REACTION_FORCE_EXAMPLES,
        faqs: super::FAQS,
        links: super::LINKS,
        implementation: super::IMPLEMENTATION,
        verification: super::VERIFICATION,
        notes: &[],
    }),
    descriptor: &CONSTRAINT_REACTION_FORCE_DESCRIPTOR,
    inference: selector_entry::infer_structural,
    integer_capabilities: &[],
    integer_audit: Some(&selector_entry::STRUCTURAL_SELECTOR_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
