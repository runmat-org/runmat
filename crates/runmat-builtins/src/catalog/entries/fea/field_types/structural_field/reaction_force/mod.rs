use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition, ConstructorEntryDefinition},
    selector_entry::{self, NO_INPUTS, SELECTOR_OUTPUT},
};
use crate::*;

const REACTION_FORCE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "field = fea.StructuralField.reactionForce()",
    inputs: &NO_INPUTS,
    outputs: &SELECTOR_OUTPUT,
}];
pub const REACTION_FORCE_DESCRIPTOR: BuiltinDescriptor =
    selector_entry::selector_descriptor(&REACTION_FORCE_SIGNATURES);
const REACTION_FORCE_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "select-reaction-force",
    title: "Select nodal reaction force",
    program: "selector = fea.StructuralField.reactionForce();",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(selector, 'fea.StructuralField'));\nassert(strcmp(selector.value, \"reaction_force\"));",
    },
}];
pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.StructuralField.reactionForce",
    documentation: catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.StructuralField.reactionForce",
        summary: "Select the nodal reaction force result field for fea.field.",
        description: "`field = fea.StructuralField.reactionForce()` returns a `fea.StructuralField` selector for `structural.reaction_force`: force that the supports exert on each node, recovered from the assembled equilibrium. Pass it to `fea.field` to read the field from a run.",
        keywords: &[
                "fea",
                "field",
                "structural",
                "nodal reaction force",
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
                "The selector is a small immutable value; it does no I/O. Its `value` property is `\"reaction_force\"`, and `fea.field` resolves it to the stored field id `structural.reaction_force`.",
                "Physical quantity: force that the supports exert on each node, recovered from the assembled equilibrium. Units: newtons (N). Shape: a three-component vector per node (zero away from constrained nodes). Location: node.",
                "Produced by structural static steps (`fea.StructuralStep.static`), nonlinear structural steps, transient structural steps, and harmonic structural steps, as a complex phasor at each frequency sample (select the sample with `fea.Frequency.hertz`). A run that did not produce the field makes `fea.field` fail with `RunMat:fea:OperationFailed`. Sum the rows over a support to get its total reaction, or use `fea.StructuralField.constraintReactionForce()`.",
            ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
                "After `run = fea.run(study)`, call `fea.field(run, fea.StructuralField.reactionForce())` (or pass `fea.results(run)` first), then read values with `fea.fieldPage` or draw them with `plot(field)`.",
            ] },
        ],
        examples: REACTION_FORCE_EXAMPLES,
        faqs: super::FAQS,
        links: super::LINKS,
        implementation: super::IMPLEMENTATION,
        verification: super::VERIFICATION,
        notes: &[],
    }),
    descriptor: &REACTION_FORCE_DESCRIPTOR,
    inference: selector_entry::infer_structural,
    integer_capabilities: &[],
    integer_audit: Some(&selector_entry::STRUCTURAL_SELECTOR_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
