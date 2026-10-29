use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition, ConstructorEntryDefinition},
    selector_entry::{self, NO_INPUTS, SELECTOR_OUTPUT},
};
use crate::*;

const REACTION_MOMENT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "field = fea.StructuralField.reactionMoment()",
    inputs: &NO_INPUTS,
    outputs: &SELECTOR_OUTPUT,
}];
pub const REACTION_MOMENT_DESCRIPTOR: BuiltinDescriptor =
    selector_entry::selector_descriptor(&REACTION_MOMENT_SIGNATURES);
const REACTION_MOMENT_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "select-reaction-moment",
    title: "Select nodal reaction moment",
    program: "selector = fea.StructuralField.reactionMoment();",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(selector, 'fea.StructuralField'));\nassert(strcmp(selector.value, \"reaction_moment\"));",
    },
}];
pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.StructuralField.reactionMoment",
    documentation: catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.StructuralField.reactionMoment",
        summary: "Select the nodal reaction moment result field for fea.field.",
        description: "`field = fea.StructuralField.reactionMoment()` returns a `fea.StructuralField` selector for `structural.reaction_moment`: moment that the supports exert on each node with rotational degrees of freedom. Pass it to `fea.field` to read the field from a run.",
        keywords: &[
                "fea",
                "field",
                "structural",
                "nodal reaction moment",
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
                "The selector is a small immutable value; it does no I/O. Its `value` property is `\"reaction_moment\"`, and `fea.field` resolves it to the stored field id `structural.reaction_moment`.",
                "Physical quantity: moment that the supports exert on each node with rotational degrees of freedom. Units: newton-meters (N*m). Shape: a three-component vector per node. Location: node.",
                "Produced by structural static and transient steps on models with rotational degrees of freedom, and harmonic structural steps, as a complex phasor at each frequency sample (select the sample with `fea.Frequency.hertz`). A run that did not produce the field makes `fea.field` fail with `RunMat:fea:OperationFailed`. Solid-only models have no rotational degrees of freedom, so the field is empty for them.",
            ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
                "After `run = fea.run(study)`, call `fea.field(run, fea.StructuralField.reactionMoment())` (or pass `fea.results(run)` first), then read values with `fea.fieldPage` or draw them with `plot(field)`.",
            ] },
        ],
        examples: REACTION_MOMENT_EXAMPLES,
        faqs: super::FAQS,
        links: super::LINKS,
        implementation: super::IMPLEMENTATION,
        verification: super::VERIFICATION,
        notes: &[],
    }),
    descriptor: &REACTION_MOMENT_DESCRIPTOR,
    inference: selector_entry::infer_structural,
    integer_capabilities: &[],
    integer_audit: Some(&selector_entry::STRUCTURAL_SELECTOR_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
