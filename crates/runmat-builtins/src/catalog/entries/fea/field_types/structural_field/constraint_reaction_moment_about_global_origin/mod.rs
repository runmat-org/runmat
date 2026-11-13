use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition, ConstructorEntryDefinition},
    selector_entry::{self, NO_INPUTS, SELECTOR_OUTPUT},
};
use crate::*;

const CONSTRAINT_REACTION_MOMENT_ABOUT_GLOBAL_ORIGIN_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "field = fea.StructuralField.constraintReactionMomentAboutGlobalOrigin()",
        inputs: &NO_INPUTS,
        outputs: &SELECTOR_OUTPUT,
    }];
pub const CONSTRAINT_REACTION_MOMENT_ABOUT_GLOBAL_ORIGIN_DESCRIPTOR: BuiltinDescriptor =
    selector_entry::selector_descriptor(&CONSTRAINT_REACTION_MOMENT_ABOUT_GLOBAL_ORIGIN_SIGNATURES);
const CONSTRAINT_REACTION_MOMENT_ABOUT_GLOBAL_ORIGIN_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "select-constraint-reaction-moment-about-global-origin",
    title: "Select total reaction moment per constraint",
    program: "selector = fea.StructuralField.constraintReactionMomentAboutGlobalOrigin();",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(selector, 'fea.StructuralField'));\nassert(strcmp(selector.value, \"constraint_reaction_moment_about_global_origin\"));",
    },
}];
pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.StructuralField.constraintReactionMomentAboutGlobalOrigin",
    documentation: catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.StructuralField.constraintReactionMomentAboutGlobalOrigin",
        summary: "Select the total reaction moment per constraint result field for fea.field.",
        description: "`field = fea.StructuralField.constraintReactionMomentAboutGlobalOrigin()` returns a `fea.StructuralField` selector for `structural.constraint_reaction_moment_about_global_origin`: resultant reaction moment of each authored constraint, taken about the global origin. Pass it to `fea.field` to read the field from a run.",
        keywords: &[
                "fea",
                "field",
                "structural",
                "total reaction moment per constraint",
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
                "The selector is a small immutable value; it does no I/O. Its `value` property is `\"constraint_reaction_moment_about_global_origin\"`, and `fea.field` resolves it to the stored field id `structural.constraint_reaction_moment_about_global_origin`.",
                "Physical quantity: resultant reaction moment of each authored constraint, taken about the global origin. Units: newton-meters (N*m). Shape: a three-component vector per constraint. Location: constraint.",
                "Produced by structural static steps (`fea.StructuralStep.static`), and harmonic structural steps, as a complex phasor at each frequency sample (select the sample with `fea.Frequency.hertz`). A run that did not produce the field makes `fea.field` fail with `RunMat:fea:OperationFailed`. The moment is about the global origin, not the constraint's centroid; shift it with the reaction force if you need another reference point.",
            ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
                "After `run = fea.run(study)`, call `fea.field(run, fea.StructuralField.constraintReactionMomentAboutGlobalOrigin())` (or pass `fea.results(run)` first), then read values with `fea.fieldPage` or draw them with `plot(field)`.",
            ] },
        ],
        examples: CONSTRAINT_REACTION_MOMENT_ABOUT_GLOBAL_ORIGIN_EXAMPLES,
        faqs: super::FAQS,
        links: super::LINKS,
        implementation: super::IMPLEMENTATION,
        verification: super::VERIFICATION,
        notes: &[],
    }),
    descriptor: &CONSTRAINT_REACTION_MOMENT_ABOUT_GLOBAL_ORIGIN_DESCRIPTOR,
    inference: selector_entry::infer_structural,
    integer_capabilities: &[],
    integer_audit: Some(&selector_entry::STRUCTURAL_SELECTOR_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
