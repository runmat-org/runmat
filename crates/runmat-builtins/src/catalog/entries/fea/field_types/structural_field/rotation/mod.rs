use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition, ConstructorEntryDefinition},
    selector_entry::{self, NO_INPUTS, SELECTOR_OUTPUT},
};
use crate::*;

const ROTATION_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "field = fea.StructuralField.rotation()",
    inputs: &NO_INPUTS,
    outputs: &SELECTOR_OUTPUT,
}];
pub const ROTATION_DESCRIPTOR: BuiltinDescriptor =
    selector_entry::selector_descriptor(&ROTATION_SIGNATURES);
const ROTATION_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "select-rotation",
    title: "Select nodal rotation",
    program: "selector = fea.StructuralField.rotation();",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(selector, 'fea.StructuralField'));\nassert(strcmp(selector.value, \"rotation\"));",
    },
}];
pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.StructuralField.rotation",
    documentation: catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.StructuralField.rotation",
        summary: "Select the nodal rotation result field for fea.field.",
        description: "`field = fea.StructuralField.rotation()` returns a `fea.StructuralField` selector for `structural.rotation`: rotational degrees of freedom of every node that carries them, in the global frame. Pass it to `fea.field` to read the field from a run.",
        keywords: &[
                "fea",
                "field",
                "structural",
                "nodal rotation",
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
                "The selector is a small immutable value; it does no I/O. Its `value` property is `\"rotation\"`, and `fea.field` resolves it to the stored field id `structural.rotation`.",
                "Physical quantity: rotational degrees of freedom of every node that carries them, in the global frame. Units: radians (rad). Shape: a three-component vector (about x, y, z) per node. Location: node.",
                "Produced by structural static steps on models with rotational degrees of freedom (beams and shells), and harmonic structural steps, as a complex phasor at each frequency sample (select the sample with `fea.Frequency.hertz`). A run that did not produce the field makes `fea.field` fail with `RunMat:fea:OperationFailed`. It is a linear recovery quantity, so it can also be passed to `fea.ResponseSpectrumField.envelope`, `fea.RandomResponseField.rms`, and `fea.RandomResponseField.variance`. On solid-only models the field has no entries.",
            ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
                "After `run = fea.run(study)`, call `fea.field(run, fea.StructuralField.rotation())` (or pass `fea.results(run)` first), then read values with `fea.fieldPage` or draw them with `plot(field)`.",
            ] },
        ],
        examples: ROTATION_EXAMPLES,
        faqs: super::FAQS,
        links: super::LINKS,
        implementation: super::IMPLEMENTATION,
        verification: super::VERIFICATION,
        notes: &[],
    }),
    descriptor: &ROTATION_DESCRIPTOR,
    inference: selector_entry::infer_structural,
    integer_capabilities: &[],
    integer_audit: Some(&selector_entry::STRUCTURAL_SELECTOR_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
