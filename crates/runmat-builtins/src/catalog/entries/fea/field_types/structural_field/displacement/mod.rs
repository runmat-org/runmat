use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition, ConstructorEntryDefinition},
    selector_entry::{self, NO_INPUTS, SELECTOR_OUTPUT},
};
use crate::*;

const DISPLACEMENT_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "field = fea.StructuralField.displacement()",
    inputs: &NO_INPUTS,
    outputs: &SELECTOR_OUTPUT,
}];
pub const DISPLACEMENT_DESCRIPTOR: BuiltinDescriptor =
    selector_entry::selector_descriptor(&DISPLACEMENT_SIGNATURES);
const DISPLACEMENT_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "select-displacement",
    title: "Select nodal displacement",
    program: "selector = fea.StructuralField.displacement();",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(selector, 'fea.StructuralField'));\nassert(strcmp(selector.value, \"displacement\"));",
    },
}];
pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.StructuralField.displacement",
    documentation: catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.StructuralField.displacement",
        summary: "Select the nodal displacement result field for fea.field.",
        description: "`field = fea.StructuralField.displacement()` returns a `fea.StructuralField` selector for `structural.displacement`: translational displacement of every mesh node in the global frame. Pass it to `fea.field` to read the field from a run.",
        keywords: &[
                "fea",
                "field",
                "structural",
                "nodal displacement",
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
                "The selector is a small immutable value; it does no I/O. Its `value` property is `\"displacement\"`, and `fea.field` resolves it to the stored field id `structural.displacement`.",
                "Physical quantity: translational displacement of every mesh node in the global frame. Units: meters (m). Shape: a three-component vector (x, y, z) per node. Location: node.",
                "Produced by structural static steps (`fea.StructuralStep.static`), nonlinear structural steps, and harmonic structural steps, as a complex phasor at each frequency sample (select the sample with `fea.Frequency.hertz`). A run that did not produce the field makes `fea.field` fail with `RunMat:fea:OperationFailed`. It is a linear recovery quantity, so it can also be passed to `fea.ResponseSpectrumField.envelope`, `fea.RandomResponseField.rms`, and `fea.RandomResponseField.variance`.",
            ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
                "After `run = fea.run(study)`, call `fea.field(run, fea.StructuralField.displacement())` (or pass `fea.results(run)` first), then read values with `fea.fieldPage` or draw them with `plot(field)`.",
            ] },
        ],
        examples: DISPLACEMENT_EXAMPLES,
        faqs: super::FAQS,
        links: super::LINKS,
        implementation: super::IMPLEMENTATION,
        verification: super::VERIFICATION,
        notes: &[],
    }),
    descriptor: &DISPLACEMENT_DESCRIPTOR,
    inference: selector_entry::infer_structural,
    integer_capabilities: &[],
    integer_audit: Some(&selector_entry::STRUCTURAL_SELECTOR_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
