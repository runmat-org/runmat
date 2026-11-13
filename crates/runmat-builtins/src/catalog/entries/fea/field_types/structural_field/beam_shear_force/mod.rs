use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition, ConstructorEntryDefinition},
    selector_entry::{self, NO_INPUTS, SELECTOR_OUTPUT},
};
use crate::*;

const BEAM_SHEAR_FORCE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "field = fea.StructuralField.beamShearForce()",
    inputs: &NO_INPUTS,
    outputs: &SELECTOR_OUTPUT,
}];
pub const BEAM_SHEAR_FORCE_DESCRIPTOR: BuiltinDescriptor =
    selector_entry::selector_descriptor(&BEAM_SHEAR_FORCE_SIGNATURES);
const BEAM_SHEAR_FORCE_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "select-beam-shear-force",
    title: "Select beam shear force",
    program: "selector = fea.StructuralField.beamShearForce();",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(selector, 'fea.StructuralField'));\nassert(strcmp(selector.value, \"beam_shear_force\"));",
    },
}];
pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.StructuralField.beamShearForce",
    documentation: catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.StructuralField.beamShearForce",
        summary: "Select the beam shear force result field for fea.field.",
        description: "`field = fea.StructuralField.beamShearForce()` returns a `fea.StructuralField` selector for `structural.beam_shear_force`: transverse section shear forces in each beam element, along the section's two local axes. Pass it to `fea.field` to read the field from a run.",
        keywords: &[
                "fea",
                "field",
                "structural",
                "beam shear force",
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
                "The selector is a small immutable value; it does no I/O. Its `value` property is `\"beam_shear_force\"`, and `fea.field` resolves it to the stored field id `structural.beam_shear_force`.",
                "Physical quantity: transverse section shear forces in each beam element, along the section's two local axes. Units: newtons (N). Shape: two components per beam element. Location: element.",
                "Produced by structural static steps on models with beam elements. A run that did not produce the field makes `fea.field` fail with `RunMat:fea:OperationFailed`. Components follow the beam section's local axes, not the global frame.",
            ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
                "After `run = fea.run(study)`, call `fea.field(run, fea.StructuralField.beamShearForce())` (or pass `fea.results(run)` first), then read values with `fea.fieldPage` or draw them with `plot(field)`.",
            ] },
        ],
        examples: BEAM_SHEAR_FORCE_EXAMPLES,
        faqs: super::FAQS,
        links: super::LINKS,
        implementation: super::IMPLEMENTATION,
        verification: super::VERIFICATION,
        notes: &[],
    }),
    descriptor: &BEAM_SHEAR_FORCE_DESCRIPTOR,
    inference: selector_entry::infer_structural,
    integer_capabilities: &[],
    integer_audit: Some(&selector_entry::STRUCTURAL_SELECTOR_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
