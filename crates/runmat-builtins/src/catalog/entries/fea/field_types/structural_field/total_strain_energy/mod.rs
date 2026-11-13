use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition, ConstructorEntryDefinition},
    selector_entry::{self, NO_INPUTS, SELECTOR_OUTPUT},
};
use crate::*;

const TOTAL_STRAIN_ENERGY_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "field = fea.StructuralField.totalStrainEnergy()",
        inputs: &NO_INPUTS,
        outputs: &SELECTOR_OUTPUT,
    }];
pub const TOTAL_STRAIN_ENERGY_DESCRIPTOR: BuiltinDescriptor =
    selector_entry::selector_descriptor(&TOTAL_STRAIN_ENERGY_SIGNATURES);
const TOTAL_STRAIN_ENERGY_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "select-total-strain-energy",
    title: "Select total strain energy",
    program: "selector = fea.StructuralField.totalStrainEnergy();",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(selector, 'fea.StructuralField'));\nassert(strcmp(selector.value, \"total_strain_energy\"));",
    },
}];
pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.StructuralField.totalStrainEnergy",
    documentation: catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.StructuralField.totalStrainEnergy",
        summary: "Select the total strain energy result field for fea.field.",
        description: "`field = fea.StructuralField.totalStrainEnergy()` returns a `fea.StructuralField` selector for `structural.total_strain_energy`: elastic strain energy stored in the whole model. Pass it to `fea.field` to read the field from a run.",
        keywords: &[
                "fea",
                "field",
                "structural",
                "total strain energy",
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
                "The selector is a small immutable value; it does no I/O. Its `value` property is `\"total_strain_energy\"`, and `fea.field` resolves it to the stored field id `structural.total_strain_energy`.",
                "Physical quantity: elastic strain energy stored in the whole model. Units: joules (J). Shape: a single scalar. Location: whole model.",
                "Produced by structural static steps (`fea.StructuralStep.static`). A run that did not produce the field makes `fea.field` fail with `RunMat:fea:OperationFailed`. For a linear static solve it equals half the work done by the applied loads, which makes it a quick global check.",
            ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
                "After `run = fea.run(study)`, call `fea.field(run, fea.StructuralField.totalStrainEnergy())` (or pass `fea.results(run)` first), then read values with `fea.fieldPage` or draw them with `plot(field)`.",
            ] },
        ],
        examples: TOTAL_STRAIN_ENERGY_EXAMPLES,
        faqs: super::FAQS,
        links: super::LINKS,
        implementation: super::IMPLEMENTATION,
        verification: super::VERIFICATION,
        notes: &[],
    }),
    descriptor: &TOTAL_STRAIN_ENERGY_DESCRIPTOR,
    inference: selector_entry::infer_structural,
    integer_capabilities: &[],
    integer_audit: Some(&selector_entry::STRUCTURAL_SELECTOR_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
