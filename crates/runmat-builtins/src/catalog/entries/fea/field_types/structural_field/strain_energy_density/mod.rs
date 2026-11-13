use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition, ConstructorEntryDefinition},
    selector_entry::{self, NO_INPUTS, SELECTOR_OUTPUT},
};
use crate::*;

const STRAIN_ENERGY_DENSITY_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "field = fea.StructuralField.strainEnergyDensity()",
        inputs: &NO_INPUTS,
        outputs: &SELECTOR_OUTPUT,
    }];
pub const STRAIN_ENERGY_DENSITY_DESCRIPTOR: BuiltinDescriptor =
    selector_entry::selector_descriptor(&STRAIN_ENERGY_DENSITY_SIGNATURES);
const STRAIN_ENERGY_DENSITY_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "select-strain-energy-density",
    title: "Select element strain energy density",
    program: "selector = fea.StructuralField.strainEnergyDensity();",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(selector, 'fea.StructuralField'));\nassert(strcmp(selector.value, \"strain_energy_density\"));",
    },
}];
pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.StructuralField.strainEnergyDensity",
    documentation: catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.StructuralField.strainEnergyDensity",
        summary: "Select the element strain energy density result field for fea.field.",
        description: "`field = fea.StructuralField.strainEnergyDensity()` returns a `fea.StructuralField` selector for `structural.strain_energy_density`: elastic strain energy stored per unit volume in each element. Pass it to `fea.field` to read the field from a run.",
        keywords: &[
                "fea",
                "field",
                "structural",
                "element strain energy density",
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
                "The selector is a small immutable value; it does no I/O. Its `value` property is `\"strain_energy_density\"`, and `fea.field` resolves it to the stored field id `structural.strain_energy_density`.",
                "Physical quantity: elastic strain energy stored per unit volume in each element. Units: joules per cubic meter (J/m^3). Shape: a scalar per element. Location: element.",
                "Produced by structural static steps (`fea.StructuralStep.static`). A run that did not produce the field makes `fea.field` fail with `RunMat:fea:OperationFailed`. High values mark where the structure stores the most energy, which is useful for stiffness optimization.",
            ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
                "After `run = fea.run(study)`, call `fea.field(run, fea.StructuralField.strainEnergyDensity())` (or pass `fea.results(run)` first), then read values with `fea.fieldPage` or draw them with `plot(field)`.",
            ] },
        ],
        examples: STRAIN_ENERGY_DENSITY_EXAMPLES,
        faqs: super::FAQS,
        links: super::LINKS,
        implementation: super::IMPLEMENTATION,
        verification: super::VERIFICATION,
        notes: &[],
    }),
    descriptor: &STRAIN_ENERGY_DENSITY_DESCRIPTOR,
    inference: selector_entry::infer_structural,
    integer_capabilities: &[],
    integer_audit: Some(&selector_entry::STRUCTURAL_SELECTOR_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
