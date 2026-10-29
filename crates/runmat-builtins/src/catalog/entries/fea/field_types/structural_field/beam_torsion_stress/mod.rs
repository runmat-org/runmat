use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition, ConstructorEntryDefinition},
    selector_entry::{self, NO_INPUTS, SELECTOR_OUTPUT},
};
use crate::*;

const BEAM_TORSION_STRESS_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "field = fea.StructuralField.beamTorsionStress()",
        inputs: &NO_INPUTS,
        outputs: &SELECTOR_OUTPUT,
    }];
pub const BEAM_TORSION_STRESS_DESCRIPTOR: BuiltinDescriptor =
    selector_entry::selector_descriptor(&BEAM_TORSION_STRESS_SIGNATURES);
const BEAM_TORSION_STRESS_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "select-beam-torsion-stress",
    title: "Select beam torsion stress",
    program: "selector = fea.StructuralField.beamTorsionStress();",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(selector, 'fea.StructuralField'));\nassert(strcmp(selector.value, \"beam_torsion_stress\"));",
    },
}];
pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.StructuralField.beamTorsionStress",
    documentation: catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.StructuralField.beamTorsionStress",
        summary: "Select the beam torsion stress result field for fea.field.",
        description: "`field = fea.StructuralField.beamTorsionStress()` returns a `fea.StructuralField` selector for `structural.beam_torsion_stress`: shear stress caused by torsion in each beam element. Pass it to `fea.field` to read the field from a run.",
        keywords: &[
                "fea",
                "field",
                "structural",
                "beam torsion stress",
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
                "The selector is a small immutable value; it does no I/O. Its `value` property is `\"beam_torsion_stress\"`, and `fea.field` resolves it to the stored field id `structural.beam_torsion_stress`.",
                "Physical quantity: shear stress caused by torsion in each beam element. Units: pascals (Pa). Shape: a scalar per beam element. Location: element.",
                "Produced by structural static steps on models with beam elements. A run that did not produce the field makes `fea.field` fail with `RunMat:fea:OperationFailed`. Pair it with `fea.StructuralField.beamTorsionMoment()` to see the moment behind it.",
            ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
                "After `run = fea.run(study)`, call `fea.field(run, fea.StructuralField.beamTorsionStress())` (or pass `fea.results(run)` first), then read values with `fea.fieldPage` or draw them with `plot(field)`.",
            ] },
        ],
        examples: BEAM_TORSION_STRESS_EXAMPLES,
        faqs: super::FAQS,
        links: super::LINKS,
        implementation: super::IMPLEMENTATION,
        verification: super::VERIFICATION,
        notes: &[],
    }),
    descriptor: &BEAM_TORSION_STRESS_DESCRIPTOR,
    inference: selector_entry::infer_structural,
    integer_capabilities: &[],
    integer_audit: Some(&selector_entry::STRUCTURAL_SELECTOR_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
