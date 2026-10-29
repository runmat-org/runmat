use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition, ConstructorEntryDefinition},
    selector_entry::{self, NO_INPUTS, SELECTOR_OUTPUT},
};
use crate::*;

const BEAM_TORSION_MOMENT_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "field = fea.StructuralField.beamTorsionMoment()",
        inputs: &NO_INPUTS,
        outputs: &SELECTOR_OUTPUT,
    }];
pub const BEAM_TORSION_MOMENT_DESCRIPTOR: BuiltinDescriptor =
    selector_entry::selector_descriptor(&BEAM_TORSION_MOMENT_SIGNATURES);
const BEAM_TORSION_MOMENT_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "select-beam-torsion-moment",
    title: "Select beam torsion moment",
    program: "selector = fea.StructuralField.beamTorsionMoment();",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(selector, 'fea.StructuralField'));\nassert(strcmp(selector.value, \"beam_torsion_moment\"));",
    },
}];
pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.StructuralField.beamTorsionMoment",
    documentation: catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.StructuralField.beamTorsionMoment",
        summary: "Select the beam torsion moment result field for fea.field.",
        description: "`field = fea.StructuralField.beamTorsionMoment()` returns a `fea.StructuralField` selector for `structural.beam_torsion_moment`: twisting moment about the axis of each beam element. Pass it to `fea.field` to read the field from a run.",
        keywords: &[
                "fea",
                "field",
                "structural",
                "beam torsion moment",
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
                "The selector is a small immutable value; it does no I/O. Its `value` property is `\"beam_torsion_moment\"`, and `fea.field` resolves it to the stored field id `structural.beam_torsion_moment`.",
                "Physical quantity: twisting moment about the axis of each beam element. Units: newton-meters (N*m). Shape: a scalar per beam element. Location: element.",
                "Produced by structural static steps on models with beam elements. A run that did not produce the field makes `fea.field` fail with `RunMat:fea:OperationFailed`. Pair it with `fea.StructuralField.beamTorsionStress()` for the resulting shear stress.",
            ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
                "After `run = fea.run(study)`, call `fea.field(run, fea.StructuralField.beamTorsionMoment())` (or pass `fea.results(run)` first), then read values with `fea.fieldPage` or draw them with `plot(field)`.",
            ] },
        ],
        examples: BEAM_TORSION_MOMENT_EXAMPLES,
        faqs: super::FAQS,
        links: super::LINKS,
        implementation: super::IMPLEMENTATION,
        verification: super::VERIFICATION,
        notes: &[],
    }),
    descriptor: &BEAM_TORSION_MOMENT_DESCRIPTOR,
    inference: selector_entry::infer_structural,
    integer_capabilities: &[],
    integer_audit: Some(&selector_entry::STRUCTURAL_SELECTOR_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
