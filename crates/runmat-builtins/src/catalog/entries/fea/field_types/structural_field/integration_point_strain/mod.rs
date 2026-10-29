use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition, ConstructorEntryDefinition},
    selector_entry::{self, NO_INPUTS, SELECTOR_OUTPUT},
};
use crate::*;

const INTEGRATION_POINT_STRAIN_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "field = fea.StructuralField.integrationPointStrain()",
        inputs: &NO_INPUTS,
        outputs: &SELECTOR_OUTPUT,
    }];
pub const INTEGRATION_POINT_STRAIN_DESCRIPTOR: BuiltinDescriptor =
    selector_entry::selector_descriptor(&INTEGRATION_POINT_STRAIN_SIGNATURES);
const INTEGRATION_POINT_STRAIN_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "select-integration-point-strain",
    title: "Select integration-point strain tensor",
    program: "selector = fea.StructuralField.integrationPointStrain();",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(selector, 'fea.StructuralField'));\nassert(strcmp(selector.value, \"integration_point_strain\"));",
    },
}];
pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.StructuralField.integrationPointStrain",
    documentation: catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.StructuralField.integrationPointStrain",
        summary: "Select the integration-point strain tensor result field for fea.field.",
        description: "`field = fea.StructuralField.integrationPointStrain()` returns a `fea.StructuralField` selector for `structural.integration_point_strain`: small-strain tensor at every element integration point, before any averaging to elements or nodes. Pass it to `fea.field` to read the field from a run.",
        keywords: &[
                "fea",
                "field",
                "structural",
                "integration-point strain tensor",
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
                "The selector is a small immutable value; it does no I/O. Its `value` property is `\"integration_point_strain\"`, and `fea.field` resolves it to the stored field id `structural.integration_point_strain`.",
                "Physical quantity: small-strain tensor at every element integration point, before any averaging to elements or nodes. Units: dimensionless (m/m). Shape: a symmetric tensor stored as six independent components per integration point. Location: integration point.",
                "Produced by structural static steps (`fea.StructuralStep.static`). A run that did not produce the field makes `fea.field` fail with `RunMat:fea:OperationFailed`. Use it to inspect strain gradients inside higher-order elements.",
            ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
                "After `run = fea.run(study)`, call `fea.field(run, fea.StructuralField.integrationPointStrain())` (or pass `fea.results(run)` first), then read values with `fea.fieldPage` or draw them with `plot(field)`.",
            ] },
        ],
        examples: INTEGRATION_POINT_STRAIN_EXAMPLES,
        faqs: super::FAQS,
        links: super::LINKS,
        implementation: super::IMPLEMENTATION,
        verification: super::VERIFICATION,
        notes: &[],
    }),
    descriptor: &INTEGRATION_POINT_STRAIN_DESCRIPTOR,
    inference: selector_entry::infer_structural,
    integer_capabilities: &[],
    integer_audit: Some(&selector_entry::STRUCTURAL_SELECTOR_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
