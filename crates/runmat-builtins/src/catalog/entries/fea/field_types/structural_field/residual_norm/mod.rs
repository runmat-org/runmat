use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition, ConstructorEntryDefinition},
    selector_entry::{self, NO_INPUTS, SELECTOR_OUTPUT},
};
use crate::*;

const RESIDUAL_NORM_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "field = fea.StructuralField.residualNorm()",
    inputs: &NO_INPUTS,
    outputs: &SELECTOR_OUTPUT,
}];
pub const RESIDUAL_NORM_DESCRIPTOR: BuiltinDescriptor =
    selector_entry::selector_descriptor(&RESIDUAL_NORM_SIGNATURES);
const RESIDUAL_NORM_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "select-residual-norm",
    title: "Select equilibrium residual norm",
    program: "selector = fea.StructuralField.residualNorm();",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(selector, 'fea.StructuralField'));\nassert(strcmp(selector.value, \"residual_norm\"));",
    },
}];
pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.StructuralField.residualNorm",
    documentation: catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.StructuralField.residualNorm",
        summary: "Select the equilibrium residual norm result field for fea.field.",
        description: "`field = fea.StructuralField.residualNorm()` returns a `fea.StructuralField` selector for `structural.residual_norm`: normalized norm of the equilibrium residual after the linear solve, a measure of how exactly the discrete equations were satisfied. Pass it to `fea.field` to read the field from a run.",
        keywords: &[
                "fea",
                "field",
                "structural",
                "equilibrium residual norm",
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
                "The selector is a small immutable value; it does no I/O. Its `value` property is `\"residual_norm\"`, and `fea.field` resolves it to the stored field id `structural.residual_norm`.",
                "Physical quantity: normalized norm of the equilibrium residual after the linear solve, a measure of how exactly the discrete equations were satisfied. Units: dimensionless. Shape: a single scalar. Location: whole model.",
                "Produced by structural static steps (`fea.StructuralStep.static`). A run that did not produce the field makes `fea.field` fail with `RunMat:fea:OperationFailed`. Small values indicate that the discrete equilibrium equations were satisfied closely.",
            ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
                "After `run = fea.run(study)`, call `fea.field(run, fea.StructuralField.residualNorm())` (or pass `fea.results(run)` first), then read values with `fea.fieldPage` or draw them with `plot(field)`.",
            ] },
        ],
        examples: RESIDUAL_NORM_EXAMPLES,
        faqs: super::FAQS,
        links: super::LINKS,
        implementation: super::IMPLEMENTATION,
        verification: super::VERIFICATION,
        notes: &[],
    }),
    descriptor: &RESIDUAL_NORM_DESCRIPTOR,
    inference: selector_entry::infer_structural,
    integer_capabilities: &[],
    integer_audit: Some(&selector_entry::STRUCTURAL_SELECTOR_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
