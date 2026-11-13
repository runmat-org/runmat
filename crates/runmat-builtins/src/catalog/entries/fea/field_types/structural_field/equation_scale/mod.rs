use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition, ConstructorEntryDefinition},
    selector_entry::{self, NO_INPUTS, SELECTOR_OUTPUT},
};
use crate::*;

const EQUATION_SCALE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "field = fea.StructuralField.equationScale()",
    inputs: &NO_INPUTS,
    outputs: &SELECTOR_OUTPUT,
}];
pub const EQUATION_SCALE_DESCRIPTOR: BuiltinDescriptor =
    selector_entry::selector_descriptor(&EQUATION_SCALE_SIGNATURES);
const EQUATION_SCALE_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "select-equation-scale",
    title: "Select equation scale",
    program: "selector = fea.StructuralField.equationScale();",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(selector, 'fea.StructuralField'));\nassert(strcmp(selector.value, \"equation_scale\"));",
    },
}];
pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.StructuralField.equationScale",
    documentation: catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.StructuralField.equationScale",
        summary: "Select the equation scale result field for fea.field.",
        description: "`field = fea.StructuralField.equationScale()` returns a `fea.StructuralField` selector for `structural.equation_scale`: scale factor used to normalize the equilibrium residual, so that `residualNorm` is comparable across models of different size and stiffness. Pass it to `fea.field` to read the field from a run.",
        keywords: &[
                "fea",
                "field",
                "structural",
                "equation scale",
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
                "The selector is a small immutable value; it does no I/O. Its `value` property is `\"equation_scale\"`, and `fea.field` resolves it to the stored field id `structural.equation_scale`.",
                "Physical quantity: scale factor used to normalize the equilibrium residual, so that `residualNorm` is comparable across models of different size and stiffness. Units: dimensionless. Shape: a single scalar. Location: whole model.",
                "Produced by structural static steps (`fea.StructuralStep.static`). A run that did not produce the field makes `fea.field` fail with `RunMat:fea:OperationFailed`. Read it together with `fea.StructuralField.residualNorm()` when auditing solver accuracy.",
            ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
                "After `run = fea.run(study)`, call `fea.field(run, fea.StructuralField.equationScale())` (or pass `fea.results(run)` first), then read values with `fea.fieldPage` or draw them with `plot(field)`.",
            ] },
        ],
        examples: EQUATION_SCALE_EXAMPLES,
        faqs: super::FAQS,
        links: super::LINKS,
        implementation: super::IMPLEMENTATION,
        verification: super::VERIFICATION,
        notes: &[],
    }),
    descriptor: &EQUATION_SCALE_DESCRIPTOR,
    inference: selector_entry::infer_structural,
    integer_capabilities: &[],
    integer_audit: Some(&selector_entry::STRUCTURAL_SELECTOR_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
