use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition, ConstructorEntryDefinition},
    selector_entry::{self, LINEAR_QUANTITY_INPUT, SELECTOR_OUTPUT},
};
use crate::*;

const VARIANCE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "field = fea.RandomResponseField.variance(quantity)",
    inputs: &LINEAR_QUANTITY_INPUT,
    outputs: &SELECTOR_OUTPUT,
}];
pub const VARIANCE_DESCRIPTOR: BuiltinDescriptor =
    selector_entry::selector_descriptor(&VARIANCE_SIGNATURES);
const VARIANCE_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "variance-displacement",
    title: "Select the variance of a structural quantity",
    program: "selector = fea.RandomResponseField.variance(fea.StructuralField.displacement());",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(selector, 'fea.RandomResponseField'));",
    },
}];
pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.RandomResponseField.variance",
    documentation: catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.RandomResponseField.variance",
        summary: "Select the random-vibration variance of a structural field such as stress or displacement.",
        description: "`field = fea.RandomResponseField.variance(quantity)` returns a `fea.RandomResponseField` selector for the variance response of a linear structural quantity under random (PSD) excitation, for example `fea.RandomResponseField.variance(fea.StructuralField.displacement())`.",
        keywords: &[
                "fea",
                "random vibration",
                "PSD",
                "variance",
                "stress",
                "displacement",
            ],
        related: &[
                "fea.RandomResponseField.rms",
                "fea.RandomResponseField.variance",
                "fea.StructuralField.stress",
                "fea.field",
            ],
        sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
                "The result selects field id `structural.random_response.variance.<quantity>`, for example `structural.random_response.variance.stress`. Shape and location are those of the wrapped quantity; units are the square of the wrapped quantity's units (for stress, Pa^2).",
                "The quantity must be one of `fea.StructuralField.displacement()`, `.rotation()`, `.strain()`, `.stress()`, `.shellMembraneForce()`, `.shellBendingMoment()`, or `.shellTransverseShear()`; derived quantities such as von Mises stress or strain energy are rejected because they are not linear in the response.",
                "The variance is computed per component from the modal transfer of the excitation spectrum, before any nonlinear transform. It is a statistic of each component, not a simultaneous state; equivalent stresses must not be formed from these components.",
                "Produced by modal random-response steps that follow a modal extraction in the study's sequence.",
            ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
                "Pass the selector to `fea.field` with the random-response step result, for example `fea.field(run.step(\"random\"), selector)`, then page or plot the field.",
            ] },
        BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
                "`RunMat:fea:InvalidInput` reports a missing argument, an argument that is not a `fea.StructuralField`, or a structural quantity that is not linear in the response.",
            ] },
        ],
        examples: VARIANCE_EXAMPLES,
        faqs: super::RR_FAQS,
        links: super::LINKS,
        implementation: super::IMPLEMENTATION,
        verification: super::VERIFICATION,
        notes: &[],
    }),
    descriptor: &VARIANCE_DESCRIPTOR,
    inference: selector_entry::infer_random_response,
    integer_capabilities: &[],
    integer_audit: Some(&selector_entry::STATISTICAL_SELECTOR_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
