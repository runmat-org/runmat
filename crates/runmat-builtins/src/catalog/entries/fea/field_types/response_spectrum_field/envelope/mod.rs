use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition, ConstructorEntryDefinition},
    selector_entry::{self, LINEAR_QUANTITY_INPUT, SELECTOR_OUTPUT},
};
use crate::*;

const ENVELOPE_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "field = fea.ResponseSpectrumField.envelope(quantity)",
    inputs: &LINEAR_QUANTITY_INPUT,
    outputs: &SELECTOR_OUTPUT,
}];
pub const ENVELOPE_DESCRIPTOR: BuiltinDescriptor =
    selector_entry::selector_descriptor(&ENVELOPE_SIGNATURES);
const ENVELOPE_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "stress-envelope",
    title: "Select the stress envelope",
    program: "selector = fea.ResponseSpectrumField.envelope(fea.StructuralField.stress());",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isa(selector, 'fea.ResponseSpectrumField'));",
    },
}];
pub(super) const ENTRY: BuiltinCatalogEntry = catalog_entry::entry(ConstructorEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "fea.ResponseSpectrumField.envelope",
    documentation: catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.ResponseSpectrumField.envelope",
        summary: "Select the response-spectrum peak envelope of a structural field such as stress or displacement.",
        description: "`field = fea.ResponseSpectrumField.envelope(quantity)` returns a `fea.ResponseSpectrumField` selector for the modally combined peak of a linear structural quantity, for example `fea.ResponseSpectrumField.envelope(fea.StructuralField.stress())`.",
        keywords: &[
                "fea",
                "response spectrum",
                "envelope",
                "stress",
                "displacement",
                "seismic",
            ],
        related: &[
                "fea.StructuralField.stress",
                "fea.StructuralField.displacement",
                "fea.ResponseSpectrumField.pseudoAcceleration",
                "fea.field",
            ],
        sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
                "The result selects field id `structural.response_spectrum.envelope.<quantity>`, for example `structural.response_spectrum.envelope.stress`. Units, shape, and location are those of the wrapped quantity: pascals per element for stress, meters per node for displacement, and so on.",
                "The quantity must be one of `fea.StructuralField.displacement()`, `.rotation()`, `.strain()`, `.stress()`, `.shellMembraneForce()`, `.shellBendingMoment()`, or `.shellTransverseShear()`; derived quantities such as von Mises stress or strain energy are rejected because they are not linear in the response.",
                "Each component is recovered per mode from the modal displacement response and then combined, so the envelope is a component-wise peak magnitude. It is not a simultaneous signed state; do not compute von Mises stress from envelope components.",
                "Produced by modal response-spectrum steps.",
            ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
                "Pass the selector to `fea.field` with the response-spectrum step result, for example `fea.field(run.step(\"response\"), fea.ResponseSpectrumField.envelope(fea.StructuralField.stress()))`.",
            ] },
        BuiltinDocumentationSection { heading: "Errors", paragraphs: &[
                "`RunMat:fea:InvalidInput` reports a missing argument, an argument that is not a `fea.StructuralField`, or a structural quantity that is not linear in the response.",
            ] },
        ],
        examples: ENVELOPE_EXAMPLES,
        faqs: super::RS_FAQS,
        links: super::LINKS,
        implementation: super::IMPLEMENTATION,
        verification: super::VERIFICATION,
        notes: &[],
    }),
    descriptor: &ENVELOPE_DESCRIPTOR,
    inference: selector_entry::infer_response_spectrum_envelope,
    integer_capabilities: &[],
    integer_audit: Some(&selector_entry::STATISTICAL_SELECTOR_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
