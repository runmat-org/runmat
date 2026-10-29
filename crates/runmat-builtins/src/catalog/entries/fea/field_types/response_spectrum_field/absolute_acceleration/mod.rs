use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition, ConstructorEntryDefinition},
    selector_entry::{self, NO_INPUTS, SELECTOR_OUTPUT},
};
use crate::*;

const ABSOLUTE_ACCELERATION_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "field = fea.ResponseSpectrumField.absoluteAcceleration()",
        inputs: &NO_INPUTS,
        outputs: &SELECTOR_OUTPUT,
    }];
pub const ABSOLUTE_ACCELERATION_DESCRIPTOR: BuiltinDescriptor =
    selector_entry::selector_descriptor(&ABSOLUTE_ACCELERATION_SIGNATURES);
const ABSOLUTE_ACCELERATION_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "select-absolute-acceleration",
    title: "Select the absolute acceleration envelope",
    program: "selector = fea.ResponseSpectrumField.absoluteAcceleration();",
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
    name: "fea.ResponseSpectrumField.absoluteAcceleration",
    documentation: catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.ResponseSpectrumField.absoluteAcceleration",
        summary: "Select the response-spectrum absolute acceleration envelope for fea.field.",
        description: "`field = fea.ResponseSpectrumField.absoluteAcceleration()` returns a `fea.ResponseSpectrumField` selector for `structural.response_spectrum.nodal.absolute_acceleration`, the modally combined peak absolute acceleration of every node in a response-spectrum analysis.",
        keywords: &[
                "fea",
                "response spectrum",
                "seismic",
                "shock",
                "absolute acceleration",
                "envelope",
            ],
        related: &[
                "fea.field",
                "fea.ResponseSpectrumField.envelope",
                "fea.ResponseSpectrum.new",
                "fea.run",
            ],
        sections: &[
        BuiltinDocumentationSection { heading: "Behavior", paragraphs: &[
                "The selector is an immutable value; it does no I/O. `fea.field` resolves it to the stored field id `structural.response_spectrum.nodal.absolute_acceleration`.",
                "Physical quantity: peak total acceleration of each node, base motion included; this is the acceleration equipment mounted at the node experiences. Units: meters per second squared (m/s^2). Shape: a three-component vector per node. Location: node.",
                "Response-spectrum results are peak envelopes: each mode's spectral response is combined with the spectrum's modal and directional combination rules. Components are nonnegative magnitudes, not a simultaneous signed state, so signs and phase between nodes are not preserved.",
                "Produced by modal response-spectrum steps that follow a modal extraction in the study's sequence and reference a `fea.ResponseSpectrum` in the model.",
            ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
                "After running a response-spectrum study, call `fea.field(run.step(\"response\"), fea.ResponseSpectrumField.absoluteAcceleration())` with the response step's id, then page or plot the field.",
            ] },
        ],
        examples: ABSOLUTE_ACCELERATION_EXAMPLES,
        faqs: super::RS_FAQS,
        links: super::LINKS,
        implementation: super::IMPLEMENTATION,
        verification: super::VERIFICATION,
        notes: &[],
    }),
    descriptor: &ABSOLUTE_ACCELERATION_DESCRIPTOR,
    inference: selector_entry::infer_response_spectrum_nodal,
    integer_capabilities: &[],
    integer_audit: Some(&selector_entry::STATISTICAL_SELECTOR_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
