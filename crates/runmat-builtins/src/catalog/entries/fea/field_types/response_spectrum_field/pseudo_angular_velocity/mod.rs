use crate::fea_contract::{
    catalog_entry::{self, ConstructorDocumentationDefinition, ConstructorEntryDefinition},
    selector_entry::{self, NO_INPUTS, SELECTOR_OUTPUT},
};
use crate::*;

const PSEUDO_ANGULAR_VELOCITY_SIGNATURES: [BuiltinSignatureDescriptor; 1] =
    [BuiltinSignatureDescriptor {
        label: "field = fea.ResponseSpectrumField.pseudoAngularVelocity()",
        inputs: &NO_INPUTS,
        outputs: &SELECTOR_OUTPUT,
    }];
pub const PSEUDO_ANGULAR_VELOCITY_DESCRIPTOR: BuiltinDescriptor =
    selector_entry::selector_descriptor(&PSEUDO_ANGULAR_VELOCITY_SIGNATURES);
const PSEUDO_ANGULAR_VELOCITY_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "select-pseudo-angular-velocity",
    title: "Select the pseudo-angular velocity envelope",
    program: "selector = fea.ResponseSpectrumField.pseudoAngularVelocity();",
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
    name: "fea.ResponseSpectrumField.pseudoAngularVelocity",
    documentation: catalog_entry::documentation(ConstructorDocumentationDefinition {
        title: "fea.ResponseSpectrumField.pseudoAngularVelocity",
        summary: "Select the response-spectrum pseudo-angular velocity envelope for fea.field.",
        description: "`field = fea.ResponseSpectrumField.pseudoAngularVelocity()` returns a `fea.ResponseSpectrumField` selector for `structural.response_spectrum.nodal.pseudo_angular_velocity`, the modally combined peak pseudo-angular velocity of every node in a response-spectrum analysis.",
        keywords: &[
                "fea",
                "response spectrum",
                "seismic",
                "shock",
                "pseudo-angular velocity",
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
                "The selector is an immutable value; it does no I/O. `fea.field` resolves it to the stored field id `structural.response_spectrum.nodal.pseudo_angular_velocity`.",
                "Physical quantity: peak pseudo-angular velocity of each node: the rotation scaled by each mode's circular frequency before modal combination. Units: radians per second (rad/s). Shape: a three-component vector per node. Location: node. Rotational quantities require nodes with rotational degrees of freedom (beams or shells).",
                "Response-spectrum results are peak envelopes: each mode's spectral response is combined with the spectrum's modal and directional combination rules. Components are nonnegative magnitudes, not a simultaneous signed state, so signs and phase between nodes are not preserved.",
                "Produced by modal response-spectrum steps that follow a modal extraction in the study's sequence and reference a `fea.ResponseSpectrum` in the model.",
            ] },
        BuiltinDocumentationSection { heading: "Use in a study", paragraphs: &[
                "After running a response-spectrum study, call `fea.field(run.step(\"response\"), fea.ResponseSpectrumField.pseudoAngularVelocity())` with the response step's id, then page or plot the field.",
            ] },
        ],
        examples: PSEUDO_ANGULAR_VELOCITY_EXAMPLES,
        faqs: super::RS_FAQS,
        links: super::LINKS,
        implementation: super::IMPLEMENTATION,
        verification: super::VERIFICATION,
        notes: &[],
    }),
    descriptor: &PSEUDO_ANGULAR_VELOCITY_DESCRIPTOR,
    inference: selector_entry::infer_response_spectrum_nodal,
    integer_capabilities: &[],
    integer_audit: Some(&selector_entry::STATISTICAL_SELECTOR_INTEGER_AUDIT),
    filesystem_read: false,
    distributed: BuiltinDistributedPolicy::Unsupported,
});
