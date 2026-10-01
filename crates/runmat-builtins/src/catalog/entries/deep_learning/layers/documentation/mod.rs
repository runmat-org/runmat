use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Layer constructor implementation",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/deep_learning/layers.rs"),
};

macro_rules! layer_documentation {
    ($name:ident, $title:literal, $summary:literal, $description:literal, $keywords:expr,
     $program:literal, $output:literal, $assertions:literal, $detail:literal) => {
        pub(in crate::catalog::entries::deep_learning::layers) const $name: BuiltinDocumentation = BuiltinDocumentation {
            authority: BuiltinDocumentationAuthority::Catalog,
            title: Some($title),
            slug: Some($title),
            summary: $summary,
            description: $description,
            keywords: $keywords,
            related: &["layerGraph", "dlnetwork", "forward"],
            sections: &[
                BuiltinDocumentationSection { heading: "Layer metadata", paragraphs: &[
                    $detail,
                    "The result is a host layer object. Optional name-value arguments may be omitted to use defaults. `Name` and `Description` are text. Recognized name-value options are stored under their canonical property names; unknown, duplicate, and cross-layer option names are rejected. A layer graph can retain recognized options that are not yet executable by `forward` or training.",
                ] },
                BuiltinDocumentationSection { heading: "Placement and validation", paragraphs: &[
                    "The constructor validates required sizes and option syntax before downloading resident controls. Numeric and logical controls needed for validation may be gathered; data, learnables, recurrent state, and initializer captures retain their original storage and provider ownership. The result is a host metadata object, not a GPU kernel invocation. Invalid host text options reject without downloading earlier resident values. Execution validates parameter geometry and provider support separately.",
                    "Structural sizes are stored in double-valued layer properties. Typed integer sizes must survive that conversion exactly: the exact `uint64` value 9,007,199,254,740,993 is rejected, while an aligned exactly representable larger integer remains admissible if it fits the host size range. This boundary does not imply that a network with that many units can be allocated.",
                ] },
            ],
            examples: &[BuiltinExample {
                id: "metadata", title: "Create and inspect a layer", program: $program,
                display_output: Some($output), compatibility: BuiltinExampleCompatibility::RunMat,
                harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None,
                requirements: BuiltinExampleRequirements::NONE,
                verification: BuiltinExampleVerification::Assertions { source: $assertions },
            }],
            example_exemption: None,
            faqs: &[
                BuiltinDocumentationFaq { question: "Does constructing the layer run a model or GPU kernel?", answer: "No. This call creates a host layer object; model execution is a separate operation." },
                BuiltinDocumentationFaq { question: "Are all recognized layer options implemented by forward?", answer: "No. Constructor acceptance records canonical options but does not establish an executable network form. Pending operational options require separate execution admission." },
            ],
            links: &[SOURCE, BuiltinDocumentationLink { label: "layerGraph", target: BuiltinDocumentationLinkTarget::Builtin("layerGraph") }],
            media: &[],
            evidence: BuiltinDocumentationEvidence {
                implementation: &[SOURCE],
                verification: &[
                    BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Layer construction and admission", location: "crates/runmat-runtime/src/builtins/deep_learning/tests.rs" },
                    BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Resident data and control boundaries", location: "crates/runmat-runtime/src/builtins/deep_learning/layers/residency_tests.rs" },
                ],
                notes: &["Constructor metadata is catalog-authoritative; execution support remains layer-specific. Native and browser behavior require integrated qualification."],
            },
            introduced: None,
            status: Some(BuiltinDocumentationStatus::Partial),
        };
    };
}

mod fully_connected;
pub(super) use fully_connected::FULLY_CONNECTED;
mod feature_input;
pub(super) use feature_input::FEATURE_INPUT;
mod sequence_input;
pub(super) use sequence_input::SEQUENCE_INPUT;
mod relu;
pub(super) use relu::RELU;
mod elu;
pub(super) use elu::ELU;
mod softmax;
pub(super) use softmax::SOFTMAX;
mod classification;
pub(super) use classification::CLASSIFICATION;
mod regression;
pub(super) use regression::REGRESSION;
mod layer_norm;
pub(super) use layer_norm::LAYER_NORM;
mod global_pool;
pub(super) use global_pool::GLOBAL_POOL;
mod lstm;
pub(super) use lstm::LSTM;
mod bilstm;
pub(super) use bilstm::BILSTM;
mod convolution;
pub(super) use convolution::CONVOLUTION;
