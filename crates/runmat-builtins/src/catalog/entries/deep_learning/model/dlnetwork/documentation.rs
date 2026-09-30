use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Network construction implementation",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/deep_learning/model.rs",
    ),
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("dlnetwork"),
    slug: Some("dlnetwork"),
    summary: "Create a value-backed network from supported sequential layers.",
    description: "`dlnetwork(layers)` stores a supported layer collection or `layerGraph` in a network object with layer names, learnables, state, and input/output names. RunMat currently executes a sequential feed-forward subset rather than arbitrary Deep Learning Toolbox architectures.",
    keywords: &["dlnetwork", "deep learning", "network", "learnables", "forward"],
    related: &["forward", "dlfeval", "dlgradient", "layerGraph", "fullyConnectedLayer"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Layers and initialization",
            paragraphs: &[
                "A layer object, cell collection, or layer graph can supply the layers. The supported forward sequence uses `featureInputLayer`, `fullyConnectedLayer`, `reluLayer`, `eluLayer`, `softmaxLayer`, `classificationLayer`, and `regressionLayer`. Branching graph connections, recurrent and convolutional layers, custom layers, and general training graphs are not executed by this constructor.",
                "The result stores `Layers`, `LayerNames`, `InputNames`, `OutputNames`, `Connections`, `Learnables`, `State`, and `Initialized`. Supplied fully connected weights and bias are retained. With the default `Initialize=true`, missing weights receive deterministic values based on the preceding feature width and missing bias values are zero. `Initialize=false` retains missing parameters and marks the network uninitialized; `forward` requires the parameters before execution.",
                "The supported `OutputNames` option names the final layer of this single-output sequential network. Unknown options and a name that does not identify that final layer are rejected. InputNames is derived from the first layer, not an accepted constructor option.",
                "A vector-valued feature input can be stored as layer metadata in RunMat mode, but this sequential forward implementation requires one feature dimension and rejects a vector width when it must execute or initialize a following fully connected layer.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Placement and learnables",
            paragraphs: &[
                "Construction creates host metadata without downloading nested numeric learnables. A resident weight remains owned by its input layer until a consumer explicitly prepares that weight. The constructor does not run a GPU kernel or start a training tape.",
                "Within `dlfeval`, supported fully connected learnables can be traced for `dlgradient` in custom training loops. This does not imply support for every network architecture or provider-resident training operation.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "sequential",
            title: "Construct a sequential network",
            program: "layers = {featureInputLayer(2, 'Name', 'in'); fullyConnectedLayer(3, 'Name', 'fc'); reluLayer('Name', 'relu')}; net = dlnetwork(layers); fprintf('%d\\n', numel(net.Layers));",
            display_output: Some("3"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(numel(net.Layers) == 3); assert(net.Initialized); assert(numel(net.LayerNames) == 3);",
            },
        },
        BuiltinExample {
            id: "deferred-initialization",
            title: "Keep missing parameters uninitialized",
            program: "layers = {featureInputLayer(2); fullyConnectedLayer(3)}; net = dlnetwork(layers, 'Initialize', false); fprintf('%d\\n', net.Initialized);",
            display_output: Some("0"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(~net.Initialized); assert(numel(net.Layers) == 2);",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Does dlnetwork run an arbitrary layer graph?",
            answer: "No. It stores layer metadata and supports the listed sequential feed-forward subset. Branching connections and other layer families are outside this runtime implementation.",
        },
        BuiltinDocumentationFaq {
            question: "What happens to supplied weights and bias?",
            answer: "They are preserved. Default initialization fills only missing fully connected parameters; `Initialize=false` leaves missing parameters absent.",
        },
        BuiltinDocumentationFaq {
            question: "Does construction download GPU learnables?",
            answer: "No. Construction copies layer metadata and retains resident learnables without gathering their payloads.",
        },
    ],
    links: &[
        SOURCE,
        BuiltinDocumentationLink {
            label: "forward",
            target: BuiltinDocumentationLinkTarget::Builtin("forward"),
        },
        BuiltinDocumentationLink {
            label: "layerGraph",
            target: BuiltinDocumentationLinkTarget::Builtin("layerGraph"),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[SOURCE],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Network construction and initialization",
            location: "crates/runmat-runtime/src/builtins/deep_learning/tests.rs",
        }],
        notes: &["Network execution and browser/provider qualification are separate from the constructor's metadata contract."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
