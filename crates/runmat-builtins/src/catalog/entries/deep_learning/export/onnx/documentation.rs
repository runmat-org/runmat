use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "ONNX export implementation",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/deep_learning/onnx/mod.rs",
    ),
};
const GRAPH_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Supported ONNX graph lowering",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/deep_learning/onnx/graph.rs",
    ),
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("exportONNXNetwork"),
    slug: Some("exportONNXNetwork"),
    summary: "Export a supported feed-forward network to an ONNX file.",
    description: "`exportONNXNetwork(net, filename)` writes an ONNX graph for supported sequential feed-forward `dlnetwork`, `SeriesNetwork`, and `DAGNetwork` compatibility objects. The call returns no outputs.",
    keywords: &["exportONNXNetwork", "deep learning", "ONNX", "export", "dlnetwork", "SeriesNetwork"],
    related: &["dlnetwork", "fullyConnectedLayer", "softmaxLayer"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Supported network and operators",
            paragraphs: &[
                "The network must begin with a one-dimensional featureInputLayer. This exporter lowers fullyConnectedLayer to ONNX Gemm, reluLayer to Relu, eluLayer to Elu, and softmaxLayer to Softmax. A classificationLayer or regressionLayer may appear only at the end. Other forward layers and general graph connectivity are outside this export path.",
                "Learned fully connected weights and biases become deterministically named ONNX initializers (`layerName.Weights` and `layerName.Bias`). Single-precision learned arrays produce ONNX FLOAT storage unless another learned array requires DOUBLE; mixed single/double learned storage promotes the graph to DOUBLE. Numeric integer learnables enter the runtime's floating model boundary, not an exact integer ONNX graph.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Options and effects",
            paragraphs: &[
                "`OpsetVersion` accepts 13 through 20 (default 14); `BatchSize` is a positive integer. `InputNames` and `OutputNames` each accept exactly one nonempty name. `Verbose` accepts a logical scalar or numeric zero or one. Option names are case-insensitive; a repeated name uses its last value. Native fixed-width integer controls are a RunMat extension. The export validates the supported graph and serializes it before opening the destination through the filesystem service.",
                "ONNX export is host graph serialization, not an accelerator kernel. A resident fully connected Weight or Bias is downloaded from its owning provider only when that property is consumed. Unrelated network properties are not gathered. Provider and filesystem operations can suspend or fail; the serialized file depends on the filesystem capability available to the current native or web runtime.",
                "Unsupported layers, multi-input/multi-output naming, malformed learnables, non-terminal output layers, and unsupported options raise deterministic errors rather than writing partial files.",
            ],
        },
    ],
    examples: &[BuiltinExample {
        id: "feed-forward-file",
        title: "Write a supported feed-forward network",
        program: "layers = {featureInputLayer(2, 'Name', 'input'), fullyConnectedLayer(2, 'Name', 'dense'), reluLayer('Name', 'relu')};\nnet = dlnetwork(layers);\nexportONNXNetwork(net, 'models/model.onnx', 'OpsetVersion', 14);\nassert(isfile('models/model.onnx'));",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
            id: BuiltinExampleFixtureId { local_name: "onnx-export" },
            root: BuiltinFilesystemRoot::IsolatedWorkspace,
            entries: &[BuiltinFilesystemEntry::Directory { relative_path: "models" }],
        }),
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isfile('models/model.onnx'));" },
    }, BuiltinExample {
        id: "softmax-classification-file",
        title: "Export a supported network",
        program: "layers = {featureInputLayer(2, \"Name\", \"features\"), fullyConnectedLayer(2, \"Name\", \"fc\"), softmaxLayer(\"Name\", \"prob\"), classificationLayer(\"Name\", \"class\")};\nnet = dlnetwork(layers);\nexportONNXNetwork(net, \"network.onnx\", \"OpsetVersion\", 14)",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::NativeFilesystem,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isfile('network.onnx'));" },
    }],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq {
        question: "Does exportONNXNetwork execute on the GPU?",
        answer: "No. The exporter serializes graph metadata and learned parameters; it does not launch numerical kernels. It downloads only consumed resident learned parameters before serialization.",
    }],
    links: &[
        SOURCE,
        GRAPH_SOURCE,
        BuiltinDocumentationLink { label: "dlnetwork", target: BuiltinDocumentationLinkTarget::Builtin("dlnetwork") },
        BuiltinDocumentationLink { label: "fullyConnectedLayer", target: BuiltinDocumentationLinkTarget::Builtin("fullyConnectedLayer") },
        BuiltinDocumentationLink { label: "softmaxLayer", target: BuiltinDocumentationLinkTarget::Builtin("softmaxLayer") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[SOURCE, GRAPH_SOURCE],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Feed-forward ONNX bytes, typed options, validation, and selective provider download",
            location: "crates/runmat-runtime/src/builtins/deep_learning/onnx/admission.rs",
        }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
