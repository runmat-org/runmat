use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Runtime graph implementation",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/deep_learning/graph.rs",
    ),
};

pub(super) const LAYER_GRAPH: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("layerGraph"),
    slug: Some("layerGraph"),
    summary: "Create a host layer-graph metadata object.",
    description: "`layerGraph()` creates an empty graph. `layerGraph(layers)` accepts one layer object or a cell collection of layer objects and stores its layers and names in a graph object. Connection editing is not yet implemented.",
    keywords: &["layerGraph", "deep learning", "network graph", "layers"],
    related: &["analyzeNetwork", "dlnetwork", "featureInputLayer", "fullyConnectedLayer"],
    sections: &[
        BuiltinDocumentationSection { heading: "Stored graph", paragraphs: &[
            "The result has `Layers`, `LayerNames`, and an empty `Connections` structure. Empty layer names receive deterministic `layer_1`, `layer_2`, and subsequent names. Each layer must be an object; the operation does not infer connections from the layer sequence.",
            "The current implementation accepts zero arguments or one layer object, cell collection, or internal output list. A direct numeric array is not a layer collection. Editing helpers such as `addLayers` and `connectLayers` are outside this implementation.",
        ] },
        BuiltinDocumentationSection { heading: "Placement", paragraphs: &[
            "Graph construction is host metadata work. Direct resident numeric input is not a layer collection and rejects without provider access. Nested resident values in valid layer objects remain owned by those objects and are not downloaded. There is no graph GPU kernel or fusion route. The graph is a new value-backed object; construction can report invalid input but does not suspend.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "empty", title: "Create an empty graph", program: "g = layerGraph(); fprintf('%d\\n', numel(g.Layers));", display_output: Some("0"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(numel(g.Layers) == 0); assert(numel(g.LayerNames) == 0);" } },
        BuiltinExample { id: "layers", title: "Store named layers", program: "layers = {featureInputLayer(4, 'Name', 'in'); fullyConnectedLayer(2, 'Name', 'fc')}; g = layerGraph(layers); fprintf('%d\\n', numel(g.Layers));", display_output: Some("2"), compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(numel(g.Layers) == 2); assert(numel(g.LayerNames) == 2);" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does layerGraph connect successive layers?", answer: "No. The current graph stores the layer list and names with an empty Connections structure; connection editing is not implemented." },
        BuiltinDocumentationFaq { question: "Does layerGraph execute on a GPU?", answer: "No. It constructs host metadata and does not submit a provider kernel." },
    ],
    links: &[SOURCE,
        BuiltinDocumentationLink { label: "analyzeNetwork", target: BuiltinDocumentationLinkTarget::Builtin("analyzeNetwork") },
        BuiltinDocumentationLink { label: "dlnetwork", target: BuiltinDocumentationLinkTarget::Builtin("dlnetwork") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[SOURCE],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Layer collection and graph construction", location: "crates/runmat-runtime/src/builtins/deep_learning/tests.rs" }],
        notes: &["The graph is a bounded metadata object, not a complete connection-editing or UI graph implementation. Native, browser, and provider behavior require integrated qualification."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};

pub(super) const ANALYZE_NETWORK: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("analyzeNetwork"),
    slug: Some("analyzeNetwork"),
    summary: "Report layer count and names without opening the network analyzer UI.",
    description: "`analyzeNetwork(networkOrLayers)` accepts a layer graph, one layer object, or a cell collection of layers. It returns a host `nnet.cnn.NetworkAnalysis` metadata object with `NumLayers`, `LayerNames`, and `Diagnostics`. Unlike MATLAB's interactive analyzer, this form does not open a UI.",
    keywords: &["analyzeNetwork", "deep learning", "network analysis", "layer names"],
    related: &["layerGraph", "dlnetwork"],
    sections: &[
        BuiltinDocumentationSection { heading: "Analysis result", paragraphs: &[
            "The analysis object reports the number and names of the supplied layers. Missing layer names receive deterministic `layer_1`, `layer_2`, and subsequent names. This is metadata inspection, not shape propagation, training, connection validation, or interactive visualization.",
            "An internal output list of layer objects is also accepted as a layer collection. Numeric arrays are not layer collections.",
        ] },
        BuiltinDocumentationSection { heading: "Placement and current limits", paragraphs: &[
            "The operation reads host layer metadata and returns a new host object. Direct resident numeric input is invalid and rejects without provider access; nested resident properties not used by the analysis remain with their owning objects. There is no GPU analysis kernel, UI effect, or fusion route; the call can throw but does not suspend.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "one-layer", title: "Inspect one layer", program: "a = analyzeNetwork({reluLayer()}); fprintf('%d\\n', a.NumLayers);", display_output: Some("1"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(a.NumLayers == 1); assert(numel(a.LayerNames) == 1);" } },
        BuiltinExample { id: "layer-collection", title: "Inspect a feature input and activation", program: "analysis = analyzeNetwork({featureInputLayer(4), reluLayer()}); disp(analysis.NumLayers);", display_output: Some("2"), compatibility: BuiltinExampleCompatibility::RunMat, harness: BuiltinExampleHarness::Portable, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Assertions { source: "assert(analysis.NumLayers == 2); assert(numel(analysis.LayerNames) == 2);" } },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Does analyzeNetwork open an interactive analyzer?", answer: "No. This RunMat implementation returns a metadata object and does not open a UI." },
        BuiltinDocumentationFaq { question: "Does it validate tensor shapes through the network?", answer: "No. It reports layer count and names only." },
    ],
    links: &[SOURCE, BuiltinDocumentationLink { label: "layerGraph", target: BuiltinDocumentationLinkTarget::Builtin("layerGraph") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[SOURCE],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Network analysis metadata", location: "crates/runmat-runtime/src/builtins/deep_learning/tests.rs" }],
        notes: &["The returned metadata object intentionally differs from MATLAB's interactive analyzer. Native and browser execution still need integrated qualification."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
