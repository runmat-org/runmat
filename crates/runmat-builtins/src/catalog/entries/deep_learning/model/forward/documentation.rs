use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Network forward implementation",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/deep_learning/model.rs",
    ),
};

const HOST_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Host network execution",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/deep_learning/model/host.rs",
    ),
};
const PROVIDER_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Provider-resident network execution",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/deep_learning/model/gpu.rs",
    ),
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("forward"),
    slug: Some("forward"),
    summary: "Evaluate a sequential network on numeric or formatted observations.",
    description: "`forward(net, X)` evaluates the network's admitted layer sequence. Raw feature matrices use rows for observations and columns for channels; sequence and spatial inputs use the declared input layer's axis convention. For a `dlarray` input, output labels follow the layer sequence: channel transforms retain the axis order, while pooling and recurrent output modes can reduce axes. `[Y, state] = forward(net, X)` also returns updated recurrent State without mutating `net`.",
    keywords: &["forward", "dlnetwork", "deep learning", "predict", "GPU"],
    related: &["dlnetwork", "dlarray", "dlfeval", "dlgradient", "predict"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Supported execution",
            paragraphs: &[
                "A sequence begins with an input layer and proceeds through admitted channel transforms, convolution, recurrent, normalization, pooling, and activation layers. The output shape follows the input's semantic axes and each consumed layer option. Output layers provide classification or regression metadata. Missing parameters from an `Initialize=false` network fail before execution.",
                "Host `double` inputs return `double`; host `single` inputs return `single`. In RunMat mode, typed integer predictors produce `double` results. Consumed integer learnables are converted for the floating-point calculation; with floating predictors, results retain the predictor's `single` or `double` class. Wide integer values can round at this numerical boundary. Host logical input produces `double`. The `dlarray` wrapper carries the executed output's semantic axis labels. Unsupported options, branches, custom layers, and invalid geometry are rejected before execution.",
                "Inside `dlfeval`, admitted host `dlarray` inputs and learnables can participate in the differentiation tape. `[Y, state]` projects the updated canonical Layer/Parameter/Value table; the value network passed to `forward` remains unchanged.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Placement",
            paragraphs: &[
                "Supported real feature matrices execute through provider operations. Sequence and spatial inputs can use global 1-D average pooling when the provider supports the complete reduction and output-presentation route. Raw resident arrays return resident arrays; GPU-backed `dlarray` inputs retain a `dlarray` wrapper and the executed output's labels. Provider operations can suspend, and resident results retain their provider ownership and placement intent.",
                "Provider routes can compose admitted layer normalization, global 1-D average pooling, fully connected layers, ReLU, ELU, and softmax. Admission checks the complete sequence, physical views, and output-axis presentation before submitting operations. Supported contiguous and lazy-permuted inputs retain residency. Resident Scale, Offset, weights, and bias must match the input precision; host parameters are converted and uploaded. Support depends on the provider's operations and layout capabilities.",
                "Fully connected weights and bias already resident in the input's single/double precision stay on the provider: weights are transposed and bias is reshaped without host transfers. Host parameters are converted and uploaded. Automatically placed resident parameters with a different precision may require host conversion or host execution; explicit resident parameter precision mismatches report an error without silently gathering. Other network properties are not downloaded. An unsupported route can gather automatically placed inputs for host execution; explicit `gpuArray` inputs report an error instead of silently gathering. Resident differentiation is limited to the admitted network stages and adjoints documented for `dlgradient`; forward execution alone does not imply a resident backward route.",
            ],
        },
    ],
    examples: &[
        BuiltinExample {
            id: "numeric-observations",
            title: "Evaluate rows of observations",
            program: "layers = {featureInputLayer(2, 'Name', 'in'); fullyConnectedLayer(2, 'Name', 'fc'); softmaxLayer('Name', 'prob')}; net = dlnetwork(layers); y = forward(net, [1 2; 3 4]); fprintf('%d %d\\n', size(y, 1), size(y, 2));",
            display_output: Some("2 2"),
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(size(y, 1) == 2); assert(size(y, 2) == 2);",
            },
        },
        BuiltinExample {
            id: "formatted-dlarray",
            title: "Preserve a dlarray format",
            program: "net = dlnetwork({featureInputLayer(2); fullyConnectedLayer(1)}); x = dlarray(single([1; 2]), 'CB'); y = forward(net, x); fprintf('%d\\n', size(y.Data, 2));",
            display_output: Some("1"),
            compatibility: BuiltinExampleCompatibility::RunMat,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions {
                source: "assert(strcmp(y.Format, 'CB')); assert(strcmp(class(y.Data), 'single'));",
            },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Are observations stored in rows or columns?",
            answer: "Raw feature matrices use rows for observations and columns for channels. Formatted dlarray and sequence or spatial inputs follow their semantic axis labels and input layer.",
        },
        BuiltinDocumentationFaq {
            question: "Does a GPU-backed dlarray have to gather?",
            answer: "Supported provider routes keep raw arrays and dlarray payloads resident. Unsupported routes may gather automatically placed inputs; explicit gpuArray inputs report an error without silently gathering.",
        },
        BuiltinDocumentationFaq {
            question: "Can forward return network state as a second output?",
            answer: "Yes. `[Y, state] = forward(net, X)` returns a canonical State table. `net` is not mutated; pass or assign the returned state explicitly for subsequent recurrent execution.",
        },
    ],
    links: &[
        SOURCE,
        HOST_SOURCE,
        PROVIDER_SOURCE,
        BuiltinDocumentationLink {
            label: "dlnetwork",
            target: BuiltinDocumentationLinkTarget::Builtin("dlnetwork"),
        },
        BuiltinDocumentationLink {
            label: "dlarray",
            target: BuiltinDocumentationLinkTarget::Builtin("dlarray"),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[SOURCE, HOST_SOURCE, PROVIDER_SOURCE],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Host and provider network forward examples",
            location: "crates/runmat-runtime/src/builtins/deep_learning/tests.rs",
        }],
        notes: &["This describes the implemented sequential subset; it is not a claim of general Deep Learning Toolbox network parity."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
