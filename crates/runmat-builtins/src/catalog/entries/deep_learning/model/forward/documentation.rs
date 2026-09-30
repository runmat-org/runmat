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
    summary: "Evaluate a supported sequential network on numeric observations.",
    description: "`forward(net, X)` executes supported feed-forward layers. Rows of `X` are observations and columns are features. A `dlarray` input produces a `dlarray` output with its format metadata retained.",
    keywords: &["forward", "dlnetwork", "deep learning", "predict", "GPU"],
    related: &["dlnetwork", "dlarray", "dlfeval", "dlgradient", "predict"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Supported execution",
            paragraphs: &[
                "A supported sequence begins with `featureInputLayer` and can use `fullyConnectedLayer`, `reluLayer`, `eluLayer`, and `softmaxLayer`, followed by classification or regression output metadata. Supplied fully connected weights and bias are consumed. Missing parameters from an `Initialize=false` network fail at execution. The output has one row per observation and as many columns as the final feature width.",
                "Host `double` inputs return `double`; host `single` inputs return `single`. In RunMat mode, typed integer predictors produce `double` results. Integer fully connected parameters are converted for the floating-point calculation; with floating predictors, results retain the predictor's `single` or `double` class. Wide integer values can round at this numerical boundary. Host logical input produces `double`. The `dlarray` wrapper retains its input format. Name-value execution options, recurrent state outputs, branches, custom layers, and unsupported input geometry are rejected.",
                "Inside `dlfeval`, supported host `dlarray` inputs and fully connected learnables can participate in the active differentiation tape. This is limited to the implemented layer sequence, not general network training.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Placement",
            paragraphs: &[
                "Supported real, two-dimensional GPU-backed `dlarray` inputs execute through provider operations and return resident `dlarray` Data. Fully connected weights and bias already resident in the input's single/double precision stay on the provider: weights are transposed and bias is reshaped without host transfers. Host parameters are converted and uploaded; resident parameters with a different precision currently cross the host conversion boundary. Other network properties are not downloaded. Unsupported resident forms are rejected explicitly; provider-resident autodiff is not provided by this path. Plain resident numeric inputs currently gather to the host numerical path and return a host array. Provider operations can suspend; a resident result retains its provider ownership.",
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
            program: "net = dlnetwork({featureInputLayer(2); fullyConnectedLayer(1)}); x = dlarray(single([1 2]), 'CB'); y = forward(net, x); fprintf('%d\\n', size(y.Data, 2));",
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
            answer: "Rows are observations and columns are features for this `forward` form.",
        },
        BuiltinDocumentationFaq {
            question: "Does a GPU-backed dlarray have to gather?",
            answer: "The supported real 2-D provider path keeps its input and output resident. Plain resident numeric arrays currently use the host path, and unsupported provider forms report an error.",
        },
        BuiltinDocumentationFaq {
            question: "Can forward return network state as a second output?",
            answer: "No. This sequential subset has one output and does not implement recurrent state output.",
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
