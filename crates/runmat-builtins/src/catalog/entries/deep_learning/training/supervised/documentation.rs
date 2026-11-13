use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Supervised training implementation",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/deep_learning/supervised.rs",
    ),
};
const ADMISSION_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Supervised admission and selective materialization",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/deep_learning/supervised/admission.rs",
    ),
};
const SOLVER_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Supervised forward and backward calculations",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/deep_learning/supervised/solver.rs",
    ),
};
const TARGET_SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Classification and regression target handling",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/deep_learning/supervised/targets.rs",
    ),
};
const TRAINING_OPTIONS: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "RunMat trainingOptions",
    target: BuiltinDocumentationLinkTarget::Builtin("trainingOptions"),
};

const TRAIN_NETWORK_EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "regression-series-network",
        title: "Train a small regression network",
        program: "layers = {featureInputLayer(1,'Name','x'); fullyConnectedLayer(1,'Name','fc'); regressionLayer('Name','out')};\nopts = trainingOptions('adam','MaxEpochs',2,'MiniBatchSize',4,'InitialLearnRate',0.05,'Shuffle','never');\nnet = trainNetwork([0;1;2;3], [1;3;5;7], layers, opts);\nfprintf('%s\\n', class(net));",
        display_output: Some("SeriesNetwork"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(strcmp(class(net),'SeriesNetwork')); assert(strcmp(net.TrainedWith,'adam')); assert(net.TrainingInfo.Epoch == 2);",
        },
    },
    BuiltinExample {
        id: "two-class-network",
        title: "Train a two-class network",
        program: "X = [1 0; 0 1; 1 0; 0 1];\nlabels = [\"left\"; \"right\"; \"left\"; \"right\"];\nlayers = {featureInputLayer(2,'Name','in'); fullyConnectedLayer(2,'Name','fc'); softmaxLayer('Name','prob'); classificationLayer('Name','class')};\nopts = trainingOptions('adam','MaxEpochs',2,'MiniBatchSize',4,'Shuffle','never');\nnet = trainNetwork(X, labels, layers, opts);\nfprintf('%s\\n', class(net));",
        display_output: Some("SeriesNetwork"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(strcmp(class(net),'SeriesNetwork')); assert(numel(net.Classes)==2); assert(strcmp(net.TrainingLoss,'crossentropy'));",
        },
    },
];
const TRAINNET_EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "crossentropy-dlnetwork",
    title: "Train a dlnetwork with crossentropy",
    program: "X = [1 0; 0 1; 1 0; 0 1];\nlabels = [\"left\"; \"right\"; \"left\"; \"right\"];\nlayers = {featureInputLayer(2,'Name','in'); fullyConnectedLayer(2,'Name','fc'); softmaxLayer('Name','prob'); classificationLayer('Name','class')};\nnet = dlnetwork(layers);\nopts = trainingOptions('adam','MaxEpochs',2,'MiniBatchSize',4,'Shuffle','never');\nnet = trainnet(X, labels, net, 'crossentropy', opts);\nfprintf('%s\\n', class(net));",
    display_output: Some("dlnetwork"),
    compatibility: BuiltinExampleCompatibility::RunMat,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(strcmp(class(net),'dlnetwork')); assert(numel(net.Classes)==2); assert(strcmp(net.TrainingLoss,'crossentropy'));",
    },
}];

const COMMON_SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Training data and targets",
        paragraphs: &[
            "RunMat's supported form uses a real finite 2-D predictor matrix with at least one observation: rows are observations and columns are features. Regression targets have one row per observation and the network's output width. Classification targets may be positive numeric class IDs, string or categorical labels, or a one-hot/probability matrix. Sorted label names are retained in the returned network's `Classes` property when labels are used.",
            "Native integer predictors and regression targets enter floating-point training arithmetic at the numeric algorithm boundary. Fixed-width integer classification IDs are read from native storage, so distinct wide IDs are not merged through `double`. Public MATLAB documentation describes numeric arrays without proving every class-specific behavior here; RunMat does not claim per-class MATLAB parity for these forms.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Supported execution",
        paragraphs: &[
            "The in-memory sequential feed-forward subset includes featureInputLayer, fullyConnectedLayer, reluLayer, eluLayer, softmaxLayer, classificationLayer, and regressionLayer. The same subset is used by forward. `trainingOptions` supplies sgdm or adam controls including MaxEpochs, MiniBatchSize, InitialLearnRate, Momentum, Adam decay factors, Epsilon, and deterministic Shuffle handling.",
            "Training runs on the host for `ExecutionEnvironment='auto'` or `'cpu'`. It selectively gathers consumed provider-resident predictor, target, and learned-parameter data after host-decidable controls are checked. Explicit `'gpu'`, `'multi-gpu'`, and `'parallel'` execution is not implemented; this is not a claim that GPU training kernels or browser hardware acceleration run here.",
            "The host loop checks cancellation at epoch and mini-batch boundaries. A cancelled call returns an error without publishing a partially trained network.",
            "Datastores, image and sequence training, recurrent/convolutional/normalization/custom layers, branching graph execution, validation datasets, callbacks, checkpoints, training plots, and autodiff tracing are outside this implementation. Some extra trainingOptions properties are retained as metadata but do not alter this training loop.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Results",
        paragraphs: &[
            "The returned network stores updated fully connected weights in Layers and exposes a derived Learnables table, plus TrainingInfo, TrainingLoss, and TrainedWith metadata. A second requested output returns the training information structure with epoch and iteration counts, loss history, solver and stop reason. An invalid output count rejects before provider data is downloaded.",
        ],
    },
];

pub(super) const TRAIN_NETWORK_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("trainNetwork"),
    slug: Some("trainNetwork"),
    summary: "Train a supported in-memory feed-forward network.",
    description: "`trainNetwork(X,Y,layers,options)` trains a supported sequential feed-forward network over in-memory data. A layer collection returns SeriesNetwork; a simple layerGraph returns DAGNetwork.",
    keywords: &["trainNetwork", "deep learning", "training", "SeriesNetwork"],
    related: &["trainingOptions", "trainnet", "dlnetwork", "predict"],
    sections: COMMON_SECTIONS,
    examples: TRAIN_NETWORK_EXAMPLES,
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Which network topologies can trainNetwork execute?",
            answer: "The supported implementation is sequential feed-forward. A simple layerGraph can be supplied and yields a DAGNetwork compatibility object, but branching graph execution and unsupported layer families reject.",
        },
        BuiltinDocumentationFaq {
            question: "Does auto use GPU training?",
            answer: "No. The current training loop is host-owned. Consumed resident arrays are gathered after host input admission; explicitly requesting GPU execution rejects.",
        },
    ],
    links: &[
        SOURCE,
        ADMISSION_SOURCE,
        SOLVER_SOURCE,
        TARGET_SOURCE,
        TRAINING_OPTIONS,
        BuiltinDocumentationLink {
            label: "MATLAB trainNetwork reference",
            target: BuiltinDocumentationLinkTarget::External(
                "https://www.mathworks.com/help/deeplearning/ref/trainnetwork.html",
            ),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[SOURCE, ADMISSION_SOURCE, SOLVER_SOURCE, TARGET_SOURCE],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Regression, classification, exact integer labels, and pre-effect admission",
            location: "crates/runmat-runtime/src/builtins/deep_learning/tests.rs",
        }],
        notes: &["Native/WASM portability is a code-placement contract, not a claim of separately executed browser training tests."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};

pub(super) const TRAINNET_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("trainnet"),
    slug: Some("trainnet"),
    summary: "Train a supported dlnetwork with a built-in loss.",
    description: "`trainnet(X,Y,net,lossFcn,options)` trains a supported sequential dlnetwork-compatible model. `mse`/`mean-squared-error` select regression; `crossentropy` selects classification.",
    keywords: &["trainnet", "deep learning", "training", "dlnetwork"],
    related: &["trainingOptions", "trainNetwork", "dlnetwork", "forward", "dlfeval", "dlgradient"],
    sections: COMMON_SECTIONS,
    examples: TRAINNET_EXAMPLES,
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Can lossFcn be a custom function handle?",
            answer: "No. This training loop accepts its built-in mse and crossentropy loss names. Custom host autodiff work can use explicit dlfeval/dlgradient loops; a custom loss handle passed to trainnet rejects.",
        },
        BuiltinDocumentationFaq {
            question: "Are model parameters kept resident during GPU training?",
            answer: "No. Training is host-owned. Only consumed resident data and learnables are gathered after host controls are admitted; explicit GPU execution is unsupported.",
        },
    ],
    links: &[
        SOURCE,
        ADMISSION_SOURCE,
        SOLVER_SOURCE,
        TARGET_SOURCE,
        TRAINING_OPTIONS,
        BuiltinDocumentationLink {
            label: "MATLAB trainnet reference",
            target: BuiltinDocumentationLinkTarget::External(
                "https://www.mathworks.com/help/deeplearning/ref/trainnet.html",
            ),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[SOURCE, ADMISSION_SOURCE, SOLVER_SOURCE, TARGET_SOURCE],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Categorical classification, loss rejection, and host execution controls",
            location: "crates/runmat-runtime/src/builtins/deep_learning/tests.rs",
        }],
        notes: &["Custom loss handles and provider-resident training are not implemented by this public route."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
