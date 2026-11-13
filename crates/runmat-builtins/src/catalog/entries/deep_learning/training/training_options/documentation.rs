use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Runtime training-option construction",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/deep_learning/training.rs",
    ),
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("trainingOptions"),
    slug: Some("trainingOptions"),
    summary: "Create a host training-configuration object.",
    description: "`trainingOptions(solver, Name, Value, ...)` creates solver-specific host configuration for `sgdm`, `adam`, or `rmsprop`. It does not start training or run a provider kernel.",
    keywords: &["trainingOptions", "deep learning", "training", "Adam", "SGDM", "RMSProp"],
    related: &["trainNetwork", "trainnet", "adamupdate"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Solvers and options",
            paragraphs: &[
                "The constructor recognizes `sgdm`, `adam`, and `rmsprop`. It initializes `MaxEpochs`, `MiniBatchSize`, `InitialLearnRate`, `Shuffle`, `Verbose`, `Plots`, and `ExecutionEnvironment`, then stores supplied name-value options as properties. `MaxEpochs` and `MiniBatchSize` default to 30 and 128; the initial learning rate is 0.01 for SGDM and 0.001 for Adam and RMSProp.",
                "The result has class `nnet.cnn.TrainingOptionsSGDM`, `nnet.cnn.TrainingOptionsADAM`, or `nnet.cnn.TrainingOptionsRMSProp`, matching the positional solver. `trainNetwork` and `trainnet` execute supported host training forms for all three solvers. Operational options that a training route cannot consume are rejected before training effects rather than silently ignored.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Representation and placement",
            paragraphs: &[
                "The result is a host object. Option values are retained as supplied, including exact typed integer controls and provider-owned values, without downloading them during construction. Each training consumer validates the controls it uses. Unknown names are retained as lower-case metadata keys for forward-compatible option storage rather than silently treated as implemented behavior.",
                "The constructor supports one result; a discarded result still validates the solver and option list. A request for more than one result fails before inspecting provider data.",
            ],
        },
    ],
    examples: &[BuiltinExample {
        id: "adam-options",
        title: "Create Adam options",
        program: "opts = trainingOptions('adam', 'MaxEpochs', 5, 'Verbose', false); fprintf('%s %d\\n', opts.SolverName, opts.MaxEpochs);",
        display_output: Some("adam 5"),
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(strcmp(opts.SolverName, 'adam')); assert(opts.MaxEpochs == 5); assert(opts.Verbose == false);",
        },
    }],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Does constructing RMSProp options run RMSProp training?",
            answer: "No. This constructor only creates options; the supported host training route applies RMSProp when those options are passed to `trainNetwork` or `trainnet`.",
        },
        BuiltinDocumentationFaq {
            question: "Are typed integer option values converted to double?",
            answer: "No. The object retains each supplied value. A training consumer separately checks and decodes the controls it actually uses.",
        },
    ],
    links: &[
        SOURCE,
        BuiltinDocumentationLink {
            label: "MATLAB trainingOptions reference",
            target: BuiltinDocumentationLinkTarget::External(
                "https://www.mathworks.com/help/deeplearning/ref/trainingoptions.html",
            ),
        },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[SOURCE],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Training option defaults, exact value retention, and solver admission",
            location: "crates/runmat-runtime/src/builtins/deep_learning/supervised/config/tests.rs",
        }],
        notes: &["The supported training-consumer subset is narrower than the metadata accepted by this constructor."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
