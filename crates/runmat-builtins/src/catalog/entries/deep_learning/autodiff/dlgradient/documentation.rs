use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Runtime gradient entry and target traversal",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/deep_learning/autodiff/gradient.rs",
    ),
};
const REVERSE_PASS: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Reverse-mode differentiation",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/deep_learning/autodiff/gradient/backward.rs",
    ),
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("dlgradient"),
    slug: Some("dlgradient"),
    summary: "Differentiate a scalar dlarray loss recorded inside dlfeval.",
    description: "`dlgradient(loss, targets...)` walks the active host reverse-mode tape. Each requested output is a gradient for the corresponding traced target; a dlarray target produces a dlarray gradient, while supported learnables structures preserve their Layer and Parameter metadata and replace Value leaves.",
    keywords: &["dlgradient", "deep learning", "automatic differentiation", "custom training"],
    related: &["dlarray", "dlfeval", "dlnetwork", "forward"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Traced inputs and outputs",
            paragraphs: &[
                "Call `dlgradient` from a `dlfeval` callback. The loss must be a scalar dlarray produced on that callback's tape; a separate dlarray constructed outside the traced calculation is not a substitute. Request no more gradient outputs than supplied targets. A target can be a traced dlarray, supported dlnetwork learnables, or a structure/table/cell tree containing traced dlarray leaves.",
                "Parameter tables use string Layer and Parameter columns and a cell Value column. Only Value is differentiated; metadata is preserved, including custom parameter names and repeated labels. Assigning gradients or values to a network separately validates that network\'s canonical typed parameter roles.",
                "The current host tape records dlarray plus, minus, elementwise times and rdivide, matrix mtimes, scalar `sum(..., 'all')`, and admitted sequential network operations: fully connected, convolution1d, LSTM/BiLSTM, layer normalization, global average pooling, ReLU, ELU, and softmax. Prediction and requested recurrent State outputs share the execution trace. A real double or single target keeps its class, geometry, Format, and Labels. Logical Data are admitted through the arithmetic 0/1-to-double boundary and produce double gradients.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Current limits and residency",
            paragraphs: &[
                "Scalar broadcasting has a reverse reduction. General higher-rank implicit-expansion gradient reduction, custom layers, sparse or complex differentiation, and provider-resident tape kernels are not implemented. A GPU-backed dlarray can still be wrapped without download outside an active tape, but differentiation rejects that input instead of silently gathering it.",
                "The callback and tape are execution state: nested callbacks keep separate graphs, and suspension does not expose one callback's tape to another invocation. Gradients are computed on the host; ordinary gpuArray arithmetic outside this tape retains its separate provider behavior.",
            ],
        },
    ],
    examples: &[BuiltinExample {
        id: "square-loss",
        title: "Differentiate an elementwise square loss",
        program: "X = dlarray(single([1 2 3])); [loss, gradients] = dlfeval(@squareLoss, X); assert(isequal(gradients.Data, single([2 4 6])));\nfunction [loss, gradients] = squareLoss(X)\n    Y = X .* X;\n    loss = sum(Y, 'all');\n    gradients = dlgradient(loss, X);\nend",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(strcmp(class(gradients.Data), 'single')); assert(isequal(size(gradients.Data), [1 3])); assert(isequal(gradients.Data, single([2 4 6])));",
        },
    }],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Can I differentiate a GPU-backed dlarray?",
            answer: "Not with this host tape. RunMat reports a provider-kernel gap rather than downloading the wrapped array implicitly.",
        },
        BuiltinDocumentationFaq {
            question: "What does dlgradient return for network learnables?",
            answer: "A matching learnables tree: Layer and Parameter metadata stay in place and traced Value leaves become dlarray gradients.",
        },
    ],
    links: &[SOURCE, REVERSE_PASS, BuiltinDocumentationLink {
        label: "dlfeval",
        target: BuiltinDocumentationLinkTarget::Builtin("dlfeval"),
    }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[SOURCE, REVERSE_PASS],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Traced scalar and network learnables gradients",
            location: "crates/runmat-runtime/src/builtins/deep_learning/tests.rs",
        }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
