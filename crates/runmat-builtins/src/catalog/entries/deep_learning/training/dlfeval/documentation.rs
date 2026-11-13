use crate::*;

const CALLBACK: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Function-handle and requested-output bridge",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/deep_learning/training.rs",
    ),
};
const TAPE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "Execution-owned autodiff tape scope",
    target: BuiltinDocumentationLinkTarget::Source(
        "https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/deep_learning/autodiff/tape.rs",
    ),
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("dlfeval"),
    slug: Some("dlfeval"),
    summary: "Evaluate a function handle under a reverse-mode autodiff tape.",
    description: "`dlfeval(fun, args...)` invokes a function handle through RunMat's ordinary function-handle dispatch, forwards the requested output count, and scopes a host autodiff tape to the callback's execution.",
    keywords: &["dlfeval", "deep learning", "automatic differentiation", "function handle"],
    related: &["dlarray", "dlgradient", "dlnetwork", "forward", "feval"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Function and outputs",
            paragraphs: &[
                "The first argument must be a function handle: named, method, bound, external, or closure handles use normal RunMat dispatch. Text function names and callable objects are not dlfeval inputs; use feval for those RunMat forms. A bracketed call forwards its requested output count to the callback, including zero or multiple outputs, without manufacturing a result type.",
                "Arguments are forwarded to the callback in their current class and residency. In particular, integer arguments are not widened merely because dlfeval is called; the callback's own operations determine their admissibility and output class.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Tape lifetime and supported differentiation",
            paragraphs: &[
                "A bounded reverse-mode tape is active only while the callback is being polled or dropped. Nested callbacks keep distinct graphs, and a suspended callback does not expose its tape to another invocation. The callback may call dlgradient on a scalar traced loss to differentiate host dlarray values and supported dlnetwork learnables.",
                "The host tape records dlarray plus, minus, elementwise times and rdivide, matrix mtimes, scalar sum over all elements, and admitted sequential network operations including convolution1d, LSTM/BiLSTM, layer normalization, global average pooling1d, fully connected, ReLU, ELU, and softmax. Complex, sparse, general higher-rank broadcast gradients, custom layers, and resident operations without an admitted adjoint remain outside this subset. A GPU-backed dlarray can be wrapped without download outside a tape, but dlfeval rejects it when tracing would require an unavailable resident kernel.",
            ],
        },
    ],
    examples: &[BuiltinExample {
        id: "square-loss-callback",
        title: "Evaluate a named loss function and return its gradient",
        program: "X = dlarray(single([1 2 3])); [loss, gradients] = dlfeval(@squareLoss, X); assert(isequal(loss.Data, single(14))); assert(isequal(gradients.Data, single([2 4 6])));\nfunction [loss, gradients] = squareLoss(X)\n    squared = X .* X;\n    loss = sum(squared, 'all');\n    gradients = dlgradient(loss, X);\nend",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(strcmp(class(loss.Data), 'single')); assert(strcmp(class(gradients.Data), 'single')); assert(isequal(loss.Data, single(14))); assert(isequal(gradients.Data, single([2 4 6])));",
        },
    }],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Can I pass a function name as text?",
            answer: "No. dlfeval requires a function handle or closure; feval accepts text names in its separate contract.",
        },
        BuiltinDocumentationFaq {
            question: "What happens to requested outputs?",
            answer: "The count is forwarded to the callback. The callback decides the outputs' classes, shapes, and residency; dlfeval does not impose one fixed result type.",
        },
        BuiltinDocumentationFaq {
            question: "Does dlfeval automatically gather GPU-backed dlarray Data?",
            answer: "No. Tracing such Data requires resident tape kernels that are not implemented; RunMat reports an explicit provider-gap error.",
        },
    ],
    links: &[CALLBACK, TAPE, BuiltinDocumentationLink {
        label: "dlgradient",
        target: BuiltinDocumentationLinkTarget::Builtin("dlgradient"),
    }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[CALLBACK, TAPE],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Function handles, requested outputs, typed gradients, and resident rejection",
            location: "crates/runmat-runtime/src/builtins/deep_learning/tests.rs",
        }],
        notes: &[],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
