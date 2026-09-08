mod examples;

use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference,
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Element-wise callback execution",
        paragraphs: &[
            "`B = arrayfun(func,A1,A2,...)` extracts one scalar from each input at every result position and calls `func` with those scalars. Ordinary host arrays must have equal sizes. RunMat compatibility mode also permits host scalar expansion; gpuArray inputs follow compatible-size expansion.",
            "A function handle is the portable callable form. Character-vector and string-scalar names are available in RunMat compatibility mode. Source, local, nested, anonymous, bound, and builtin handles use the same callable-resolution boundary.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Output and error handling",
        paragraphs: &[
            "`UniformOutput` defaults to true. Every invocation must then return one scalar, and the results must have one compatible class. Double, single, logical, complex, and all eight fixed-width integer classes retain their supported representation.",
            "Set `UniformOutput` to false to keep each callback result in a cell. `ErrorHandler` receives an error structure followed by the scalar arguments for the failed position. RunMat currently supports one callback output per invocation.",
            "An empty input produces an empty result with the planned output shape. When the callback's result class is statically known, analysis retains that class even though no callback invocation runs.",
            "Numeric, logical, character, string, and complex arrays are supported. Multiple callback outputs and cell, structure, and user-object inputs are not yet supported by `arrayfun`.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Provider-resident inputs",
        paragraphs: &[
            "Supported builtin callbacks can execute directly when every input is a gpuArray. Other callbacks gather authoritative typed values through the owning provider, execute on the host, and re-upload supported uniform numeric or logical output.",
            "The direct provider path currently covers `sin`, `cos`, `abs`, `exp`, `log`, `sqrt`, `plus`, `minus`, `times`, `rdivide`, and `ldivide`. Other callbacks follow the typed gather path.",
            "The callback makes `arrayfun` a fusion boundary. Nonuniform, character, and currently unsupported complex provider outputs remain on the host. `UniformOutput` and `ErrorHandler` on gpuArray input are RunMat compatibility-mode extensions.",
        ],
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Must I create a gpuArray first?", answer: "No. The planner may select acceleration for suitable work. An explicit gpuArray is useful when source compatibility or deliberate residency matters." },
    BuiltinDocumentationFaq { question: "What if callback results have different classes?", answer: "Use `UniformOutput`, false to retain the values in cells. Uniform output rejects heterogeneous result classes rather than silently choosing a new class." },
    BuiltinDocumentationFaq { question: "Can arrayfun accept character or string arrays?", answer: "Yes. Character elements are passed as scalar character arrays and string elements as string scalars; result collection follows the selected uniform-output policy." },
    BuiltinDocumentationFaq { question: "What happens after a callback error?", answer: "Without an ErrorHandler, the first failure ends the call. With a handler, arrayfun passes error information and the current scalar inputs to that handler." },
    BuiltinDocumentationFaq { question: "How are logical results stored on a GPU?", answer: "The provider uses its logical buffer representation while the result is resident. `gather` restores a RunMat logical array rather than exposing the provider encoding." },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "cellfun",
        target: BuiltinDocumentationLinkTarget::Builtin("cellfun"),
    },
    BuiltinDocumentationLink {
        label: "gpuArray",
        target: BuiltinDocumentationLinkTarget::Builtin("gpuArray"),
    },
    BuiltinDocumentationLink {
        label: "gather",
        target: BuiltinDocumentationLinkTarget::Builtin("gather"),
    },
    BuiltinDocumentationLink {
        label: "bsxfun",
        target: BuiltinDocumentationLinkTarget::Builtin("bsxfun"),
    },
    BuiltinDocumentationLink {
        label: "gpuDevice",
        target: BuiltinDocumentationLinkTarget::Builtin("gpuDevice"),
    },
    BuiltinDocumentationLink {
        label: "gpuInfo",
        target: BuiltinDocumentationLinkTarget::Builtin("gpuInfo"),
    },
    BuiltinDocumentationLink {
        label: "pagefun",
        target: BuiltinDocumentationLinkTarget::Builtin("pagefun"),
    },
];

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/tree/main/crates/runmat-runtime/src/builtins/acceleration/gpu/arrayfun") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Host, callback, output-class, error, and provider behavior", location: "builtins::acceleration::gpu::arrayfun::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed callback, shape, option, effect, and residency inference", location: "catalog::entries::acceleration::arrayfun::inference_tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Executable catalog examples", location: "scripts/runtime/verify-builtin-examples.mjs" },
    ],
    notes: &[],
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("arrayfun"),
    slug: Some("arrayfun"),
    summary: "Apply a scalar function element-wise across array inputs.",
    description: "`arrayfun` invokes a scalar callback at every position of one or more arrays and collects either uniform array output or a cell array of individual results.",
    keywords: &["arrayfun", "gpuArray", "element-wise map", "function handle", "UniformOutput", "ErrorHandler"],
    related: &["cellfun", "gpuArray", "gather", "bsxfun"],
    sections: SECTIONS,
    examples: examples::EXAMPLES,
    example_exemption: None,
    faqs: FAQS,
    links: LINKS,
    media: &[],
    evidence: EVIDENCE,
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
