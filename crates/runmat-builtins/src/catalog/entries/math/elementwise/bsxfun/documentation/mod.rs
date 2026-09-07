mod examples;

use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference,
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Singleton expansion",
        paragraphs: &[
            "`C = bsxfun(fun,A,B)` aligns trailing dimensions of `A` and `B`. Corresponding dimensions must be equal or one; singleton dimensions expand to the other extent, including compatible zero-length extents.",
            "The callback receives one scalar from each expanded position and must return one scalar. RunMat collects a numeric, logical, complex, or character result only when every callback result has the same class.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Callable and class behavior",
        paragraphs: &[
            "`fun` may be a builtin, source function, local or nested function, bound function handle, or anonymous function. A scalar text name is available in RunMat compatibility mode and is rejected by the MATLAB compatibility pin.",
            "Input scalar extraction preserves double, single, all eight fixed-width integer classes, logical, complex, and character values. The callback owns its argument admission, conversions, overflow behavior, effects, and output class.",
            "Static analysis queries the typed callable contract with scalarized input facts. A builtin handle therefore receives the same argument-dependent inference as a direct builtin call; unresolved or shadowable names remain dynamic.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Accelerated inputs",
        paragraphs: &[
            "The current implementation gathers provider-resident inputs through their owning providers, executes callbacks on the host, and returns a host value. This is a compatibility gap for GPU callbacks rather than a RunMat extension.",
            "Use direct element-wise operators when provider-native execution and resident output are required. `bsxfun` is a fusion boundary because an arbitrary callback may suspend, throw, or perform other declared effects.",
        ],
    },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Should new code use bsxfun?", answer: "Implicit expansion is clearer for ordinary element-wise operators. bsxfun remains useful for existing code and for binary callbacks that are not written as a direct operator expression." },
    BuiltinDocumentationFaq { question: "Can the callback return an array?", answer: "No. Each invocation must return one scalar, and all invocations must return the same class. Use arrayfun or cellfun for their corresponding mapping contracts." },
    BuiltinDocumentationFaq { question: "How is an empty result typed?", answer: "A typed builtin callback uses its catalog inference contract. A callable whose result cannot be established without execution remains dynamically typed and uses the runtime's compatibility fallback." },
    BuiltinDocumentationFaq { question: "Does bsxfun keep gpuArray input resident?", answer: "Not currently. Inputs are gathered through their exact owner and callback execution produces a host result. Direct supported operators provide the provider-native path." },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "arrayfun",
        target: BuiltinDocumentationLinkTarget::Builtin("arrayfun"),
    },
    BuiltinDocumentationLink {
        label: "cellfun",
        target: BuiltinDocumentationLinkTarget::Builtin("cellfun"),
    },
    BuiltinDocumentationLink {
        label: "plus",
        target: BuiltinDocumentationLinkTarget::Builtin("plus"),
    },
    BuiltinDocumentationLink {
        label: "minus",
        target: BuiltinDocumentationLinkTarget::Builtin("minus"),
    },
    BuiltinDocumentationLink {
        label: "times",
        target: BuiltinDocumentationLinkTarget::Builtin("times"),
    },
];

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/tree/main/crates/runmat-runtime/src/builtins/math/elementwise/bsxfun") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Broadcast, callback, class, empty, and diagnostic behavior", location: "builtins::math::elementwise::bsxfun::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed callback-result and broadcast inference", location: "catalog::entries::math::elementwise::bsxfun::inference_tests" },
    ],
    notes: &[],
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("bsxfun"),
    slug: Some("bsxfun"),
    summary: "Apply a binary function element-wise with singleton expansion.",
    description: "`bsxfun` expands compatible singleton dimensions, calls a binary function once for each result position, and collects uniform scalar results in the expanded shape.",
    keywords: &["bsxfun", "binary function", "singleton expansion", "implicit expansion", "function handle", "broadcasting"],
    related: &["arrayfun", "cellfun", "plus", "minus", "times"],
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
