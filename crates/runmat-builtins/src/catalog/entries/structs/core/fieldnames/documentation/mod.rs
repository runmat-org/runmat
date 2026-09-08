mod examples;

use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference,
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Structure field order", paragraphs: &[
        "`fieldnames(S)` returns an N-by-1 cell array of character rows. A scalar structure retains insertion order and comparisons remain case-sensitive.",
        "RunMat currently represents structure arrays with cell-backed storage. For that representation, every element must be a scalar structure and the result is the sorted union of represented field names. Empty scalar structures and empty represented arrays return a 0-by-1 cell array.",
    ] },
    BuiltinDocumentationSection { heading: "Object-family extension", paragraphs: &[
        "In RunMat compatibility mode, value objects, handle objects, and listeners expose the property metadata currently represented by the runtime. Class-declared nonstatic properties, inherited properties, dynamic instance properties, and valid handle-target metadata are merged into a deterministic result.",
        "Object-family introspection is a RunMat extension rather than a claim of complete MATLAB object-property policy. MATLAB compatibility mode rejects these object inputs before inspecting them.",
    ] },
    BuiltinDocumentationSection { heading: "Residency and execution", paragraphs: &[
        "`fieldnames` reads host-side metadata only. Values nested in a structure or object are not evaluated or gathered, so resident tensor handles remain owned by their provider. The operation launches no accelerator kernel and is a fusion boundary.",
    ] },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "What does fieldnames return?", answer: "An N-by-1 cell array whose elements are character-row field names." },
    BuiltinDocumentationFaq { question: "Are field names sorted?", answer: "Scalar structures retain insertion order. RunMat's current cell-backed structure-array representation returns a sorted union, and supported object families use deterministic property ordering." },
    BuiltinDocumentationFaq { question: "Can fieldnames inspect structure arrays?", answer: "Yes, for RunMat's current cell-backed representation when every element is a scalar structure. A cell containing only scalar structures is therefore interpreted as a represented structure array. Empty arrays return an empty column cell array." },
    BuiltinDocumentationFaq { question: "Can fieldnames inspect objects?", answer: "RunMat compatibility mode supports the currently represented value, handle, and listener properties as an explicit extension. MATLAB compatibility mode rejects that extension." },
    BuiltinDocumentationFaq { question: "Does fieldnames gather GPU data?", answer: "No. It inspects metadata and leaves nested provider-resident values untouched." },
    BuiltinDocumentationFaq { question: "Are unsupported inputs converted?", answer: "No. Numeric, logical, text, and resident array inputs reject without conversion." },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "isfield",
        target: BuiltinDocumentationLinkTarget::Builtin("isfield"),
    },
    BuiltinDocumentationLink {
        label: "orderfields",
        target: BuiltinDocumentationLinkTarget::Builtin("orderfields"),
    },
    BuiltinDocumentationLink {
        label: "struct",
        target: BuiltinDocumentationLinkTarget::Builtin("struct"),
    },
    BuiltinDocumentationLink {
        label: "getfield",
        target: BuiltinDocumentationLinkTarget::Builtin("getfield"),
    },
    BuiltinDocumentationLink {
        label: "setfield",
        target: BuiltinDocumentationLinkTarget::Builtin("setfield"),
    },
    BuiltinDocumentationLink {
        label: "rmfield",
        target: BuiltinDocumentationLinkTarget::Builtin("rmfield"),
    },
    BuiltinDocumentationLink {
        label: "class",
        target: BuiltinDocumentationLinkTarget::Builtin("class"),
    },
];

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/tree/main/crates/runmat-runtime/src/builtins/structs/core/fieldnames") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Structure, object-family, compatibility, and no-gather behavior", location: "builtins::structs::core::fieldnames::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed output and input diagnostics", location: "catalog::entries::structs::core::fieldnames::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Executable catalog examples", location: "scripts/runtime/verify-builtin-examples.mjs" },
    ],
    notes: &[],
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("fieldnames"), slug: Some("fieldnames"),
    summary: "List structure fields and supported RunMat object properties.",
    description: "`fieldnames` returns a column cell array of character-row names without reading or gathering stored values.",
    keywords: &["fieldnames", "struct", "structure", "struct array", "fields", "object", "handle", "properties", "introspection"],
    related: &["struct", "isfield", "getfield", "setfield", "rmfield", "class", "orderfields"],
    sections: SECTIONS, examples: examples::EXAMPLES, example_exemption: None, faqs: FAQS,
    links: LINKS, media: &[], evidence: EVIDENCE, introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
