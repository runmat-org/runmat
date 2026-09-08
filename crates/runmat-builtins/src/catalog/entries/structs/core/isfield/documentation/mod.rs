mod examples;

use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference,
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Queries and results", paragraphs: &[
        "`isfield(S,name)` compares field names exactly and case-sensitively. A character row or scalar string produces one logical scalar. A string array or cell collection of scalar text produces a logical array with the query collection's shape.",
        "If `S` is not a structure, the result is false with the same scalar or array form. Unsupported field-name values reject instead of converting numbers or other values to text.",
    ] },
    BuiltinDocumentationSection { heading: "Represented structure arrays", paragraphs: &[
        "RunMat currently stores structure arrays in a cell-backed representation. A queried name is present only when every represented element defines it. An empty represented array has no retained schema and returns false for every name.",
        "Because this backing is shared with ordinary cells, a nonempty cell containing only scalar structures is interpreted as a represented structure array. Other cell values are non-structure inputs and produce false results.",
    ] },
    BuiltinDocumentationSection { heading: "Residency and execution", paragraphs: &[
        "`isfield` reads outer host-side metadata only. It does not inspect, copy, or gather values stored in a structure, launches no accelerator kernel, and forms a fusion boundary.",
    ] },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Which field-name inputs are accepted?", answer: "Character rows, scalar strings, string arrays, and cell arrays whose elements are character rows or scalar strings." },
    BuiltinDocumentationFaq { question: "What happens when the first input is not a structure?", answer: "The result is false, or a same-shaped logical array of false values. Use `isstruct` when the value's class must be checked separately." },
    BuiltinDocumentationFaq { question: "How are represented structure arrays queried?", answer: "A name is true only when every represented element contains it." },
    BuiltinDocumentationFaq { question: "What happens for an empty represented array?", answer: "RunMat's current representation retains no field schema after all elements are removed, so every query is false." },
    BuiltinDocumentationFaq { question: "Are comparisons case-sensitive?", answer: "Yes. Names must match exactly, including case." },
    BuiltinDocumentationFaq { question: "Does isfield gather GPU data?", answer: "No. Nested resident values remain with their provider because only outer metadata is read." },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "fieldnames",
        target: BuiltinDocumentationLinkTarget::Builtin("fieldnames"),
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
        label: "orderfields",
        target: BuiltinDocumentationLinkTarget::Builtin("orderfields"),
    },
    BuiltinDocumentationLink {
        label: "rmfield",
        target: BuiltinDocumentationLinkTarget::Builtin("rmfield"),
    },
];

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/tree/main/crates/runmat-runtime/src/builtins/structs/core/isfield") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Scalar, collection, represented-array, rejection, and no-gather behavior", location: "builtins::structs::core::isfield::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed result shapes and field-name diagnostics", location: "catalog::entries::structs::core::isfield::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Executable catalog examples", location: "scripts/runtime/verify-builtin-examples.mjs" },
    ],
    notes: &[],
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("isfield"), slug: Some("isfield"),
    summary: "Test whether structure metadata contains one or more field names.",
    description: "`isfield` returns scalar or same-shaped logical results for exact, case-sensitive field-name queries without reading stored values.",
    keywords: &["isfield", "struct", "struct array", "field existence", "metadata"],
    related: &["fieldnames", "struct", "getfield", "setfield", "orderfields", "rmfield"],
    sections: SECTIONS, examples: examples::EXAMPLES, example_exemption: None, faqs: FAQS,
    links: LINKS, media: &[], evidence: EVIDENCE, introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
