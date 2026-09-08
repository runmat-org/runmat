mod examples;

use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference,
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Field ordering", paragraphs: &[
        "`S = orderfields(S1)` sorts top-level field names in ASCII order, so uppercase letters precede lowercase letters. The returned structure contains the same field values as `S1`.",
        "The second input may be a structure whose field order is used as a template, a cell or string collection containing every field name once, or a real numeric vector containing every original 1-based field position once.",
    ] },
    BuiltinDocumentationSection { heading: "Permutation output", paragraphs: &[
        "`[S,Pout] = orderfields(...)` returns the original position of every field in the new order. `Pout` is a host double column vector and can be passed to another structure with the same field schema.",
        "The operation applies only to top-level fields. Nested structures retain their own field order.",
    ] },
    BuiltinDocumentationSection { heading: "Represented arrays and resident values", paragraphs: &[
        "RunMat currently represents structure arrays with a cell-backed container. Every element must have the same field set; `orderfields` preserves the container dimensions and applies one order to every element.",
        "Field ordering changes host metadata only. Fixed-width integers retain their exact class and payload, and nested device handles retain provider ownership without a gather or accelerator kernel.",
    ] },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "How are names sorted when no order is supplied?", answer: "RunMat uses ASCII ordering, with uppercase letters before lowercase letters." },
    BuiltinDocumentationFaq { question: "What can the second input contain?", answer: "A reference structure, a cell array of character vectors, a string array, or a real numeric permutation vector." },
    BuiltinDocumentationFaq { question: "Must the field sets match?", answer: "Yes. A reference or name collection must contain exactly the input fields, once each." },
    BuiltinDocumentationFaq { question: "What does Pout contain?", answer: "It is a double column vector of original 1-based field positions in the returned order." },
    BuiltinDocumentationFaq { question: "Does orderfields reorder nested structures?", answer: "No. It changes top-level field order only." },
    BuiltinDocumentationFaq { question: "Does orderfields gather GPU values?", answer: "No. Nested resident handles move with their field entries and remain owned by their provider." },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "struct",
        target: BuiltinDocumentationLinkTarget::Builtin("struct"),
    },
    BuiltinDocumentationLink {
        label: "fieldnames",
        target: BuiltinDocumentationLinkTarget::Builtin("fieldnames"),
    },
    BuiltinDocumentationLink {
        label: "isfield",
        target: BuiltinDocumentationLinkTarget::Builtin("isfield"),
    },
    BuiltinDocumentationLink {
        label: "rmfield",
        target: BuiltinDocumentationLinkTarget::Builtin("rmfield"),
    },
    BuiltinDocumentationLink {
        label: "getfield",
        target: BuiltinDocumentationLinkTarget::Builtin("getfield"),
    },
    BuiltinDocumentationLink {
        label: "setfield",
        target: BuiltinDocumentationLinkTarget::Builtin("setfield"),
    },
];

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/tree/main/crates/runmat-runtime/src/builtins/structs/core/orderfields") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Ordering, validation, output, typed-value, and no-gather behavior", location: "builtins::structs::core::orderfields::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed output facts and argument diagnostics", location: "catalog::entries::structs::core::orderfields::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Executable catalog examples", location: "scripts/runtime/verify-builtin-examples.mjs" },
    ],
    notes: &[],
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("orderfields"), slug: Some("orderfields"),
    summary: "Reorder the top-level fields of a structure or represented structure array.",
    description: "`orderfields` sorts fields or applies a reference, name list, or numeric permutation while preserving values and represented-array dimensions.",
    keywords: &[
        "orderfields",
        "field order",
        "reorder fields",
        "alphabetical",
        "struct",
        "struct array",
        "permutation",
    ],
    related: &["struct", "fieldnames", "isfield", "rmfield", "getfield", "setfield"],
    sections: SECTIONS, examples: examples::EXAMPLES, example_exemption: None, faqs: FAQS,
    links: LINKS, media: &[], evidence: EVIDENCE, introduced: Some("before R2006a"),
    status: Some(BuiltinDocumentationStatus::Stable),
};
