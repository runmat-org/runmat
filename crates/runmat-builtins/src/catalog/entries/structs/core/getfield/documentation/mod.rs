mod examples;

use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference,
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Field paths", paragraphs: &["`value = getfield(S,field)` reads a named field. Additional names traverse nested structures, so `getfield(S,'a','b')` is the functional form of `S.a.b`.", "Field names are exact and case-sensitive scalar text. A missing member or a path that reaches a non-structure value produces an error."] },
    BuiltinDocumentationSection { heading: "Structure arrays and indexed fields", paragraphs: &["Place an index cell before the first field to select one structure-array element: `getfield(S,{row,col},'field')`. Place one after a field name to apply ordinary parenthesis indexing to that field's contents. Intermediate selections must identify one value; the final selector may return a subarray. Every index is one-based.", "Numeric vectors and logical arrays select subarrays under the indexed value's class and shape rules. Parenthesis indexing of a cell field returns a cell array; it does not extract cell contents. RunMat mode additionally accepts textual positive indices and `end` inside selector cells."] },
    BuiltinDocumentationSection { heading: "Objects and resident values", paragraphs: &["RunMat mode extends `getfield` to supported object, handle, listener, and exception values. Property access observes declared access rules and invokes an available dependent-property getter.", "Reading a field directly preserves the stored value, including an accelerator-resident handle. Indexing a resident field is a separately gated RunMat extension that gathers authoritative storage through its owning provider before host indexing."] },
];
const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "How do I access a nested field?", answer: "List every field in order, as in `getfield(S,'parent','child')`." },
    BuiltinDocumentationFaq { question: "How do I select a structure-array element?", answer: "Put a cell array of one-based indices before the first field name." },
    BuiltinDocumentationFaq { question: "Can I index the value stored in a field?", answer: "Yes. Put a numeric or logical selector cell immediately after that field name." },
    BuiltinDocumentationFaq { question: "Does direct access gather GPU data?", answer: "No. The stored resident handle is returned unchanged. Indexing that value requires the RunMat indexed-resident extension and a provider-owned gather." },
    BuiltinDocumentationFaq { question: "Do dependent properties run getter methods?", answer: "In RunMat mode, supported dependent properties invoke their getter when one is available; object-family access is an explicit extension." },
    BuiltinDocumentationFaq { question: "What happens for an invalid handle?", answer: "The RunMat object-family extension reports an invalid-handle error before attempting property access." },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "setfield",
        target: BuiltinDocumentationLinkTarget::Builtin("setfield"),
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
        label: "struct",
        target: BuiltinDocumentationLinkTarget::Builtin("struct"),
    },
    BuiltinDocumentationLink {
        label: "class",
        target: BuiltinDocumentationLinkTarget::Builtin("class"),
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
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/tree/main/crates/runmat-runtime/src/builtins/structs/core/getfield") }], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Structure, object, indexed, compatibility, and resident-value behavior", location: "builtins::structs::core::getfield::tests" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed member-path inference", location: "catalog::entries::structs::core::getfield::tests" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Executable catalog examples", location: "scripts/runtime/verify-builtin-examples.mjs" }], notes: &[] };
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation { authority: BuiltinDocumentationAuthority::Catalog, title: Some("getfield"), slug: Some("getfield"), summary: "Read a field, nested path, or indexed field value.", description: "`getfield` provides functional field access for scalar structures, structure arrays, and supported RunMat object-family values.", keywords: &["getfield", "struct", "struct array", "field access", "object property", "metadata"], related: &["setfield", "fieldnames", "isfield", "struct", "class", "orderfields", "rmfield"], sections: SECTIONS, examples: examples::EXAMPLES, example_exemption: None, faqs: FAQS, links: LINKS, media: &[], evidence: EVIDENCE, introduced: None, status: Some(BuiltinDocumentationStatus::Stable) };
