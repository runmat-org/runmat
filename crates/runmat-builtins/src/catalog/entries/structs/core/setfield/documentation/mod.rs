mod examples;

use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference,
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Field assignment", paragraphs: &["`S2 = setfield(S,field,value)` returns the updated value. Additional field names traverse nested structures; absent fields and absent intermediate structures are created where the path permits them.", "The input follows ordinary value semantics. Scalar structures and value objects produce updated values; a valid handle object updates its referenced instance and returns the same handle identity."] },
    BuiltinDocumentationSection { heading: "Structure arrays and indexed fields", paragraphs: &["Place an index cell before the first field to select one structure-array element. Place one after a field name to apply ordinary parenthesis assignment to that field's contents. Intermediate selections must identify one value; the final selector may update several elements. Every index is one-based.", "Indexed assignment observes the selected container's class and shape rules. A numeric or logical final selector supports multi-element assignment when the right-hand side is scalar or has a compatible shape. Parenthesis assignment into a cell field requires a cell-array right-hand side and preserves the cell container."] },
    BuiltinDocumentationSection { heading: "Objects and resident values", paragraphs: &["RunMat mode extends `setfield` to supported object and handle values. Property assignment observes declared access, static, dynamic, and dependent-property behavior.", "Replacing a field with a resident value preserves that handle without provider work. Indexing into a resident field is a separately gated RunMat extension: the owning provider gathers the target for host mutation, and the updated field becomes host-resident."] },
];
const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Does setfield modify its input?", answer: "Structures and value objects return updated values under ordinary value semantics. Handle objects update their referenced instance." },
    BuiltinDocumentationFaq { question: "Can setfield create nested fields?", answer: "Yes. Missing intermediate structures are created along an otherwise valid nested field path." },
    BuiltinDocumentationFaq { question: "How do I update a structure-array element?", answer: "Put a cell array of one-based indices before the first field name." },
    BuiltinDocumentationFaq { question: "Does every GPU field get gathered?", answer: "No. Direct replacement preserves untouched and newly assigned resident handles. Only indexed mutation of the resident field gathers that target." },
    BuiltinDocumentationFaq { question: "Can I continue a path through a selected cell?", answer: "No. Selector cells express parenthesis indexing, so a selected cell remains a cell array. Use ordinary brace indexing outside `setfield` when you need its contents." },
    BuiltinDocumentationFaq { question: "How are object properties assigned?", answer: "RunMat mode supports value and handle object properties as an explicit extension and observes static, private, dynamic, and dependent-property rules." },
];
const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "getfield",
        target: BuiltinDocumentationLinkTarget::Builtin("getfield"),
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
        label: "gpuArray",
        target: BuiltinDocumentationLinkTarget::Builtin("gpuArray"),
    },
    BuiltinDocumentationLink {
        label: "gather",
        target: BuiltinDocumentationLinkTarget::Builtin("gather"),
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
const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence { implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/tree/main/crates/runmat-runtime/src/builtins/structs/core/setfield") }], verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Structure, object, indexed, compatibility, typed-value, and resident-value behavior", location: "builtins::structs::core::setfield::tests" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed member-path inference", location: "catalog::entries::structs::core::setfield::tests" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Executable catalog examples", location: "scripts/runtime/verify-builtin-examples.mjs" }], notes: &[] };
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation { authority: BuiltinDocumentationAuthority::Catalog, title: Some("setfield"), slug: Some("setfield"), summary: "Assign a field, nested path, or indexed field value.", description: "`setfield` provides functional field assignment for scalar structures, structure arrays, and supported RunMat object-family values.", keywords: &["setfield", "struct", "assignment", "struct array", "field assignment", "object property"], related: &["getfield", "fieldnames", "isfield", "struct", "gpuArray", "gather", "orderfields", "rmfield"], sections: SECTIONS, examples: examples::EXAMPLES, example_exemption: None, faqs: FAQS, links: LINKS, media: &[], evidence: EVIDENCE, introduced: None, status: Some(BuiltinDocumentationStatus::Stable) };
