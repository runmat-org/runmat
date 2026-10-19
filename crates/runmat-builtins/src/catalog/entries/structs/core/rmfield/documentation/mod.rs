mod examples;

use crate::{
    BuiltinDocumentation, BuiltinDocumentationAuthority, BuiltinDocumentationEvidence,
    BuiltinDocumentationFaq, BuiltinDocumentationLink, BuiltinDocumentationLinkTarget,
    BuiltinDocumentationSection, BuiltinDocumentationStatus, BuiltinEvidenceKind,
    BuiltinEvidenceReference,
};

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Field removal", paragraphs: &[
        "`S2 = rmfield(S,fields)` returns a structure with every requested top-level field removed. `S` may be a scalar structure or structure array. `fields` may be a character row, string value, string array, or cell array of character rows and scalar strings.",
        "Every distinct requested name must exist on every structure element. Names are exact and case-sensitive. Duplicate names are removed once; an empty collection removes nothing, while an empty individual name is invalid.",
    ] },
    BuiltinDocumentationSection { heading: "Compatibility and structure arrays", paragraphs: &[
        "MATLAB compatibility mode accepts one field-name argument. RunMat mode also accepts separate variadic field-name arguments and flattens them in call order. Use one string array or cell array when the source must remain within the MATLAB form.",
        "Removing fields preserves structure-array dimensions and updates the shared ordered schema. Ordinary cell arrays are not structure arrays, regardless of their contents.",
    ] },
    BuiltinDocumentationSection { heading: "Value and residency preservation", paragraphs: &[
        "The input value is not mutated. RunMat consumes or copies the outer value according to ordinary value semantics and removes only field-map entries. Retained values preserve their class, payload, and provider ownership.",
        "`rmfield` launches no accelerator kernel and does not gather nested resident values. It runs as a host metadata operation and forms a fusion boundary.",
    ] },
];

const FAQS: &[BuiltinDocumentationFaq] = &[
    BuiltinDocumentationFaq { question: "Does rmfield modify its input in place?", answer: "No. It returns a value with the selected fields removed; the caller's original value retains ordinary value semantics." },
    BuiltinDocumentationFaq { question: "Which field-name inputs are accepted?", answer: "One character row, string value, string array, or cell array of character rows and scalar strings. RunMat mode additionally accepts multiple separate field-name arguments." },
    BuiltinDocumentationFaq { question: "What happens when a field is missing?", answer: "The call fails with a missing-field error. For a structure array, every requested name must exist on every element." },
    BuiltinDocumentationFaq { question: "Are duplicate or empty names accepted?", answer: "Duplicate names are removed once. Empty individual names are invalid; an empty string or cell collection requests no removals." },
    BuiltinDocumentationFaq { question: "Can rmfield remove nested fields?", answer: "No. It removes top-level fields only." },
    BuiltinDocumentationFaq { question: "Does rmfield gather GPU data?", answer: "No. Retained resident field values remain with their owning provider because only outer metadata changes." },
];

const LINKS: &[BuiltinDocumentationLink] = &[
    BuiltinDocumentationLink {
        label: "fieldnames",
        target: BuiltinDocumentationLinkTarget::Builtin("fieldnames"),
    },
    BuiltinDocumentationLink {
        label: "isfield",
        target: BuiltinDocumentationLinkTarget::Builtin("isfield"),
    },
    BuiltinDocumentationLink {
        label: "setfield",
        target: BuiltinDocumentationLinkTarget::Builtin("setfield"),
    },
    BuiltinDocumentationLink {
        label: "struct",
        target: BuiltinDocumentationLinkTarget::Builtin("struct"),
    },
    BuiltinDocumentationLink {
        label: "orderfields",
        target: BuiltinDocumentationLinkTarget::Builtin("orderfields"),
    },
    BuiltinDocumentationLink {
        label: "getfield",
        target: BuiltinDocumentationLinkTarget::Builtin("getfield"),
    },
];

const EVIDENCE: BuiltinDocumentationEvidence = BuiltinDocumentationEvidence {
    implementation: &[BuiltinDocumentationLink { label: "Runtime implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/tree/main/crates/runmat-runtime/src/builtins/structs/core/rmfield") }],
    verification: &[
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Scalar, structure-array, validation, compatibility, typed-value, and no-gather behavior", location: "builtins::structs::core::rmfield::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Typed result facts and input diagnostics", location: "catalog::entries::structs::core::rmfield::tests" },
        BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Executable catalog examples", location: "scripts/runtime/verify-builtin-examples.mjs" },
    ],
    notes: &[],
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("rmfield"), slug: Some("rmfield"),
    summary: "Remove one or more top-level fields from a structure or structure array.",
    description: "`rmfield` returns a structure with exact, case-sensitive field names removed while preserving structure-array shape and retained values.",
    keywords: &["rmfield", "remove field", "struct", "struct array", "metadata"],
    related: &["fieldnames", "isfield", "setfield", "struct", "orderfields", "getfield"],
    sections: SECTIONS, examples: examples::EXAMPLES, example_exemption: None, faqs: FAQS,
    links: LINKS, media: &[], evidence: EVIDENCE, introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
