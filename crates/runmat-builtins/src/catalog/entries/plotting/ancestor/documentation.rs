use crate::*;
mod sections;
mod examples;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("ancestor"),
    summary: "Find a graphics object ancestor with a requested type.",
    description: "`ancestor` returns the closest graphics handle whose object type matches the requested type name. The search includes the input object itself, follows the encoded handle parent chain, and returns an empty handle array when no match exists.",
    slug: Some("ancestor"),
    keywords: &[
        "ancestor",
        "graphics handles",
        "graphics parent",
        "handle ancestry",
        "plotting",
    ],
    related: &[
        "get",
        "isgraphics",
        "gobjects",
    ],
    sections: sections::SECTIONS,
    examples: examples::EXAMPLES,
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Does `ancestor` scan every plot object in the figure?",
            answer: "No. RunMat graphics handles encode or snapshot their own object type and parent location, so `ancestor` builds the parent chain directly from handle metadata.",
        },
        BuiltinDocumentationFaq {
            question: "Are unmatched array entries preserved as placeholders?",
            answer: "No. Unmatched or invalid entries are omitted from array results. If none of the entries has a matching ancestor, `ancestor` returns an empty handle array.",
        },
    ],
    links: &[
        BuiltinDocumentationLink { label: "get", target: BuiltinDocumentationLinkTarget::Builtin("get") },
        BuiltinDocumentationLink { label: "isgraphics", target: BuiltinDocumentationLinkTarget::Builtin("isgraphics") },
        BuiltinDocumentationLink { label: "gobjects", target: BuiltinDocumentationLinkTarget::Builtin("gobjects") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "`crates/runmat-runtime/src/builtins/plotting/ops/ancestor.rs`", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/ancestor.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "builtins::plotting::ancestor::tests", location: "builtins::plotting::ancestor::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "runmat-vm plotting::ancestor_dispatches_graphics_parent_queries", location: "runmat-vm plotting::ancestor_dispatches_graphics_parent_queries" },
        ],
        notes: &[
            "Source-only migration: listed preexisting test locations and independently asserted examples are not claims of fresh execution. BrowserGraphics needs actual graphics-host qualification; headless semantic tests do not prove Desktop presentation.",
            "Legacy GPU/fusion hints are normalized to the canonical typed host/fusion-boundary contract. No optional feature gate is added; no provider precision, native OpenGL availability, or finite allocation completion is inferred from prose.",
        ],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
