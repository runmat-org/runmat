use crate::*;
mod sections;
mod examples;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("gobjects"),
    summary: "Create an array of invalid graphics-handle placeholders.",
    description: "`gobjects` creates arrays suitable for preallocating graphics handles. RunMat represents graphics handles as numeric values, so placeholders are stored as invalid numeric handle sentinels that can later be overwritten by real plotting handles.",
    slug: Some("gobjects"),
    keywords: &[
        "gobjects",
        "graphics objects",
        "graphics handles",
        "handle preallocation",
        "plotting",
    ],
    related: &[
        "isgraphics",
        "ishandle",
        "plot",
    ],
    sections: sections::SECTIONS,
    examples: examples::EXAMPLES,
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Are `gobjects` placeholders valid graphics handles?",
            answer: "No. They are invalid placeholders used for preallocation until a real graphics handle is assigned into the array.",
        },
        BuiltinDocumentationFaq {
            question: "Does `gobjects` create figures or axes?",
            answer: "No. It only allocates placeholder handle values; plotting functions create the actual graphics objects.",
        },
    ],
    links: &[
        BuiltinDocumentationLink { label: "isgraphics", target: BuiltinDocumentationLinkTarget::Builtin("isgraphics") },
        BuiltinDocumentationLink { label: "ishandle", target: BuiltinDocumentationLinkTarget::Builtin("ishandle") },
        BuiltinDocumentationLink { label: "plot", target: BuiltinDocumentationLinkTarget::Builtin("plot") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "`crates/runmat-runtime/src/builtins/plotting/ops/gobjects.rs`", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/gobjects.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "builtins::plotting::gobjects::tests", location: "builtins::plotting::gobjects::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "runmat-vm plotting::gobjects_preallocates_assignable_graphics_handle_arrays", location: "runmat-vm plotting::gobjects_preallocates_assignable_graphics_handle_arrays" },
        ],
        notes: &[
            "Source-only migration: listed preexisting test locations and independently asserted examples are not claims of fresh execution. BrowserGraphics needs actual graphics-host qualification; headless semantic tests do not prove Desktop presentation.",
            "Legacy GPU/fusion hints are normalized to the canonical typed host/fusion-boundary contract. No optional feature gate is added; no provider precision, native OpenGL availability, or finite allocation completion is inferred from prose.",
        ],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
