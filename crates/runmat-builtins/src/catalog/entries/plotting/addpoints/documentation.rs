use crate::*;
mod sections;
mod examples;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("addpoints"),
    summary: "Append 2-D or 3-D coordinates to an animated line.",
    description: "`addpoints` appends coordinates to a handle returned by `animatedline`. It updates the underlying line geometry and honors the handle's `MaximumNumPoints` limit by trimming oldest samples.",
    slug: Some("addpoints"),
    keywords: &[
        "addpoints",
        "animatedline",
        "animated line",
        "streaming plot",
        "live plot",
        "matlab plotting",
    ],
    related: &[
        "animatedline",
        "line",
        "get",
        "set",
    ],
    sections: sections::SECTIONS,
    examples: examples::EXAMPLES,
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Can addpoints be used with ordinary line handles?",
            answer: "No. The first argument must be a handle returned by `animatedline`; ordinary `line` and `plot` handles should be updated with `set(h,'XData',...)` and related properties.",
        },
        BuiltinDocumentationFaq {
            question: "Does addpoints clear the axes?",
            answer: "No. It mutates the existing animated-line object in place.",
        },
        BuiltinDocumentationFaq {
            question: "Can I promote a marker-styled 2-D animated line to 3-D?",
            answer: "No. RunMat rejects that promotion so marker metadata is not silently lost. Create a 3-D animated line without marker styling instead.",
        },
    ],
    links: &[
        BuiltinDocumentationLink { label: "animatedline", target: BuiltinDocumentationLinkTarget::Builtin("animatedline") },
        BuiltinDocumentationLink { label: "line", target: BuiltinDocumentationLinkTarget::Builtin("line") },
        BuiltinDocumentationLink { label: "get", target: BuiltinDocumentationLinkTarget::Builtin("get") },
        BuiltinDocumentationLink { label: "set", target: BuiltinDocumentationLinkTarget::Builtin("set") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "`crates/runmat-runtime/src/builtins/plotting/ops/addpoints.rs`", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/addpoints.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "builtins::plotting::addpoints::tests", location: "builtins::plotting::addpoints::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "crates/runmat-vm/tests/plotting.rs", location: "crates/runmat-vm/tests/plotting.rs" },
        ],
        notes: &[
            "Source-only migration: listed preexisting test locations and independently asserted examples are not claims of fresh execution. BrowserGraphics needs actual graphics-host qualification; headless semantic tests do not prove Desktop presentation.",
            "Legacy GPU/fusion hints are normalized to the canonical typed host/fusion-boundary contract. No optional feature gate is added; no provider precision, native OpenGL availability, or finite allocation completion is inferred from prose.",
        ],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
