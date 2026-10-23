use crate::*;
mod sections;
mod examples;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("opengl"),
    summary: "Report deterministic RunMat renderer compatibility information.",
    description: "`opengl` provides MATLAB-compatible renderer status and information commands for scripts that probe or select graphics rendering modes. RunMat maps these calls onto the `runmat-plot` renderer instead of exposing a process-global native OpenGL switch.",
    slug: Some("opengl"),
    keywords: &[
        "opengl",
        "renderer",
        "graphics",
        "plotting",
    ],
    related: &[
        "figure",
        "drawnow",
    ],
    sections: sections::SECTIONS,
    examples: examples::EXAMPLES,
    example_exemption: None,
    faqs: &[
    ],
    links: &[
        BuiltinDocumentationLink { label: "figure", target: BuiltinDocumentationLinkTarget::Builtin("figure") },
        BuiltinDocumentationLink { label: "drawnow", target: BuiltinDocumentationLinkTarget::Builtin("drawnow") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "crates/runmat-runtime/src/builtins/plotting/ops/opengl.rs", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/opengl.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "builtins::plotting::opengl::tests", location: "builtins::plotting::opengl::tests" },
        ],
        notes: &[
            "Source-only migration: listed preexisting test locations and independently asserted examples are not claims of fresh execution. BrowserGraphics needs actual graphics-host qualification; headless semantic tests do not prove Desktop presentation.",
            "Legacy GPU/fusion hints are normalized to the canonical typed host/fusion-boundary contract. No optional feature gate is added; no provider precision, native OpenGL availability, or finite allocation completion is inferred from prose.",
        ],
    },
    introduced: Some("0.5.3"),
    status: Some(BuiltinDocumentationStatus::Partial),
};
