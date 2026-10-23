use crate::*;
mod sections;
mod examples;
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("copyobj"),
    summary: "Copy plot-child graphics objects to a compatible axes parent.",
    description: "`copyobj` duplicates supported graphics objects and returns fresh graphics handles. MATLAB's public inputs are graphics objects rather than integer data. RunMat currently supports plot-child object copies, including line, scatter, bar, stairs, stem, area, quiver, image, heatmap, binscatter, surface, patch, line3, scatter3, contour, fcontour, fsurf, constant-line, pie, and histogram-backed plot children.",
    slug: Some("copyobj"),
    keywords: &[
        "copyobj",
        "copy graphics object",
        "graphics handle",
        "axes",
        "plotting",
    ],
    related: &[
        "plot",
        "axes",
        "isgraphics",
        "get",
        "set",
    ],
    sections: sections::SECTIONS,
    examples: examples::EXAMPLES,
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq {
            question: "Does `copyobj` copy figures or axes?",
            answer: "Not yet. This implementation covers plot-child graphics objects and reports a deterministic invalid-argument error for figure, axes, text, legend, ruler, and root deep-copy requests.",
        },
        BuiltinDocumentationFaq {
            question: "Does copying share mutable plot properties with the source?",
            answer: "No. The copied plot receives a fresh handle and independent plot state.",
        },
        BuiltinDocumentationFaq {
            question: "Does `copyobj` launch GPU kernels?",
            answer: "No. It operates on host-side graphics metadata. Plot geometry already stored in graphics state is cloned without provider dispatch.",
        },
    ],
    links: &[
        BuiltinDocumentationLink { label: "plot", target: BuiltinDocumentationLinkTarget::Builtin("plot") },
        BuiltinDocumentationLink { label: "axes", target: BuiltinDocumentationLinkTarget::Builtin("axes") },
        BuiltinDocumentationLink { label: "isgraphics", target: BuiltinDocumentationLinkTarget::Builtin("isgraphics") },
        BuiltinDocumentationLink { label: "get", target: BuiltinDocumentationLinkTarget::Builtin("get") },
        BuiltinDocumentationLink { label: "set", target: BuiltinDocumentationLinkTarget::Builtin("set") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "`crates/runmat-runtime/src/builtins/plotting/ops/copyobj.rs`", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/copyobj.rs") },
        ],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "builtins::plotting::copyobj::tests", location: "builtins::plotting::copyobj::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "runmat-vm plotting::copyobj_dispatches_plot_child_copies", location: "runmat-vm plotting::copyobj_dispatches_plot_child_copies" },
        ],
        notes: &[
            "Source-only migration: listed preexisting test locations and independently asserted examples are not claims of fresh execution. BrowserGraphics needs actual graphics-host qualification; headless semantic tests do not prove Desktop presentation.",
            "Legacy GPU/fusion hints are normalized to the canonical typed host/fusion-boundary contract. No optional feature gate is added; no provider precision, native OpenGL availability, or finite allocation completion is inferred from prose.",
        ],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
