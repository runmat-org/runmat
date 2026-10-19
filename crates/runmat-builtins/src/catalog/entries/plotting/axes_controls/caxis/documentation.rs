use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("caxis"),
    slug: Some("caxis"),
    summary: "Query or set colormap limits using the legacy name for clim.",
    description: "`caxis` is the legacy name for `clim` and shares its color-limit query, setter, mode, compatibility, and host graphics-state contracts.",
    keywords: &["caxis", "clim", "color limits", "colormap", "matlab caxis"],
    related: &["clim", "colormap", "colorbar"],
    sections: &[
        BuiltinDocumentationSection { heading: "Assigned setter outputs", paragraphs: &[
            "The MATLAB-compatible setter form has no assigned output. RunMat mode preserves the existing one-output setter result as a gated extension; strict compatibility rejects that request before changing graphics state.",
        ] },
        BuiltinDocumentationSection { heading: "Behavior and compatibility", paragraphs: &[
            "`caxis()` returns the displayed color limits as a two-element double row vector, including automatic scalar-color bounds when available.",
            "`caxis([cmin cmax])` accepts a two-element single or double vector with finite values and `cmin < cmax`.",
            "The `auto` token restores automatic color scaling; `manual` freezes the currently available scalar color extent. Empty or non-mapped scalar plots use the renderer's default range; an unavailable scalar extent cannot be frozen. `tight` is not a color-limit mode.",
            "All eight typed integer limit classes are the same RunMat-only extension declared by canonical `clim`, with compatibility rejection before conversion and one deliberate host double graphics boundary when enabled.",
            "Resident arguments reject without provider dispatch.",
            "An existing axes handle can be supplied first for query or setter forms without selecting it as the current axes. Multi-target color-limit forms and non-surface color contributors remain graphics-surface gaps.",
        ] },
        BuiltinDocumentationSection { heading: "Placement", paragraphs: &[
            "The control updates host graphics state. Resident control operands reject before provider access; existing resident plot data retains its rendering owner.",
            "The whole-call allocation, graphics presentation, and callback execution routes still require integrated qualification.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "example-1", title: "Use the legacy alias to set color limits", program: "imagesc([1 2; 3 4]);\ncaxis([1 4]);", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(isequal(caxis(),[1 4]));" } },
    ],
    example_exemption: None,
    faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "clim", target: BuiltinDocumentationLinkTarget::Builtin("clim") },
        BuiltinDocumentationLink { label: "colormap", target: BuiltinDocumentationLinkTarget::Builtin("colormap") },
        BuiltinDocumentationLink { label: "colorbar", target: BuiltinDocumentationLinkTarget::Builtin("colorbar") },
        BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/caxis.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Shared admitted color-limit controls", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/common/color_limits.rs") },
            BuiltinDocumentationLink { label: "Current displayed color-range owner", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/core/state/axes_modes.rs") },
        ],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "caxis state and validation tests", location: "builtins::plotting::caxis::tests" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Retained integration test source (execution unqualified)", location: "crates/runmat-vm/tests/plotting.rs" }],
        notes: &["Legacy behavior, examples, and links are retained. Native and browser graphics execution remains subject to integration gates."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
