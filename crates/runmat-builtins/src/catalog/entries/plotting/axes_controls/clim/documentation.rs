use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("clim"),
    slug: Some("clim"),
    summary: "Query or set colormap limits.",
    description: "`clim` queries or configures the color limits used to map scaled chart data through the active colormap. `caxis` is its legacy alias.",
    keywords: &["clim", "caxis", "color limits", "colormap", "matlab clim"],
    related: &["caxis", "colormap", "colorbar"],
    sections: &[
        BuiltinDocumentationSection { heading: "Assigned setter outputs", paragraphs: &[
            "The MATLAB-compatible setter form has no assigned output. RunMat mode preserves the existing one-output setter result as a gated extension; strict compatibility rejects that request before changing graphics state.",
        ] },
        BuiltinDocumentationSection { heading: "Behavior and compatibility", paragraphs: &[
            "`clim()` returns the displayed color limits as a two-element double row vector, including automatic scalar-color bounds when available.",
            "`clim([cmin cmax])` accepts a two-element single or double vector with finite values and `cmin < cmax`.",
            "The `auto` token restores automatic color scaling; `manual` freezes the currently available scalar color extent. Empty or non-mapped scalar plots use the renderer's default range; an unavailable scalar extent cannot be frozen. `tight` is not a color-limit mode.",
            "All eight typed integer limit classes are a RunMat-only extension. MATLAB-compatible mode rejects them before conversion; RunMat mode validates authoritative storage before one deliberate conversion into host double graphics metadata.",
            "`caxis` shares this exact integer capability and compatibility gate as an audited alias.",
            "Resident arguments reject without provider dispatch because color limits are host graphics controls and no interactive gpuArray input is documented.",
            "An existing axes handle can be supplied first for query or setter forms without selecting it as the current axes. Axes-handle arrays and color contributors outside the shared scalar-color range owner remain unsupported.",
        ] },
        BuiltinDocumentationSection { heading: "Placement", paragraphs: &[
            "The control updates host graphics state. Resident control operands reject before provider access; existing resident plot data retains its rendering owner.",
            "The whole-call allocation, graphics presentation, and callback execution routes still require integrated qualification.",
        ] },
    ],
    examples: &[
        BuiltinExample { id: "example-1", title: "Set and query color limits", program: "imagesc([1 2; 3 4]);\nclim([1 4]);\nlimits = clim();", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(isequal(limits,[1 4]));" } },
        BuiltinExample { id: "example-2", title: "Return to automatic color limits", program: "clim('auto');", display_output: None, compatibility: BuiltinExampleCompatibility::Matlab, harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None, requirements: BuiltinExampleRequirements::NONE, verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "limits = clim(); assert(numel(limits)==2 && all(isfinite(limits)) && limits(1)<limits(2));" } },
    ],
    example_exemption: None,
    faqs: &[],
    links: &[
        BuiltinDocumentationLink { label: "caxis", target: BuiltinDocumentationLinkTarget::Builtin("caxis") },
        BuiltinDocumentationLink { label: "colormap", target: BuiltinDocumentationLinkTarget::Builtin("colormap") },
        BuiltinDocumentationLink { label: "colorbar", target: BuiltinDocumentationLinkTarget::Builtin("colorbar") },
        BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/clim.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[
            BuiltinDocumentationLink { label: "Shared admitted color-limit controls", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/common/color_limits.rs") },
            BuiltinDocumentationLink { label: "Current displayed color-range owner", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/core/state/axes_modes.rs") },
        ],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "clim state and validation tests", location: "builtins::plotting::clim::tests" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Retained integration test source (execution unqualified)", location: "crates/runmat-vm/tests/plotting.rs" }, BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Shared caxis compatibility tests", location: "builtins::plotting::caxis::tests" }],
        notes: &["Legacy behavior, examples, and links are retained. Native and browser graphics execution remains subject to integration gates."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Stable),
};
