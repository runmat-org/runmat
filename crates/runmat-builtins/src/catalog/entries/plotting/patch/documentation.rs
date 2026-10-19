use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Polygon colors",
        paragraphs: &[
            "`patch(X,Y,C)` and `patch(X,Y,Z,C)` use a coordinate color grid: indexed C matches the X/Y grid, and RGB C adds a final three-component plane. For uncolored 3-D geometry, specify XData, YData, and ZData properties. `patch('Faces',F,'Vertices',V,'FaceVertexCData',C)` uses indexed scalar/per-face/per-vertex colors or n-by-3 RGB rows. Source colors remain separate from triangulated renderer vertices.",
            "`FaceColor='flat'` uses one color per face (the first vertex for per-vertex data). `FaceColor='interp'` requires per-vertex colors, which are interpolated across each triangle. Floating truecolor components lie in [0,1], with NaN making the affected face/vertex transparent; integer components are normalized from their native class range at rendering.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Properties and replay",
        paragraphs: &[
            "`get` and `set` accept `FaceVertexCData`, `CData`, and `CDataMapping` on patch handles. Color source class and shape survive scene replay; scaled indexed colors use axes color limits or the source extent. In direct mode, floating values are one-based colormap indices and integer values are zero-based indices.",
            "Explicit GPU coordinate/color inputs use the existing plotting gather boundary where supported. Patch output is a host graphics handle, not a resident numeric GPU result. Finite whole-scene allocation and browser renderer execution remain separately qualified.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
    id: "interpolated-vertex-color",
    title: "Interpolate colors across a triangular patch",
    program: "F = [1 2 3]; V = [0 0; 1 0; 0 1]; C = [1 0 0; 0 1 0; 0 0 1]; h = patch('Faces',F,'Vertices',V,'FaceVertexCData',C,'FaceColor','interp'); assert(isequal(get(h,'FaceVertexCData'),C));",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::BrowserGraphics,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Figure {
        minimum_figures: 1,
        assertions: "assert(strcmp(get(h,'FaceColor'),'interp'));",
    },
}];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("patch"),
    slug: Some("patch"),
    summary: "Create a colored polygon patch from coordinates or faces and vertices.",
    description: "`patch` publishes polygon geometry with constant, flat, or interpolated face coloring. Indexed and truecolor source data remain available through graphics properties and scene replay.",
    keywords: &["patch", "FaceVertexCData", "CData", "flat", "interp"],
    related: &["fill", "fill3", "surf", "colormap"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq {
        question: "When can FaceColor be interp?",
        answer: "Provide color data for every vertex, then set FaceColor to interp.",
    }],
    links: &[BuiltinDocumentationLink {
        label: "Implementation",
        target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/patch.rs"),
    }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "Patch renderer",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-plot/src/plots/patch.rs"),
        }],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Patch face-vertex color tests",
            location: "builtins::plotting::patch::tests",
        }],
        notes: &["Catalog status remains Partial until GPU plotting/resource and browser replay qualification is complete."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
