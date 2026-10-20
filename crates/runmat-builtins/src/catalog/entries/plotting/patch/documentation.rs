use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Geometry and properties", paragraphs: &[
        "Vector coordinate inputs create one polygon; matrix coordinate inputs create one polygon per column. Polygon edges close automatically. Faces are MATLAB-style one-based vertex indices; Vertices accepts N-by-2 coordinates for 2-D geometry or N-by-3 for 3-D geometry.",
        "XData, YData, ZData, CData, Faces, and Vertices retain native single, double, or integer class, shape, and exact values for properties and scene persistence. Host triangulation and renderer packing are explicit floating boundaries. The returned patch handle works with get, set, ishandle, and isgraphics.",
        "A typed-integer numeric alias for a leading axes handle is a gated RunMat extension; ordinary axes objects remain the compatibility form.",
        "FaceColor accepts named colors, RGB triples, flat, interp, or none. EdgeColor accepts named colors, RGB triples, or none. FaceAlpha and EdgeAlpha lie in [0,1]. LineWidth, DisplayName, and Visible use the shared plotting property model. Faces, Vertices, XData, YData, and ZData are supported properties; lighting, normals, texture coordinates, and callback properties remain deferred.",
        "Patch is a plotting sink and fusion boundary. Host and gathered-GPU inputs produce the same object and handle behavior. GPU coordinate inputs gather for shape validation, one-based Faces handling, NaN padding, and polygon triangulation before submission to the shared GPU renderer; there is no direct resident triangulation/packing path.",
    ] },
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

const EXAMPLES: &[BuiltinExample] = &[
BuiltinExample {
    id: "retained-1", title: "Create a triangle patch",
    program: "x = [0 1 0];\ny = [0 0 1];\nh = patch(x, y, 'red');", display_output: None,
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(strcmp(get(h,'Type'),'patch')); assert(isequal(get(h,'XData'),x));" },
},
BuiltinExample {
    id: "retained-2", title: "Create two rectangle patches from matrix columns",
    program: "X = [0 2; 1 3; 1 3; 0 2];\nY = [0 0; 0 0; 1 1; 1 1];\nh = patch(X, Y, 'cyan', 'EdgeColor', 'black');", display_output: None,
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(numel(h) == 2);" },
},
BuiltinExample {
    id: "retained-3", title: "Create a patch from Faces and Vertices",
    program: "V = [0 0; 1 0; 1 1; 0 1];\nF = [1 2 3 4];\nh = patch('Faces', F, 'Vertices', V, 'FaceColor', [0.2 0.6 0.9]);\nget(h, 'Type')", display_output: Some("ans =\n    'patch'"),
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(strcmp(get(h,'Type'),'patch')); assert(isequal(get(h,'Faces'),F)); assert(isequal(get(h,'Vertices'),V));" },
},
BuiltinExample {
    id: "retained-4", title: "Create a 3-D patch",
    program: "V = [0 0 0; 1 0 0; 0.5 1 0.5];\nF = [1 2 3];\nh = patch('Faces', F, 'Vertices', V, 'FaceAlpha', 0.6);\nview(3);", display_output: None,
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(isequal(get(h,'Vertices'),V)); assert(abs(get(h,'FaceAlpha') - 0.6) < 1e-6);" },
},
BuiltinExample {
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
    keywords: &["patch", "FaceVertexCData", "CData", "flat", "interp", "polygon", "filled polygon", "faces", "vertices", "matlab patch", "graphics primitive"],
    related: &["fill", "fill3", "surf", "colormap"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[
BuiltinDocumentationFaq { question: "Does patch close polygons automatically?", answer: "Yes. You can provide open polygon coordinate lists; RunMat closes the edge loop when rendering the patch." },
BuiltinDocumentationFaq { question: "Are Faces indices zero-based or one-based?", answer: "They are one-based to match MATLAB. `Faces = [1 2 3]` references the first, second, and third rows of `Vertices`." },
BuiltinDocumentationFaq { question: "Does patch have a direct GPU-resident geometry path?", answer: "Not yet. RunMat gathers GPU-resident patch inputs for validation and triangulation, then submits the resulting geometry to the shared GPU renderer. A direct GPU triangulation/packing path can be added later for very large patch meshes." },
BuiltinDocumentationFaq {
        question: "When can FaceColor be interp?",
        answer: "Provide color data for every vertex, then set FaceColor to interp.",
    }],
    links: &[
BuiltinDocumentationLink { label: "surf", target: BuiltinDocumentationLinkTarget::Builtin("surf") },
BuiltinDocumentationLink { label: "plot3", target: BuiltinDocumentationLinkTarget::Builtin("plot3") },
BuiltinDocumentationLink { label: "fill", target: BuiltinDocumentationLinkTarget::Builtin("fill") },
BuiltinDocumentationLink { label: "get", target: BuiltinDocumentationLinkTarget::Builtin("get") },
BuiltinDocumentationLink { label: "set", target: BuiltinDocumentationLinkTarget::Builtin("set") },
BuiltinDocumentationLink { label: "Graphics Handles", target: BuiltinDocumentationLinkTarget::Documentation("/docs/plotting/graphics-handles") },
BuiltinDocumentationLink { label: "Plot replay and export", target: BuiltinDocumentationLinkTarget::Documentation("/docs/plotting/plot-replay-and-export") },
BuiltinDocumentationLink {
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
        }, BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::IntegrationTest,
            label: "Patch renderer tests",
            location: "runmat-plot/src/plots/patch.rs",
        }],
        notes: &["Catalog status remains Partial until GPU plotting/resource and browser replay qualification is complete."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
