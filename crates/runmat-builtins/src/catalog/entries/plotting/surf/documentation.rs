use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Surface geometry and axes", paragraphs: &[
        "`XData`, `YData`, and `ZData` retain native single, double, or integer class, exact values, and shape. Vertex generation, color preparation, and rendering are explicit floating boundaries and do not replace authoritative scene data.",
        "Surface handles can be inspected and updated with get/set. view, zlabel, colorbar, colormap, and shading use the same subplot-local axes state. surf fills and shades a surface; mesh emphasizes wireframe geometry, and surfc adds contour overlays.",
            "When vector-axis GPU buffers can be exported through the shared plotting device, surface geometry can remain resident through the floating rendering path. Full host coordinate matrices use the CPU parametric-surface path; other unsupported direct-render combinations gather through the existing source-preparation boundary. Color limits and subplot-local metadata follow the same surface pipeline on either path. The shared surface pipeline also serves image, contour-composite, and surface-style plot families.",
    ] },
    BuiltinDocumentationSection {
        heading: "Coordinate and color forms",
        paragraphs: &[
            "`surf(Z)` uses column and row indices as X and Y. `surf(X,Y,Z)` accepts matching coordinate grids or axis vectors. `surf(Z,C)` and `surf(X,Y,Z,C)` retain C independently of Z; C is indexed m-by-n data or truecolor m-by-n-by-3 data where [m,n]=size(Z).",
            "Indexed CData uses the axes colormap and color limits in scaled mode. In direct mode, floating values are one-based colormap indices and integer values are zero-based indices. Floating truecolor components lie in [0,1]; NaN makes an affected vertex transparent. Integer truecolor components are normalized from their native class range only at the renderer boundary.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Graphics properties and residency",
        paragraphs: &[
            "The returned graphics handle exposes CData and CDataMapping through get/set. Color data is retained with its numeric class and shape for scene replay, while rasterization converts it to renderer colors.",
            "Explicit GPU inputs use the existing plotting-source preparation and may require a documented host gather; this plotting sink does not promise a device-resident graphics handle or an automatic elementwise GPU result. Whole-scene host allocation and browser renderer behavior still require integrated qualification.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
BuiltinExample {
    id: "retained-1", title: "Create a shaded surface from a height grid",
    program: "[X, Y] = meshgrid(linspace(-2, 2, 80), linspace(-2, 2, 80));\nZ = sin(X.^2 + Y.^2) ./ (1 + X.^2 + Y.^2);\nh = surf(X, Y, Z);", display_output: None,
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(strcmp(get(h,'Type'),'surface')); assert(isequal(get(h,'ZData'),Z));" },
},
BuiltinExample {
    id: "retained-2", title: "Pair surf with colormap, shading, and colorbar",
    program: "[X, Y] = meshgrid(linspace(-3, 3, 100), linspace(-3, 3, 100));\nZ = cos(X) .* sin(Y);\nh = surf(X, Y, Z);\ncolormap('jet');\nshading interp;\ncolorbar;", display_output: None,
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(strcmp(get(h,'Type'),'surface')); assert(isequal(get(h,'ZData'),Z));" },
},
BuiltinExample {
    id: "retained-3", title: "Adjust camera state and z-axis labeling for a surface",
    program: "[X, Y] = meshgrid(linspace(-2, 2, 60), linspace(-2, 2, 60));\nZ = X .* exp(-X.^2 - Y.^2);\nh = surf(X, Y, Z);\nzlabel('Height');\nview(45, 30);\nget(h, 'Type')", display_output: Some("ans =\n    'surface'"),
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(strcmp(get(h,'Type'),'surface')); assert(isequal(get(h,'ZData'),Z));" },
},
BuiltinExample {
    id: "retained-4", title: "Damped radial ripple with interpolated shading",
    program: "[X, Y] = meshgrid(linspace(-4, 4, 200), linspace(-4, 4, 200));\nR = sqrt(X.^2 + Y.^2) + 0.01;\nZ = sin(3*R) ./ R;\n\nh = surf(X, Y, Z);\nshading interp;\ncolormap('turbo');\ncolorbar;\ntitle('Damped Radial Wavefield');\nxlabel('x');\nylabel('y');\nzlabel('Amplitude');\nview(35, 25);", display_output: None,
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::BrowserGraphics, fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(strcmp(get(h,'Type'),'surface')); assert(isequal(get(h,'ZData'),Z));" },
},
BuiltinExample {
    id: "independent-color-grid",
    title: "Color a surface independently of height",
    program: "Z = [1 2; 3 4]; C = [4 3; 2 1]; h = surf(Z,C); assert(isequal(get(h,'CData'),C));",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::BrowserGraphics,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Figure {
        minimum_figures: 1,
        assertions: "assert(isequal(size(get(h,'CData')),[2 2]));",
    },
}];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("surf"),
    slug: Some("surf"),
    summary: "Plot a surface with optional independent indexed or truecolor CData.",
    description: "`surf` creates a surface graphics object from Z or X,Y,Z, optionally with independent color data C. The renderer preserves source geometry and color storage separately.",
    keywords: &["surf", "surface", "CData", "truecolor", "colormap", "surface plot", "3-D surface", "height map", "matlab surf", "gpu surface rendering"],
    related: &["mesh", "surfc", "patch", "colormap", "caxis"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[
BuiltinDocumentationFaq { question: "How do I change the colormap on a surf plot?", answer: "Call `colormap` after `surf` to swap the palette. The colormap applies to the current axes, so it works per-subplot.\n\n```matlab\nsurf(X, Y, Z);\ncolormap('turbo');\n```\n\nAny named colormap (`'parula'`, `'jet'`, `'hot'`, etc.) or an Nx3 matrix of RGB values works here." },
BuiltinDocumentationFaq { question: "How do I rotate or set the 3-D viewing angle?", answer: "Use `view(az, el)` where `az` is the azimuth in degrees and `el` is the elevation. For example, `view(45, 30)` gives you a 45-degree orbit at 30 degrees above the xy-plane. `view(3)` is a shortcut for the default 3-D perspective, and `view(2)` snaps to a top-down 2-D projection." },
BuiltinDocumentationFaq { question: "What's the difference between surf and mesh?", answer: "`surf` renders a shaded, filled surface where face colors come from the colormap. `mesh` draws only the wireframe grid lines with no face fill. Use `surf` when you want to visualize the shape as a solid surface, and `mesh` when you want to see through the geometry to understand the grid structure underneath." },
BuiltinDocumentationFaq {
        question: "Can colors differ from surface height?",
        answer: "Yes. Pass C as a separate m-by-n indexed grid or m-by-n-by-3 truecolor grid.",
    }],
    links: &[
BuiltinDocumentationLink { label: "mesh", target: BuiltinDocumentationLinkTarget::Builtin("mesh") },
BuiltinDocumentationLink { label: "surfc", target: BuiltinDocumentationLinkTarget::Builtin("surfc") },
BuiltinDocumentationLink { label: "view", target: BuiltinDocumentationLinkTarget::Builtin("view") },
BuiltinDocumentationLink { label: "zlabel", target: BuiltinDocumentationLinkTarget::Builtin("zlabel") },
BuiltinDocumentationLink { label: "colormap", target: BuiltinDocumentationLinkTarget::Builtin("colormap") },
BuiltinDocumentationLink { label: "shading", target: BuiltinDocumentationLinkTarget::Builtin("shading") },
BuiltinDocumentationLink { label: "colorbar", target: BuiltinDocumentationLinkTarget::Builtin("colorbar") },
BuiltinDocumentationLink { label: "Choosing the right plot type", target: BuiltinDocumentationLinkTarget::Documentation("/docs/plotting/choosing-the-right-plot-type") },
BuiltinDocumentationLink { label: "Styling plots and axes", target: BuiltinDocumentationLinkTarget::Documentation("/docs/plotting/styling-plots-and-axes") },
BuiltinDocumentationLink { label: "Plot replay and export", target: BuiltinDocumentationLinkTarget::Documentation("/docs/plotting/plot-replay-and-export") },
BuiltinDocumentationLink { label: "Complete plotting guide", target: BuiltinDocumentationLinkTarget::Documentation("/blog/matlab-plotting-guide") },
BuiltinDocumentationLink {
        label: "Implementation",
        target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/surf.rs"),
    }],
    media: &[BuiltinDocumentationMedia { kind: BuiltinDocumentationMediaKind::Image, url: "https://web.runmatstatic.com/builtin-image/runmat-matlab-plot-surf-damped-radial-ripple.webp", alt: "Damped radial ripple surface with interpolated shading" }],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "Surface renderer",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-plot/src/plots/surface.rs"),
        }],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Surface color and parser tests",
            location: "builtins::plotting::surf::tests",
        }, BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::IntegrationTest,
            label: "Shared renderer tests",
            location: "runmat-plot/tests/renderer_tests.rs",
        }],
        notes: &["Catalog status remains Partial until GPU plotting/resource and browser replay qualification is complete."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
