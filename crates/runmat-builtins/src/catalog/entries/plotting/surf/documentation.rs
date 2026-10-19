use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
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

const EXAMPLES: &[BuiltinExample] = &[BuiltinExample {
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
    keywords: &["surf", "surface", "CData", "truecolor", "colormap"],
    related: &["mesh", "surfc", "patch", "colormap", "caxis"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq {
        question: "Can colors differ from surface height?",
        answer: "Yes. Pass C as a separate m-by-n indexed grid or m-by-n-by-3 truecolor grid.",
    }],
    links: &[BuiltinDocumentationLink {
        label: "Implementation",
        target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/surf.rs"),
    }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink {
            label: "Surface renderer",
            target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-plot/src/plots/surface.rs"),
        }],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Surface color and parser tests",
            location: "builtins::plotting::surf::tests",
        }],
        notes: &["Catalog status remains Partial until GPU plotting/resource and browser replay qualification is complete."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
