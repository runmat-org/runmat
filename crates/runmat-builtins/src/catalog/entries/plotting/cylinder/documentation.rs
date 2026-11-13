use crate::*;

const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Coordinates and shapes",
        paragraphs: &[
            "`[X,Y,Z] = cylinder` returns three 2-by-21 host Double matrices for a radius-one, unit-height cylinder. The final column closes the circumferential seam.",
            "A vector profile r contributes one row per radius at equally spaced heights; scalar r is repeated at the base and top. With resolution n, the grids are m-by-(n+1), with m=2 for scalar r.",
            "One-, two-, and three-output calls return the leading coordinate grids without plotting. An empty profile retains a 0-by-(n+1) coordinate shape but cannot be plotted.",
        ],
    },
    BuiltinDocumentationSection {
        heading: "Plot forms and placement",
        paragraphs: &[
            "Statement-form `cylinder`, `cylinder(r)`, and `cylinder(r,n)` publish a surface in the current axes. `cylinder(ax,...)` targets an existing axes; supported surface name-value pairs are accepted only on the plotting forms.",
            "The radius is a real host scalar or vector and n is a finite positive whole scalar. Typed integer n is a gated RunMat extension decoded exactly for the structural extent. Coordinate values remain Double.",
            "No cylinder provider kernel or automatic upload is registered. Explicit GPU radius or resolution inputs reject before a readback, while resident arrays in other plot calls retain their existing owners.",
        ],
    },
];

const EXAMPLES: &[BuiltinExample] = &[
    BuiltinExample {
        id: "profile-grids",
        title: "Construct tapered cylinder coordinates",
        program: "[X,Y,Z] = cylinder([1 0.5 1],12);\nsize(X)\nsize(Y)\nsize(Z)",
        display_output: Some("ans =\n     3    13"),
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::Portable,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions {
            source: "assert(isequal(size(X),[3 13])); assert(isequal(size(Y),[3 13])); assert(isequal(size(Z),[3 13])); assert(max(abs(X(:,1)-X(:,end)))<1e-12);",
        },
    },
    BuiltinExample {
        id: "plot-default",
        title: "Plot the default unit cylinder",
        program: "cylinder;\naxis equal;",
        display_output: None,
        compatibility: BuiltinExampleCompatibility::Matlab,
        harness: BuiltinExampleHarness::BrowserGraphics,
        fixture: BuiltinExampleFixture::None,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Figure {
            minimum_figures: 1,
            assertions: "assert(strcmp(get(gca,'Type'),'axes'));",
        },
    },
];

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("cylinder"),
    slug: Some("cylinder"),
    summary: "Create or plot cylinder surface coordinates.",
    description: "`cylinder` constructs X, Y, and Z grids for a unit-height cylinder with a configurable real radius profile and circumferential resolution. A call with no requested outputs plots through the existing surface renderer.",
    keywords: &["cylinder", "surface", "3-D coordinates", "radius profile"],
    related: &["surf", "mesh", "sphere", "axis"],
    sections: SECTIONS,
    examples: EXAMPLES,
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "What is the default output size?", answer: "Each of X, Y, and Z is 2-by-21; a vector of m radii and count n gives m-by-(n+1) grids." },
        BuiltinDocumentationFaq { question: "Does a call without outputs return coordinates?", answer: "No. Statement form plots; assign outputs to obtain coordinate arrays." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "surf", target: BuiltinDocumentationLinkTarget::Builtin("surf") },
        BuiltinDocumentationLink { label: "mesh", target: BuiltinDocumentationLinkTarget::Builtin("mesh") },
        BuiltinDocumentationLink { label: "Implementation", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/cylinder.rs") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Cylinder runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/plotting/ops/cylinder.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Coordinate, plot, and extension tests", location: "builtins::plotting::cylinder::tests" }],
        notes: &["Original sphere-era plotting owner supplies surface publication. Native/browser display and finite-client-renderer scratch require integrated verification; catalog declaration alone is not that evidence."],
    },
    introduced: None,
    status: Some(BuiltinDocumentationStatus::Partial),
};
