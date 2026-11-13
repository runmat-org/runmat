use crate::*;

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("DelaunayTri"),
    slug: Some("DelaunayTri"),
    summary: "Construct a legacy two-dimensional Delaunay triangulation object.",
    description: "`DelaunayTri` creates a host geometry object with both legacy (`X`, `Triangulation`, `Constraints`) and modern (`Points`, `ConnectivityList`, `Dimension`) properties for the same two-dimensional triangulation. `triplot`, `freeBoundary`, `nearestNeighbor`, and `pointLocation` consume that object.",
    keywords: &["DelaunayTri", "delaunay", "triangulation", "geometry", "mesh"],
    related: &["triplot", "freeBoundary", "nearestNeighbor", "pointLocation"],
    sections: &[
        BuiltinDocumentationSection { heading: "Construction", paragraphs: &[
            "`DelaunayTri()` creates 0-by-2 points and 0-by-3 connectivity. `DelaunayTri(X)` accepts finite N-by-2 points; `DelaunayTri(x,y)` accepts equal-length coordinate vectors. The public point matrix preserves supplied rows even when duplicate coordinates are ignored by triangulation.",
            "`DelaunayTri(X,C)` and `DelaunayTri(x,y,C)` accept empty constraints. Nonempty constraints and three-dimensional point forms fail with `RunMat:DelaunayTri:Unsupported`, not placeholder topology. The returned object has one-based double connectivity.",
        ] },
        BuiltinDocumentationSection { heading: "Classes and placement", paragraphs: &[
            "Double coordinates and topology are the MATLAB-compatible form. Typed-integer coordinates and typed-integer empty constraints are RunMat-only extensions, gated before the host geometry computation. Resident numeric inputs are gathered through their owning provider; the result is a host object, not a resident tensor.",
            "The constructor is an object/topology boundary, not an elementwise or reduction provider kernel. It terminates fusion; no GPU triangulation operation is claimed.",
        ] },
    ],
    examples: &[
        BuiltinExample {
            id: "matrix-points",
            title: "Construct from a point matrix",
            program: "X = [0 0; 1 0; 0 1; 1 1];\ndt = DelaunayTri(X);\ntri = dt.Triangulation",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(dt.X),[4 2])); assert(size(tri,2)==3); assert(isequal(dt.Points,dt.X)); assert(isequal(dt.ConnectivityList,tri));" },
        },
        BuiltinExample {
            id: "coordinate-vectors",
            title: "Construct from coordinate vectors",
            program: "x = [0 1 0 1];\ny = [0 0 1 1];\ndt = DelaunayTri(x,y);\nsize(dt.Triangulation)",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::Portable,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Assertions { source: "assert(isequal(size(dt.X),[4 2])); assert(size(dt.Triangulation,2)==3);" },
        },
        BuiltinExample {
            id: "plot-coordinate-vectors",
            title: "Plot the constructed triangulation",
            program: "x = [0 1 0 1]; y = [0 0 1 1];\ndt = DelaunayTri(x,y);\ntriplot(dt);",
            display_output: None,
            compatibility: BuiltinExampleCompatibility::Matlab,
            harness: BuiltinExampleHarness::BrowserGraphics,
            fixture: BuiltinExampleFixture::None,
            requirements: BuiltinExampleRequirements::NONE,
            verification: BuiltinExampleVerification::Figure { minimum_figures: 1, assertions: "assert(isequal(size(dt.X),[4 2]));" },
        },
    ],
    example_exemption: None,
    faqs: &[
        BuiltinDocumentationFaq { question: "Which properties are available?", answer: "`X`, `Triangulation`, `Constraints`, `Points`, `ConnectivityList`, and `Dimension`." },
        BuiltinDocumentationFaq { question: "Are constrained or 3-D forms supported?", answer: "No. Those forms fail deterministically with `RunMat:DelaunayTri:Unsupported`." },
        BuiltinDocumentationFaq { question: "Does the result stay on the GPU?", answer: "No. Resident coordinates gather for host triangulation and the result is a host object." },
    ],
    links: &[
        BuiltinDocumentationLink { label: "triplot", target: BuiltinDocumentationLinkTarget::Builtin("triplot") },
        BuiltinDocumentationLink { label: "freeBoundary", target: BuiltinDocumentationLinkTarget::Builtin("freeBoundary") },
        BuiltinDocumentationLink { label: "nearestNeighbor", target: BuiltinDocumentationLinkTarget::Builtin("nearestNeighbor") },
        BuiltinDocumentationLink { label: "pointLocation", target: BuiltinDocumentationLinkTarget::Builtin("pointLocation") },
    ],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "DelaunayTri runtime", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/geometry/triangulation.rs") }],
        verification: &[
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Constructor geometry and compatibility tests", location: "builtins::geometry::triangulation::tests" },
            BuiltinEvidenceReference { kind: BuiltinEvidenceKind::IntegrationTest, label: "Triangulation object plotting", location: "builtins::plotting::triplot::tests::triplot_accepts_delaunaytri_object" },
        ],
        notes: &[],
    },
    introduced: None,
    status: None,
};
