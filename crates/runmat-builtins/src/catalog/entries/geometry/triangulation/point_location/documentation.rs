use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "DelaunayTri point-location execution",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/geometry/triangulation.rs"),
};
const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection { heading: "Query geometry and outputs", paragraphs: &[
        "Supply an N-by-2 query matrix or paired x and y vectors. The first result is an N-by-1 double array of one-based containing-triangle indices; outside points produce NaN.",
        "Requesting two outputs also returns N-by-3 double barycentric coordinates. The query is evaluated against host triangulation topology; no provider kernel or fusion path is claimed.",
    ] },
    BuiltinDocumentationSection { heading: "Compatible classes", paragraphs: &[
        "MATLAB-compatible object points, connectivity, and query coordinates use double storage. Typed-integer coordinates and connectivity are RunMat-only extensions with explicit compatibility gates.",
    ] },
];
const FUNCTION_EXAMPLE: BuiltinExample = BuiltinExample {
    id: "locate-in-triangle",
    title: "Locate a point",
    program: "dt = DelaunayTri([0 0; 1 0; 0 1]);\nti = pointLocation(dt, [0.25 0.25])",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isequal(ti,1));",
    },
};
const METHOD_EXAMPLE: BuiltinExample = BuiltinExample {
    id: "method-locate-in-triangle",
    title: "Request barycentric coordinates",
    program: "dt = DelaunayTri([0 0; 1 0; 0 1]);\n[ti,bc] = dt.pointLocation([0.25 0.25])",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source:
            "assert(isequal(ti,1)); assert(isequal(size(bc),[1 3])); assert(abs(sum(bc)-1)<1e-12);",
    },
};
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("pointLocation"), slug: Some("pointLocation"),
    summary: "Locate query points in a DelaunayTri triangulation.",
    description: "`pointLocation(dt,q)` returns containing-triangle indices in a 2-D `DelaunayTri` object; with two outputs, it also returns barycentric coordinates.",
    keywords: &["pointLocation", "DelaunayTri", "triangulation", "barycentric", "geometry"],
    related: &["DelaunayTri", "freeBoundary", "nearestNeighbor"],
    sections: SECTIONS, examples: &[FUNCTION_EXAMPLE], example_exemption: None, faqs: &[],
    links: &[BuiltinDocumentationLink { label: "DelaunayTri", target: BuiltinDocumentationLinkTarget::Builtin("DelaunayTri") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[SOURCE],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Point-location runtime tests", location: "builtins::geometry::triangulation::tests" }],
        notes: &[],
    },
    introduced: None, status: None,
};
pub(super) const METHOD_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    title: Some("DelaunayTri.pointLocation"),
    slug: Some("DelaunayTri.pointLocation"),
    examples: &[METHOD_EXAMPLE],
    ..DOCUMENTATION
};
