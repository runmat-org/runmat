use crate::*;

const SOURCE: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "DelaunayTri nearest-neighbor execution",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/geometry/triangulation.rs"),
};
const SECTIONS: &[BuiltinDocumentationSection] = &[BuiltinDocumentationSection {
    heading: "Query coordinates and result",
    paragraphs: &[
        "Supply an N-by-2 query matrix or equally sized x and y vectors. The result is an N-by-1 double array of one-based nearest-vertex indices; an empty point set produces NaN indices.",
        "MATLAB-compatible object and query coordinates use double storage. Typed-integer coordinates are accepted in RunMat compatibility mode. The search is performed on the host, without a provider kernel or fusion path.",
    ],
}];
const FUNCTION_EXAMPLE: BuiltinExample = BuiltinExample {
    id: "nearest-triangulation-vertices",
    title: "Find nearest vertices",
    program: "dt = DelaunayTri([0 0; 1 0; 0 1]);\nidx = nearestNeighbor(dt, [0.2 0.1; 0.8 0.1])",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isequal(idx,[1;2]));",
    },
};
const METHOD_EXAMPLE: BuiltinExample = BuiltinExample {
    id: "method-nearest-triangulation-vertices",
    title: "Query through the object method",
    program: "dt = DelaunayTri([0 0; 1 0; 0 1]);\nidx = dt.nearestNeighbor([0.2 0.1; 0.8 0.1])",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isequal(idx,[1;2]));",
    },
};
pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("nearestNeighbor"),
    slug: Some("nearestNeighbor"),
    summary: "Find nearest DelaunayTri vertices for query points.",
    description: "`nearestNeighbor(dt,q)` locates the closest triangulation vertex for each query point; `nearestNeighbor(dt,x,y)` accepts paired coordinate vectors.",
    keywords: &["nearestNeighbor", "DelaunayTri", "triangulation", "geometry"],
    related: &["DelaunayTri", "freeBoundary", "pointLocation"],
    sections: SECTIONS,
    examples: &[FUNCTION_EXAMPLE],
    example_exemption: None,
    faqs: &[],
    links: &[BuiltinDocumentationLink { label: "DelaunayTri", target: BuiltinDocumentationLinkTarget::Builtin("DelaunayTri") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[SOURCE],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Nearest-neighbor runtime tests", location: "builtins::geometry::triangulation::tests" }],
        notes: &[],
    },
    introduced: None,
    status: None,
};
pub(super) const METHOD_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    title: Some("DelaunayTri.nearestNeighbor"),
    slug: Some("DelaunayTri.nearestNeighbor"),
    examples: &[METHOD_EXAMPLE],
    ..DOCUMENTATION
};
