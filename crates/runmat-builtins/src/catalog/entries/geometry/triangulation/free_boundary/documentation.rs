use crate::*;

const IMPLEMENTATION: BuiltinDocumentationLink = BuiltinDocumentationLink {
    label: "DelaunayTri boundary execution",
    target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/geometry/triangulation.rs"),
};
const SECTIONS: &[BuiltinDocumentationSection] = &[
    BuiltinDocumentationSection {
        heading: "Boundary topology",
        paragraphs: &[
            "`freeBoundary(dt)` returns one-based vertex pairs for edges used by exactly one triangle. Edges are sorted deterministically. Empty connectivity returns a 0-by-2 double array.",
            "The receiver is a `DelaunayTri` object. Its double connectivity is the MATLAB-compatible form; manually supplied typed-integer connectivity requires RunMat compatibility mode. Geometry is evaluated on the host, with no provider kernel or fusion path.",
        ],
    },
];
const EXAMPLE: BuiltinExample = BuiltinExample {
    id: "boundary-of-triangle",
    title: "Find outer edges",
    program: "dt = DelaunayTri([0 0; 1 0; 0 1]);\ne = freeBoundary(dt)",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isequal(e,[1 2;1 3;2 3]));",
    },
};
const METHOD_EXAMPLE: BuiltinExample = BuiltinExample {
    id: "method-boundary-of-triangle",
    title: "Call the object method",
    program: "dt = DelaunayTri([0 0; 1 0; 0 1]);\ne = dt.freeBoundary()",
    display_output: None,
    compatibility: BuiltinExampleCompatibility::Matlab,
    harness: BuiltinExampleHarness::Portable,
    fixture: BuiltinExampleFixture::None,
    requirements: BuiltinExampleRequirements::NONE,
    verification: BuiltinExampleVerification::Assertions {
        source: "assert(isequal(size(e),[3 2]));",
    },
};

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("freeBoundary"),
    slug: Some("freeBoundary"),
    summary: "Return boundary edges of a DelaunayTri object.",
    description:
        "`freeBoundary` extracts the boundary edge list from host triangulation connectivity.",
    keywords: &[
        "freeBoundary",
        "DelaunayTri",
        "triangulation",
        "boundary",
        "mesh",
    ],
    related: &["DelaunayTri", "nearestNeighbor", "pointLocation"],
    sections: SECTIONS,
    examples: &[EXAMPLE],
    example_exemption: None,
    faqs: &[],
    links: &[BuiltinDocumentationLink {
        label: "DelaunayTri",
        target: BuiltinDocumentationLinkTarget::Builtin("DelaunayTri"),
    }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[IMPLEMENTATION],
        verification: &[BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Boundary and empty topology tests",
            location: "builtins::geometry::triangulation::tests",
        }],
        notes: &[],
    },
    introduced: None,
    status: None,
};
pub(super) const METHOD_DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    title: Some("DelaunayTri.freeBoundary"),
    slug: Some("DelaunayTri.freeBoundary"),
    summary: "Return boundary edges through the DelaunayTri object method.",
    examples: &[METHOD_EXAMPLE],
    ..DOCUMENTATION
};
