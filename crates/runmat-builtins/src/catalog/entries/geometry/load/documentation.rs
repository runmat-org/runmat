use crate::*;

const STL: BuiltinExampleFixture = BuiltinExampleFixture::Filesystem(BuiltinFilesystemFixture {
    id: BuiltinExampleFixtureId { local_name: "triangle-stl" },
    root: BuiltinFilesystemRoot::IsolatedWorkspace,
    entries: &[BuiltinFilesystemEntry::File { relative_path: "part.stl", content: BuiltinFixtureContent::Utf8(
        "solid tri\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid tri\n"
    ) }],
});

pub(super) const DOCUMENTATION: BuiltinDocumentation = BuiltinDocumentation {
    authority: BuiltinDocumentationAuthority::Catalog,
    title: Some("geometry.load"), slug: Some("geometry.load"),
    summary: "Import a geometry file as a typed geometry.Asset.",
    description: "`geometry.load(path)` asynchronously reads a local geometry file, imports it through the configured geometry service, and returns a host `geometry.Asset` object.",
    keywords: &["geometry", "load", "STL", "OBJ", "CAD", "mesh"],
    related: &["geometry.inspect", "geometry.listRegions", "geometry.meshes", "fea.study"],
    sections: &[
        BuiltinDocumentationSection { heading: "Input and result", paragraphs: &[
            "The only argument is a scalar text path. The result is a `geometry.Asset` with source, revision, mesh, region, and mapping metadata; its importer-owned payload is retained for typed method calls. No numeric or integer input is accepted, and the result is host-owned.",
            "STL and other supported mesh formats use the portable importer. Topology-bearing CAD forms require a configured native CAD importer when the selected format needs it; this identity does not promise universal OCCT availability.",
        ] },
        BuiltinDocumentationSection { heading: "Effects and errors", paragraphs: &[
            "Reading the file can suspend and requires filesystem access. File I/O failure, import rejection, and result conversion retain distinct geometry.load errors. The import operation checks cancellation; a failed import publishes no asset.",
            "This is a host materialization and fusion boundary. It does not launch a numeric provider kernel or accept a resident array in place of a path. Browser execution needs an available filesystem bridge and supported importer; native registration alone does not establish browser file access.",
        ] },
    ],
    examples: &[BuiltinExample {
        id: "triangle", title: "Import a triangle STL", program: "asset = geometry.load('part.stl');\nclass(asset)",
        display_output: Some("ans = geometry.Asset"), compatibility: BuiltinExampleCompatibility::RunMat,
        harness: BuiltinExampleHarness::NativeFilesystem, fixture: STL,
        requirements: BuiltinExampleRequirements::NONE,
        verification: BuiltinExampleVerification::Assertions { source: "assert(isa(asset,'geometry.Asset')); assert(~isempty(asset.regions));" },
    }],
    example_exemption: None,
    faqs: &[BuiltinDocumentationFaq { question: "Does load always need native CAD support?", answer: "No. Portable mesh imports such as STL do not require OCCT; supported topology CAD forms can require a configured native importer." }],
    links: &[BuiltinDocumentationLink { label: "Using FEA geometry", target: BuiltinDocumentationLinkTarget::Documentation("/docs/fea/geometry") }],
    media: &[],
    evidence: BuiltinDocumentationEvidence {
        implementation: &[BuiltinDocumentationLink { label: "Geometry import builtin", target: BuiltinDocumentationLinkTarget::Source("https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/geometry/entrypoints.rs") }],
        verification: &[BuiltinEvidenceReference { kind: BuiltinEvidenceKind::UnitTest, label: "Closed STL import and typed asset", location: "crates/runmat-runtime/src/builtins/geometry/tests.rs" }],
        notes: &[],
    },
    introduced: None, status: None,
};
