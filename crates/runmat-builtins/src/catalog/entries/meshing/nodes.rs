//! `meshing.nodes` catalog identity.

use super::shared::*;
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let output = struct_result(vec![
        ("ids", uint64_matrix(Some(1))),
        ("coordinates", double_matrix(Some(3))),
    ]);
    infer_inspection(request, entry, "NODES", output)
}

const EXAMPLES: &[BuiltinExample] = &[native_example(
    "bounding-box",
    "Read node coordinates and check the part extents",
    concat!(
        box_mesh_setup!(),
        "nodes = meshing.nodes(solverMesh);\n",
        "extent = max(nodes.coordinates) - min(nodes.coordinates)",
    ),
    "assert(isa(nodes.ids, 'uint64'));\nassert(size(nodes.coordinates, 2) == 3);\nassert(numel(nodes.ids) == double(solverMesh.node_count));\nassert(max(abs(extent - [1 2 3])) < 1e-9);",
)];

const DOCUMENTATION: BuiltinDocumentation = documentation(MeshingDocumentation {
    title: "meshing.nodes",
    summary: "Return solver-mesh node identifiers and coordinates in meters.",
    description: "`nodes = meshing.nodes(mesh)` returns a struct with the node identifiers of a `meshing.Mesh` in `nodes.ids` and their coordinates in `nodes.coordinates`.",
    keywords: &["meshing", "nodes", "coordinates", "vertices", "solver mesh"],
    related: &["meshing.elements", "meshing.boundaryFaces", "meshing.generate", "mesh"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "`ids` is an N-by-1 `uint64` column of persistent node identifiers. `coordinates` is an N-by-3 `double` matrix of x, y, and z in meters, row-aligned with `ids`. For Tet10 meshes the list includes midside nodes.",
                "Identifiers are exact `uint64` values and are not guaranteed to be `1:N`; use them to match the node columns of `meshing.elements` and `meshing.boundaryFaces`, for example with `ismember`. The call decodes the mesh in memory and does not touch the filesystem.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Use in a study",
            paragraphs: &[
                "Use node coordinates to check geometry scale and placement before a solve, to compute section planes for `mesh(solverMesh, 'Section', plane)`, or to post-process nodal fields that are reported per node identifier.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Errors",
            paragraphs: &[MESHING_ERRORS_PARAGRAPH, "The input must be a `meshing.Mesh` from `meshing.generate` or `meshing.load`."],
        },
    ],
    examples: EXAMPLES,
    faqs: &[BuiltinDocumentationFaq {
        question: "Are coordinates in the CAD file's units?",
        answer: "No. Solver-mesh coordinates are always meters; the CAD units are applied when the geometry is loaded and meshed.",
    }],
    implementation: &[INSPECTION_SOURCE],
    verification: &[BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Meshing inspection tests",
        location: "crates/runmat-runtime/src/builtins/meshing/mod.rs",
    }],
});

pub const MESHING_NODES_CATALOG_ENTRY: BuiltinCatalogEntry = entry(MeshingEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "meshing.nodes",
    documentation: DOCUMENTATION,
    descriptor: &MESHING_NODES_DESCRIPTOR,
    inference: infer,
    effects: MeshingEffects::Inspection,
    integer_capabilities: &[],
});

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&MESHING_NODES_CATALOG_ENTRY];
