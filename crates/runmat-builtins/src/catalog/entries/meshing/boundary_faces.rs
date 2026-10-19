//! `meshing.boundaryFaces` catalog identity.

use super::shared::*;
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let output = struct_result(vec![
        ("ids", uint64_matrix(Some(1))),
        ("connectivity", uint64_matrix(None)),
        ("roles", cell_of(text_scalar(), vec![None, Some(1)])),
        (
            "adjacent_element_ids",
            cell_of(uint64_matrix(None), vec![None, Some(1)]),
        ),
        (
            "provenance",
            cell_of(
                cell_of(
                    runmat_types::ValueFact::unknown(runmat_types::DynamicReason::RuntimeValue),
                    vec![Some(1), None],
                ),
                vec![None, Some(1)],
            ),
        ),
    ]);
    infer_inspection(request, entry, "BOUNDARY_FACES", output)
}

const EXAMPLES: &[BuiltinExample] = &[native_example(
    "exterior-skin",
    "Count exterior faces and read their CAD provenance",
    concat!(
        box_mesh_setup!(),
        "faces = meshing.boundaryFaces(solverMesh);\n",
        "exterior = sum(strcmp(faces.roles, 'exterior'))\n",
        "firstOwner = faces.provenance{1}{1}",
    ),
    "assert(size(faces.connectivity, 2) == 3);\nassert(numel(faces.ids) == double(solverMesh.boundary_face_count));\nassert(exterior == numel(faces.ids));\nassert(strcmp(firstOwner.kind, 'face'));",
)];

const DOCUMENTATION: BuiltinDocumentation = documentation(MeshingDocumentation {
    title: "meshing.boundaryFaces",
    summary: "Return solver-mesh boundary triangles with roles, adjacency, and CAD provenance.",
    description: "`faces = meshing.boundaryFaces(mesh)` returns a struct describing every boundary face of a `meshing.Mesh`: identifiers, node connectivity, boundary role, adjacent elements, and the exact CAD entities each face came from.",
    keywords: &["meshing", "boundary", "faces", "triangles", "provenance", "surface"],
    related: &["meshing.nodes", "meshing.elements", "meshing.regions", "geometry.region"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "`ids` is an F-by-1 `uint64` column. `connectivity` is F-by-3 for Tri3 faces of Tet4 meshes and F-by-6 for Tri6 faces of Tet10 meshes, holding node identifiers from `meshing.nodes`. `roles` is an F-by-1 cell with `\"exterior\"`, `\"conformal_interface\"`, `\"contact_primary\"`, or `\"contact_secondary\"`.",
                "`adjacent_element_ids` is an F-by-1 cell whose entries are 1-by-n `uint64` rows of the volume elements that share the face: one for an exterior face, two for an interface between regions. `provenance` is an F-by-1 cell whose entries are 1-by-p cells of structs with `kind` (for example `\"face\"`), `source_topology_id`, and `assembly_path`, naming the persistent CAD entities the face lies on.",
                "Rows are aligned across all fields. The call decodes the mesh in memory and does not touch the filesystem.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Use in a study",
            paragraphs: &[
                "Provenance is what connects solver faces back to CAD faces: boundary conditions applied to a geometry region are projected onto exactly the mesh faces whose provenance names that region's faces. Use it to audit which triangles a load or support will act on.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Errors",
            paragraphs: &[MESHING_ERRORS_PARAGRAPH, "The input must be a `meshing.Mesh`; a mesh that mixes boundary connectivity widths is rejected."],
        },
    ],
    examples: EXAMPLES,
    faqs: &[BuiltinDocumentationFaq {
        question: "Why do some faces have two adjacent elements?",
        answer: "Interior faces between two regions, such as bonded parts of an assembly, are reported as `conformal_interface` faces shared by one element on each side.",
    }],
    implementation: &[INSPECTION_SOURCE],
    verification: &[BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Boundary faces preserve topology roles, adjacency, and full identity",
        location: "crates/runmat-runtime/src/builtins/meshing/mod.rs::boundary_faces_preserve_topology_roles_adjacency_and_full_identity",
    }],
});

pub const MESHING_BOUNDARY_FACES_CATALOG_ENTRY: BuiltinCatalogEntry =
    entry(MeshingEntryDefinition {
        provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
        name: "meshing.boundaryFaces",
        documentation: DOCUMENTATION,
        descriptor: &MESHING_BOUNDARY_FACES_DESCRIPTOR,
        inference: infer,
        effects: MeshingEffects::Inspection,
        integer_capabilities: &[],
    });

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&MESHING_BOUNDARY_FACES_CATALOG_ENTRY];
