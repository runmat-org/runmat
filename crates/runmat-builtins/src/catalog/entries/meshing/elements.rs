//! `meshing.elements` catalog identity.

use super::shared::*;
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let output = struct_result(vec![
        ("ids", uint64_matrix(Some(1))),
        ("connectivity", uint64_matrix(None)),
        ("order", text_scalar()),
        ("region_ids", cell_of(text_scalar(), vec![None, Some(1)])),
    ]);
    infer_inspection(request, entry, "ELEMENTS", output)
}

const EXAMPLES: &[BuiltinExample] = &[native_example(
    "connectivity",
    "Read tetrahedron connectivity",
    concat!(
        box_mesh_setup!(),
        "elements = meshing.elements(solverMesh);\n",
        "size(elements.connectivity)",
    ),
    "assert(strcmp(elements.order, 'tet4'));\nassert(size(elements.connectivity, 2) == 4);\nassert(isa(elements.connectivity, 'uint64'));\nassert(numel(elements.ids) == double(solverMesh.element_count));\nassert(iscell(elements.region_ids) && numel(elements.region_ids) == numel(elements.ids));",
)];

const DOCUMENTATION: BuiltinDocumentation = documentation(MeshingDocumentation {
    title: "meshing.elements",
    summary: "Return solver-mesh tetrahedron identifiers, connectivity, order, and regions.",
    description: "`elements = meshing.elements(mesh)` returns a struct describing every volume element of a `meshing.Mesh`: identifiers, node connectivity, element order, and owning region.",
    keywords: &["meshing", "elements", "tetrahedra", "connectivity", "tet4", "tet10"],
    related: &["meshing.nodes", "meshing.regions", "meshing.quality", "meshing.boundaryFaces"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "`ids` is an E-by-1 `uint64` column of element identifiers. `connectivity` is an E-by-4 (`tet4`) or E-by-10 (`tet10`) `uint64` matrix of node identifiers from `meshing.nodes`, one row per element. `order` is `\"tet4\"` or `\"tet10\"`. `region_ids` is an E-by-1 cell of text giving each element's physical region, matching the `id` field of `meshing.regions`.",
                "Rows are aligned across all fields. Node identifiers are exact `uint64` values; map them to rows of `meshing.nodes` rather than assuming they are `1:N`. A mesh that mixes connectivity widths is rejected.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Use in a study",
            paragraphs: &[
                "Use the connectivity to post-process element results, export the mesh, or count elements per region to confirm that every material region was meshed.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Errors",
            paragraphs: &[MESHING_ERRORS_PARAGRAPH, "The input must be a `meshing.Mesh`."],
        },
    ],
    examples: EXAMPLES,
    faqs: &[BuiltinDocumentationFaq {
        question: "In what order are Tet10 nodes listed?",
        answer: "The first four columns are the corner nodes; the remaining six are midside nodes. `meshing.quality` uses the first four columns as the element corners.",
    }],
    implementation: &[INSPECTION_SOURCE],
    verification: &[BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Meshing inspection tests",
        location: "crates/runmat-runtime/src/builtins/meshing/mod.rs",
    }],
});

pub const MESHING_ELEMENTS_CATALOG_ENTRY: BuiltinCatalogEntry = entry(MeshingEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "meshing.elements",
    documentation: DOCUMENTATION,
    descriptor: &MESHING_ELEMENTS_DESCRIPTOR,
    inference: infer,
    effects: MeshingEffects::Inspection,
    integer_capabilities: &[],
});

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&MESHING_ELEMENTS_CATALOG_ENTRY];
