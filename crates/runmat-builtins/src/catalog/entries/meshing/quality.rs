//! `meshing.quality` catalog identity.

use super::shared::*;
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let output = struct_result(vec![
        ("element_order", text_scalar()),
        ("element_ids", uint64_matrix(Some(1))),
        ("corner_scaled_jacobian", double_matrix(Some(1))),
        ("corner_aspect_ratio", double_matrix(Some(1))),
        ("corner_volume_m3", double_matrix(Some(1))),
        ("minimum_corner_scaled_jacobian", double_scalar()),
        ("mean_corner_scaled_jacobian", double_scalar()),
        ("p05_corner_scaled_jacobian", double_scalar()),
        ("maximum_corner_aspect_ratio", double_scalar()),
        ("mean_corner_aspect_ratio", double_scalar()),
        ("p95_corner_aspect_ratio", double_scalar()),
        ("minimum_corner_volume_m3", double_scalar()),
        ("total_corner_volume_m3", double_scalar()),
    ]);
    infer_inspection(request, entry, "QUALITY", output)
}

const EXAMPLES: &[BuiltinExample] = &[native_example(
    "quality-check",
    "Check element shape and recover the part volume",
    concat!(
        box_mesh_setup!(),
        "q = meshing.quality(solverMesh);\n",
        "fprintf('min scaled Jacobian %.3f, p95 aspect ratio %.2f, volume %.6f m^3\\n', ...\n",
        "    q.minimum_corner_scaled_jacobian, q.p95_corner_aspect_ratio, q.total_corner_volume_m3);",
    ),
    "assert(q.minimum_corner_scaled_jacobian > 0);\nassert(abs(q.total_corner_volume_m3 - 6) < 1e-9);\nassert(numel(q.corner_scaled_jacobian) == numel(q.element_ids));\nassert(q.p05_corner_scaled_jacobian >= q.minimum_corner_scaled_jacobian);",
)];

const DOCUMENTATION: BuiltinDocumentation = documentation(MeshingDocumentation {
    title: "meshing.quality",
    summary: "Compute per-element corner quality and summary statistics for a solver mesh.",
    description: "`quality = meshing.quality(mesh)` returns a struct of per-tetrahedron scaled Jacobian, edge aspect ratio, and volume, with summary statistics, for a `meshing.Mesh`.",
    keywords: &["meshing", "quality", "jacobian", "aspect ratio", "volume", "element shape"],
    related: &["meshing.evidence", "meshing.elements", "meshing.generate"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "`element_ids` is an E-by-1 `uint64` column, and `corner_scaled_jacobian`, `corner_aspect_ratio`, and `corner_volume_m3` are E-by-1 `double` columns aligned with it. Each value is computed from the element's four corner nodes. The scaled Jacobian lies in `[-1, 1]`; 1 is a regular tetrahedron and values near 0 are nearly flat. The edge aspect ratio is 1 for a regular tetrahedron and grows as elements degrade. Volume is in cubic meters.",
                "Summary fields are `minimum_corner_scaled_jacobian`, `mean_corner_scaled_jacobian`, `p05_corner_scaled_jacobian`, `maximum_corner_aspect_ratio`, `mean_corner_aspect_ratio`, `p95_corner_aspect_ratio`, `minimum_corner_volume_m3`, and `total_corner_volume_m3`, plus `element_order`. Percentiles use the nearest rank `round((E - 1) * q)` of the sorted values.",
                "For Tet4 the corner measures describe the element itself. For Tet10 they describe the straight-sided corner tetrahedron; certified curved Jacobians and achieved high-order geometry checks are in `meshing.evidence`. A mesh with no volume elements is rejected.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Use in a study",
            paragraphs: &[
                "Check quality before a solve: a low minimum or 5th-percentile scaled Jacobian or a high 95th-percentile aspect ratio points to elements that can degrade stress accuracy. Compare `total_corner_volume_m3` with the CAD volume to confirm that the mesh fills the part. Compare quality together with evidence and solver convergence, not element counts alone, when assessing two meshes.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Errors",
            paragraphs: &[MESHING_ERRORS_PARAGRAPH, "The input must be a `meshing.Mesh` with at least one volume element."],
        },
    ],
    examples: EXAMPLES,
    faqs: &[BuiltinDocumentationFaq {
        question: "Is a positive minimum Jacobian guaranteed?",
        answer: "Yes for an admitted mesh: generation fails rather than returning inverted or degenerate elements. The statistics let you judge how far above that floor the mesh is.",
    }],
    implementation: &[INSPECTION_SOURCE],
    verification: &[BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Quality returns aligned exact ids and summary statistics",
        location: "crates/runmat-runtime/src/builtins/meshing/mod.rs::quality_returns_aligned_exact_ids_and_summary_statistics",
    }],
});

pub const MESHING_QUALITY_CATALOG_ENTRY: BuiltinCatalogEntry = entry(MeshingEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "meshing.quality",
    documentation: DOCUMENTATION,
    descriptor: &MESHING_QUALITY_DESCRIPTOR,
    inference: infer,
    effects: MeshingEffects::Inspection,
    integer_capabilities: &[],
});

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&MESHING_QUALITY_CATALOG_ENTRY];
