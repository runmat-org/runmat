//! `meshing.regions` catalog identity.

use super::shared::*;
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let region = struct_result(vec![
        ("id", text_scalar()),
        ("kind", text_scalar()),
        ("assembly_path", cell_of(text_scalar(), vec![None, Some(1)])),
        ("element_count", uint64_scalar()),
    ]);
    infer_inspection(
        request,
        entry,
        "REGIONS",
        cell_of(region, vec![Some(1), None]),
    )
}

const EXAMPLES: &[BuiltinExample] = &[native_example(
    "region-inventory",
    "List the physical regions and their element counts",
    concat!(
        box_mesh_setup!(),
        "regions = meshing.regions(solverMesh);\n",
        "body = regions{1};\n",
        "body.element_count",
    ),
    "assert(iscell(regions) && numel(regions) == 1);\nassert(strcmp(body.kind, 'region'));\nassert(body.element_count == solverMesh.element_count);",
)];

const DOCUMENTATION: BuiltinDocumentation = documentation(MeshingDocumentation {
    title: "meshing.regions",
    summary: "List the persistent physical regions of a solver mesh.",
    description: "`regions = meshing.regions(mesh)` returns a 1-by-R cell array with one struct per physical volume region of a `meshing.Mesh`.",
    keywords: &["meshing", "regions", "physical", "volume", "assembly", "solid"],
    related: &["meshing.elements", "meshing.generate", "mesh", "geometry.region"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "Each struct has `id`, the persistent region identifier that also appears in `meshing.elements(mesh).region_ids`; `kind`, the persistent entity kind (normally `\"region\"`); `assembly_path`, a cell column naming the assembly occurrence path in the source CAD; and `element_count`, the exact `uint64` number of elements in the region.",
                "Every retained tetrahedron belongs to exactly one region, so the element counts sum to the mesh's `element_count`. A single solid has one region; an assembly has one per solid occurrence.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Use in a study",
            paragraphs: &[
                "Region identifiers select what `mesh(solverMesh, 'Regions', id)` displays. They are mesh identities, not material names: materials are assigned to geometry regions in the model, and those assignments are projected onto these mesh regions.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Errors",
            paragraphs: &[MESHING_ERRORS_PARAGRAPH, "The input must be a `meshing.Mesh`."],
        },
    ],
    examples: EXAMPLES,
    faqs: &[BuiltinDocumentationFaq {
        question: "Why is the region id a long hash rather than a name?",
        answer: "The identifier is derived from the exact CAD topology so that it stays stable across regeneration. Names come from the geometry and are applied by the model.",
    }],
    implementation: &[INSPECTION_SOURCE],
    verification: &[BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Regions preserve persistent identity components",
        location: "crates/runmat-runtime/src/builtins/meshing/mod.rs::regions_preserve_persistent_identity_components",
    }],
});

pub const MESHING_REGIONS_CATALOG_ENTRY: BuiltinCatalogEntry = entry(MeshingEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "meshing.regions",
    documentation: DOCUMENTATION,
    descriptor: &MESHING_REGIONS_DESCRIPTOR,
    inference: infer,
    effects: MeshingEffects::Inspection,
    integer_capabilities: &[],
});

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&MESHING_REGIONS_CATALOG_ENTRY];
