//! `meshing.load` catalog identity.

use super::shared::*;
use crate::*;
use runmat_types::{CallInference, CallRequest};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let name = entry.identity.name;
    let mut diagnostics = Vec::new();
    arity(
        &mut diagnostics,
        "LOAD",
        name,
        request.arguments.len(),
        2,
        Some(2),
    );
    require_text(&mut diagnostics, "LOAD", name, request, 0);
    require_text(&mut diagnostics, "LOAD", name, request, 1);
    finish(entry, request, mesh_result(), diagnostics)
}

const EXAMPLES: &[BuiltinExample] = &[native_example(
    "round-trip",
    "Reload a saved mesh and confirm its identity",
    concat!(
        box_mesh_setup!(),
        "meshing.save(solverMesh, 'artifacts/box.solver-mesh.cbor', 'artifacts/box.meshing-evidence.cbor');\n",
        "reloaded = meshing.load('artifacts/box.solver-mesh.cbor', 'artifacts/box.meshing-evidence.cbor');\n",
        "reloaded.canonical_digest",
    ),
    "assert(isa(reloaded, 'meshing.Mesh'));\nassert(strcmp(reloaded.canonical_digest, solverMesh.canonical_digest));\nassert(reloaded.element_count == solverMesh.element_count);",
)];

const DOCUMENTATION: BuiltinDocumentation = documentation(MeshingDocumentation {
    title: "meshing.load",
    summary: "Load a saved solver mesh and its evidence as a jointly validated meshing.Mesh.",
    description: "`mesh = meshing.load(meshPath, evidencePath)` reads a canonical solver-mesh artifact and its matching meshing-evidence artifact and returns them as one `meshing.Mesh`.",
    keywords: &["meshing", "load", "artifact", "evidence", "cbor", "reproducibility"],
    related: &["meshing.save", "meshing.generate", "meshing.evidence", "fea.study"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "Both paths are scalar text. Each file is decoded with its canonical codec, which checks its artifact kind and schema version, and the evidence is then validated against the mesh: the evidence must describe exactly this artifact, its request, and its stage results. Passing the paths in the wrong order, a mismatched pair, or a corrupted file fails without returning a mesh.",
                "The files are the ones `meshing.save` and `runmat mesh` write, so a mesh generated on the command line or on another machine can be inspected and used in scripts. The loaded mesh has the same `canonical_digest` as the mesh that was saved.",
                "If a region-alias sidecar `<meshPath>.regions.json` written by `meshing.save` is present, the importer's `face_`, `edge_`, and `solid_` region aliases are restored with the mesh, so a reloaded mesh resolves model regions exactly like the mesh that was saved. The sidecar must belong to this mesh (it records the mesh's canonical digest); a sidecar from another mesh is rejected. Meshes written by `runmat mesh` have no sidecar and load without aliases.",
                "Reading is synchronous and requires filesystem access.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Use in a study",
            paragraphs: &[
                "Load an archived mesh to rerun a study on exactly the same discretization: pass the result to `fea.study(id, part, model, \"Mesh\", mesh)`. Check `meshing.evidence(mesh)` to confirm the request and geometry revision it was generated from.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Errors",
            paragraphs: &[MESHING_ERRORS_PARAGRAPH, "Typical causes are the wrong number of arguments, an unreadable file, a file of the wrong artifact kind, evidence that does not match the mesh, or a region-alias sidecar that belongs to another mesh."],
        },
    ],
    examples: EXAMPLES,
    faqs: &[BuiltinDocumentationFaq {
        question: "Can I load only the mesh file?",
        answer: "No. A solver mesh is admitted only together with the evidence that proves how it was generated, so both files are required.",
    }],
    implementation: &[BUILTINS_SOURCE, OBJECT_SOURCE],
    verification: &[
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "Save and load retain region aliases through the sidecar",
            location: "crates/runmat-runtime/src/builtins/meshing/mod.rs::save_and_load_retain_region_aliases_through_the_sidecar",
        },
        BuiltinEvidenceReference {
            kind: BuiltinEvidenceKind::UnitTest,
            label: "A sidecar from another mesh is rejected",
            location: "crates/runmat-runtime/src/builtins/meshing/mod.rs::load_rejects_a_sidecar_from_another_mesh_and_save_protects_it",
        },
    ],
});

pub const MESHING_LOAD_CATALOG_ENTRY: BuiltinCatalogEntry = entry(MeshingEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "meshing.load",
    documentation: DOCUMENTATION,
    descriptor: &MESHING_LOAD_DESCRIPTOR,
    inference: infer,
    effects: MeshingEffects::Load,
    integer_capabilities: &[],
});

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&MESHING_LOAD_CATALOG_ENTRY];
