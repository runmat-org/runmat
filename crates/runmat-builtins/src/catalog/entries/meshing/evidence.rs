//! `meshing.evidence` catalog identity.

use super::shared::*;
use crate::*;
use runmat_types::{CallInference, CallRequest, ValueKindFact};

fn infer(request: &CallRequest, entry: &BuiltinCatalogEntry) -> CallInference {
    let mut output = host_object(MESHING_EVIDENCE_CLASS);
    if let ValueKindFact::Object(object) = &mut output.kind {
        object
            .properties
            .insert("artifact_digest".into(), text_scalar());
        for count in [
            "stage_count",
            "generated_nodes",
            "generated_elements",
            "wall_time_ms",
            "deterministic_seed",
            "geometry_revision",
        ] {
            object.properties.insert(count.into(), uint64_scalar());
        }
    }
    infer_inspection(request, entry, "EVIDENCE", output)
}

const EXAMPLES: &[BuiltinExample] = &[native_example(
    "stage-checks",
    "Confirm what the evidence proves about a mesh",
    concat!(
        box_mesh_setup!(),
        "ev = meshing.evidence(solverMesh);\n",
        "stageNames = cellfun(@(s) s.stage, ev.stages, 'UniformOutput', false)\n",
        "allPassed = all(cellfun(@(s) all(cellfun(@(i) i.passed, s.invariants)), ev.stages))",
    ),
    "assert(isa(ev, 'meshing.Evidence'));\nassert(strcmp(ev.artifact_digest, solverMesh.canonical_digest));\nassert(ev.deterministic_seed == 42);\nassert(ev.generated_elements == solverMesh.element_count);\nassert(numel(ev.stages) == double(ev.stage_count));\nassert(allPassed);",
)];

const DOCUMENTATION: BuiltinDocumentation = documentation(MeshingDocumentation {
    title: "meshing.evidence",
    summary: "Return the validated stage, resource, and identity evidence of a solver mesh.",
    description: "`evidence = meshing.evidence(mesh)` returns a `meshing.Evidence` object describing how a `meshing.Mesh` was generated and what each meshing stage proved.",
    keywords: &["meshing", "evidence", "stages", "invariants", "resources", "provenance", "reproducibility"],
    related: &["meshing.quality", "meshing.save", "meshing.load", "meshing.generate"],
    sections: &[
        BuiltinDocumentationSection {
            heading: "Behavior",
            paragraphs: &[
                "Identity properties: `artifact_digest` is the canonical digest of the mesh the evidence belongs to and equals `mesh.canonical_digest`; `deterministic_seed` is the seed used; `geometry_revision` is the revision of the source geometry. Totals: `stage_count`, `generated_nodes`, `generated_elements`, and `wall_time_ms`. All counters are exact `uint64` values.",
                "`resources` is a struct of aggregate measurements: `generated_nodes`, `generated_elements`, `peak_memory_bytes`, `peak_scratch_bytes`, `wall_time_ms`, `artifact_bytes`, `search_work`, `maximum_recursion_depth`, and `iterations`.",
                "`stages` is a 1-by-S cell of structs in execution order. Each has `stage` (for example `\"curve_mesh\"`), progress counters `completed_work` and `estimated_work`, `peak_memory_bytes`, `peak_scratch_bytes`, `search_work`, `iterations`, `elapsed_time_ms`, `cancellation_checkpoints`, and `invariants`, a cell of structs with `id`, `passed`, and, when the check is quantitative, `measured`, `required`, and `unit`.",
                "The evidence is validated against the mesh whenever a `meshing.Mesh` is created or loaded, so an admitted mesh never carries a failed invariant. Timing and memory values describe the run that produced the mesh and can differ between otherwise identical runs; the mesh itself does not. When the native host reuses accepted stage results from an earlier run, those stages keep the counters and timings measured when they were produced.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Use in a study",
            paragraphs: &[
                "Record the evidence with analysis results to show which request, seed, and geometry revision produced the mesh and which checks it passed. For Tet10 meshes, the evidence is the authority for curved-element Jacobians and achieved boundary accuracy; `meshing.quality` reports straight-sided corner measures only.",
            ],
        },
        BuiltinDocumentationSection {
            heading: "Errors",
            paragraphs: &[MESHING_ERRORS_PARAGRAPH, "The input must be a `meshing.Mesh`."],
        },
    ],
    examples: EXAMPLES,
    faqs: &[BuiltinDocumentationFaq {
        question: "Why does wall_time_ms change between runs with the same seed?",
        answer: "Resource measurements describe the execution, not the result. The mesh and its canonical digest are reproducible; timing and memory are not.",
    }],
    implementation: &[OBJECT_SOURCE],
    verification: &[BuiltinEvidenceReference {
        kind: BuiltinEvidenceKind::UnitTest,
        label: "Meshing builtin tests",
        location: "crates/runmat-runtime/src/builtins/meshing/mod.rs",
    }],
});

pub const MESHING_EVIDENCE_CATALOG_ENTRY: BuiltinCatalogEntry = entry(MeshingEntryDefinition {
    provenance: BuiltinCatalogProvenance::new(file!(), module_path!()),
    name: "meshing.evidence",
    documentation: DOCUMENTATION,
    descriptor: &MESHING_EVIDENCE_DESCRIPTOR,
    inference: infer,
    effects: MeshingEffects::Inspection,
    integer_capabilities: &[],
});

pub(super) const ENTRIES: &[&BuiltinCatalogEntry] = &[&MESHING_EVIDENCE_CATALOG_ENTRY];
