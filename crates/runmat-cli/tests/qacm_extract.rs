//! Local validation extractor (uncommitted): dump a run's fields and mesh nodes as JSON.
use runmat_runtime::analysis::storage::{AnalysisArtifactStore, FilesystemAnalysisArtifactStore};

#[test]
#[ignore]
fn dump() {
    let root = std::env::var("QACM_ARTIFACTS").unwrap();
    let run_id = std::env::var("QACM_RUN").unwrap();
    let mesh_path = std::env::var("QACM_MESH").unwrap();
    let out = std::env::var("QACM_OUT").unwrap();
    let store = FilesystemAnalysisArtifactStore::new(root);
    let run = store.load_run(&run_id).unwrap().expect("run");
    let mesh = runmat_meshing_core::SolverMeshArtifact::canonical_decode(
        &std::fs::read(mesh_path).unwrap(),
    )
    .unwrap();
    let fields: Vec<_> = run
        .run
        .fields
        .iter()
        .map(|f| {
            serde_json::json!({
                "field_id": f.field_id,
                "shape": f.shape,
                "values": f.as_host_f64().map(|v| v.to_vec()),
            })
        })
        .collect();
    let nodes: Vec<_> = mesh
        .topology
        .nodes
        .iter()
        .map(|n| serde_json::json!([n.node_id, n.coordinates_m]))
        .collect();
    std::fs::write(
        out,
        serde_json::to_vec(&serde_json::json!({
            "residual_norms": run.run.residual_norms(),
            "fields": fields,
            "nodes": nodes,
        }))
        .unwrap(),
    )
    .unwrap();
}
