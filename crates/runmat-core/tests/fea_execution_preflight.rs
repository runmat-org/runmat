#![cfg(not(target_arch = "wasm32"))]
//! Choosing an execution host must not require an exact geometry backend or mesh.
use runmat_core::RunMatSession;

#[test]
fn checked_study_can_be_preflighted_without_materializing_geometry() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("study.fea");
    std::fs::write(
        &path,
        r#"
geometry part {
    construct {
        feature body = box(width = 1 m, depth = 1 m, height = 1 m)
        result = feature.body
    }
}
mesh volume on geometry.part {
    tetrahedra { order = linear; target_size = 0.1 m; maximum_deviation = 0.01 m }
}
model structure {}
study static {
    geometry = geometry.part
    mesh = mesh.volume
    model = model.structure
    analysis solve { structural static {} }
    output displacement = analysis.solve.field.node.displacement
}
"#,
    )
    .unwrap();
    let mut session = RunMatSession::with_options(false, false).unwrap();
    let source =
        futures::executor::block_on(session.load_and_check_fea_source_path(&path)).unwrap();
    assert!(source.check_result().status.is_valid());
    let preflight = source.fea_execution_preflight().unwrap();
    assert!(preflight.browser_supported);
    // No imported-preview measurement exists for this constructed source.
    // Unknown demand must not be replaced by a fabricated precise number.
    assert_eq!(preflight.approximate_memory_bytes, None);
    assert!(preflight.desktop_reason.is_none());
}
