use std::path::Path;

#[test]
fn meshing_crate_layout_keeps_stage_implementations_out_of_core() {
    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    for stage_crate in [
        "cad",
        "size",
        "curve",
        "surface",
        "plc",
        "tetrahedron",
        "opt",
        "evidence",
    ] {
        assert!(
            crate_root.join(stage_crate).join("Cargo.toml").is_file(),
            "missing stage crate: {stage_crate}"
        );
    }

    for stale_core_path in [
        "core/src/plc",
        "core/src/solid",
        "core/src/volume.rs",
        "core/src/solid.rs",
        "core/src/tetrahedron_candidate.rs",
        "core/src/volume_candidate.rs",
    ] {
        assert!(
            !crate_root.join(stale_core_path).exists(),
            "old volume-first meshing path still exists: {stale_core_path}"
        );
    }

    for root_orchestration_path in [
        "src/solid/mod.rs",
        "src/solid/artifact.rs",
        "src/solid/tetrahedron_stage.rs",
    ] {
        assert!(
            crate_root.join(root_orchestration_path).is_file(),
            "missing focused root orchestration module: {root_orchestration_path}"
        );
    }
}
