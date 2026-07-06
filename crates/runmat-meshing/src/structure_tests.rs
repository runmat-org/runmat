use std::{fs, path::Path};

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

    let core_manifest = fs::read_to_string(crate_root.join("core").join("Cargo.toml"))
        .expect("core manifest should be readable");
    let core_lib = fs::read_to_string(crate_root.join("core").join("src").join("lib.rs"))
        .expect("core lib should be readable");
    for implementation_crate in [
        "runmat-meshing-curve",
        "runmat-meshing-surface",
        "runmat-meshing-plc",
        "runmat-meshing-tetrahedron",
        "runmat-meshing-opt",
        "runmat-meshing-evidence",
    ] {
        assert!(
            !core_manifest.contains(implementation_crate),
            "core depends on implementation crate: {implementation_crate}"
        );
    }
    for facade_export in [
        "pub use runmat_meshing_cad as cad",
        "pub use runmat_meshing_size as size",
        "pub mod source_topology",
        "pub use cad::",
        "pub use size::",
        "pub use source_topology::",
    ] {
        assert!(
            !core_lib.contains(facade_export),
            "core exposes CAD implementation facade: {facade_export}"
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

    for stale_root_adapter in ["src/curve_contract.rs", "src/surface_contract.rs"] {
        assert!(
            !crate_root.join(stale_root_adapter).exists(),
            "stage contract adapter still lives in root orchestration: {stale_root_adapter}"
        );
    }
}
