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
        "runmat-meshing-cad",
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
    let evidence_manifest = fs::read_to_string(crate_root.join("evidence").join("Cargo.toml"))
        .expect("evidence manifest should be readable");
    for implementation_crate in ["runmat-meshing-size"] {
        assert!(
            !evidence_manifest.contains(implementation_crate),
            "evidence depends on implementation crate instead of core contract exports: {implementation_crate}"
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
        "src/solid/artifact/mod.rs",
        "src/solid/artifact/backend_counts.rs",
        "src/solid/artifact/backend_generation.rs",
        "src/solid/artifact/backend_optimization.rs",
        "src/solid/artifact/backend_quality.rs",
        "src/solid/artifact/backend_recovery.rs",
        "src/solid/artifact/backend_summary.rs",
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

#[test]
fn meshing_development_observability_stays_feature_gated() {
    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));

    let evidence_manifest = fs::read_to_string(crate_root.join("evidence").join("Cargo.toml"))
        .expect("evidence manifest should be readable");
    assert!(
        evidence_manifest.contains("dev-evidence = []"),
        "mesh debug evidence must remain behind the evidence crate dev-evidence feature"
    );

    let evidence_lib = fs::read_to_string(crate_root.join("evidence").join("src").join("lib.rs"))
        .expect("evidence lib should be readable");
    assert!(
        evidence_lib.contains("#[cfg(feature = \"dev-evidence\")]\npub mod dev_traces;"),
        "dev_traces module must not be exported without the dev-evidence feature"
    );
    assert!(
        evidence_lib.contains("#[cfg(feature = \"dev-evidence\")]\npub use dev_traces::"),
        "debug evidence API must not be exported without the dev-evidence feature"
    );

    let evidence_artifact =
        fs::read_to_string(crate_root.join("evidence").join("src").join("artifact.rs"))
            .expect("evidence artifact module should be readable");
    assert!(
        evidence_artifact
            .contains("#[cfg(feature = \"dev-evidence\")]\nuse crate::MeshDebugEvidence;"),
        "MeshDebugEvidence must not be part of the default evidence artifact type"
    );
    assert!(
        evidence_artifact.contains("#[cfg(feature = \"dev-evidence\")]\n    #[serde(default, skip_serializing_if = \"Option::is_none\")]\n    pub debug: Option<MeshDebugEvidence>,"),
        "debug evidence field must remain feature-gated and omitted from default artifacts"
    );

    let constrained_cavity_mod = fs::read_to_string(
        crate_root
            .join("tetrahedron")
            .join("src")
            .join("cavity")
            .join("constrained")
            .join("mod.rs"),
    )
    .expect("constrained cavity module should be readable");
    assert!(
        constrained_cavity_mod.contains("#[cfg(test)]\nmod diagnostic_metrics;"),
        "constrained-cavity diagnostic metrics should stay out of release builds"
    );
    assert!(
        constrained_cavity_mod.contains("#[cfg(test)]\nmod diagnostics;"),
        "constrained-cavity diagnostics should stay out of release builds"
    );

    let constrained_cavity_exact_cover_mod = fs::read_to_string(
        crate_root
            .join("tetrahedron")
            .join("src")
            .join("cavity")
            .join("constrained")
            .join("exact_cover")
            .join("mod.rs"),
    )
    .expect("constrained exact-cover module should be readable");
    assert!(
        constrained_cavity_exact_cover_mod.contains("#[cfg(test)]\nmod diagnostics;"),
        "exact-cover diagnostics should stay out of release builds"
    );
    assert!(
        constrained_cavity_exact_cover_mod.contains("#[cfg(test)]\npub(crate) use diagnostics::*;"),
        "exact-cover diagnostic exports should stay test-only"
    );
}
