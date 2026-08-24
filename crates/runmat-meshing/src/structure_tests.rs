use std::{fs, path::Path};

fn read_source(path: &Path) -> String {
    fs::read_to_string(path)
        .unwrap_or_else(|error| {
            panic!("source file {} should be readable: {error}", path.display())
        })
        .replace("\r\n", "\n")
}

fn collect_active_source_files(root: &Path, files: &mut Vec<std::path::PathBuf>) {
    for entry in fs::read_dir(root).unwrap_or_else(|error| {
        panic!(
            "source directory {} should be readable: {error}",
            root.display()
        )
    }) {
        let path = entry
            .expect("source directory entry should be readable")
            .path();
        if path.is_dir() {
            if matches!(
                path.file_name().and_then(|name| name.to_str()),
                Some("artifacts" | "target")
            ) {
                continue;
            }
            collect_active_source_files(&path, files);
        } else if matches!(
            path.extension().and_then(|extension| extension.to_str()),
            Some("json" | "md" | "py" | "rs" | "toml" | "ts" | "tsx" | "yaml" | "yml")
        ) {
            files.push(path);
        }
    }
}

#[test]
fn meshing_crate_layout_keeps_stage_implementations_out_of_core() {
    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    for stage_crate in ["size", "curve", "surface", "tetrahedron", "evidence"] {
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
            "removed meshing path still exists: {stale_core_path}"
        );
    }

    let core_manifest = read_source(&crate_root.join("core").join("Cargo.toml"));
    let core_lib = read_source(&crate_root.join("core").join("src").join("lib.rs"));
    for implementation_crate in [
        "runmat-meshing-curve",
        "runmat-meshing-surface",
        "runmat-meshing-tetrahedron",
        "runmat-meshing-evidence",
    ] {
        assert!(
            !core_manifest.contains(implementation_crate),
            "core depends on implementation crate: {implementation_crate}"
        );
    }
    let evidence_manifest = read_source(&crate_root.join("evidence").join("Cargo.toml"));
    let implementation_crate = "runmat-meshing-size";
    assert!(
        !evidence_manifest.contains(implementation_crate),
        "evidence depends on implementation crate instead of core contract exports: {implementation_crate}"
    );
    for facade_export in [
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

    for removed_legacy_path in [
        "cad/Cargo.toml",
        "cad/src/lib.rs",
        "curve/src/contract.rs",
        "curve/src/discretize/mod.rs",
        "curve/src/validate/mod.rs",
        "surface/src/contract.rs",
        "surface/src/param_tri/mod.rs",
        "surface/src/recovery/mod.rs",
        "surface/src/validate/mod.rs",
        "plc/Cargo.toml",
        "plc/src/lib.rs",
        "opt/Cargo.toml",
        "opt/src/lib.rs",
        "src/solid/mod.rs",
        "src/solid/artifact/mod.rs",
        "src/solid/artifact/backend_counts.rs",
        "src/solid/artifact/backend_generation.rs",
        "src/solid/artifact/backend_optimization.rs",
        "src/solid/artifact/backend_quality.rs",
        "src/solid/artifact/backend_recovery.rs",
        "src/solid/artifact/backend_summary.rs",
        "src/solid/tetrahedron_stage.rs",
        "tetrahedron/src/generate/mod.rs",
        "tetrahedron/src/recover/mod.rs",
        "tetrahedron/src/optimize/mod.rs",
    ] {
        assert!(
            !crate_root.join(removed_legacy_path).exists(),
            "retired meshing pipeline still exists: {removed_legacy_path}"
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
fn tessellation_derived_cad_authority_is_absent() {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("meshing crate should be nested beneath the workspace root");
    let mut files = Vec::new();
    collect_active_source_files(&workspace_root.join("crates"), &mut files);

    let retired_fragments = [
        ["extract", "_source_topology"].concat(),
        ["build", "_cad_topology"].concat(),
        ["triangulate", "_face_points"].concat(),
        ["discretize", "_cad_topology_curves"].concat(),
        ["discretize", "_cad_topology_surfaces"].concat(),
        ["runmat", "_meshing_cad"].concat(),
    ];

    let mut violations = Vec::new();
    for path in files {
        let source = read_source(&path);
        for retired in &retired_fragments {
            if source.contains(retired) {
                violations.push(format!("{} contains {retired}", path.display()));
            }
        }
    }
    assert!(
        violations.is_empty(),
        "tessellation-derived CAD authority remains:\n{}",
        violations.join("\n")
    );
}

#[test]
fn retired_standalone_plc_and_optimization_authorities_are_absent() {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("meshing crate should be nested beneath the workspace root");
    let mut files = Vec::new();
    collect_active_source_files(&workspace_root.join("crates"), &mut files);

    let retired_fragments = [
        ["runmat", "_meshing_plc"].concat(),
        ["runmat", "_meshing_opt"].concat(),
    ];
    let mut violations = Vec::new();
    for path in files {
        let source = read_source(&path);
        for retired in &retired_fragments {
            if source.contains(retired) {
                violations.push(format!("{} contains {retired}", path.display()));
            }
        }
    }
    assert!(
        violations.is_empty(),
        "retired standalone meshing authority remains:\n{}",
        violations.join("\n")
    );
}

#[test]
fn meshing_development_observability_stays_feature_gated() {
    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));

    let evidence_manifest = read_source(&crate_root.join("evidence").join("Cargo.toml"));
    assert!(
        evidence_manifest.contains("dev-evidence = []"),
        "mesh debug evidence must remain behind the evidence crate dev-evidence feature"
    );

    let evidence_lib = read_source(&crate_root.join("evidence").join("src").join("lib.rs"));
    assert!(
        evidence_lib.contains("#[cfg(feature = \"dev-evidence\")]\npub mod dev_traces;"),
        "dev_traces module must not be exported without the dev-evidence feature"
    );
    assert!(
        evidence_lib.contains("#[cfg(feature = \"dev-evidence\")]\npub use dev_traces::"),
        "debug evidence API must not be exported without the dev-evidence feature"
    );

    let evidence_artifact =
        read_source(&crate_root.join("evidence").join("src").join("artifact.rs"));
    assert!(
        evidence_artifact
            .contains("#[cfg(feature = \"dev-evidence\")]\nuse crate::MeshDebugEvidence;"),
        "MeshDebugEvidence must not be part of the default evidence artifact type"
    );
    assert!(
        evidence_artifact.contains("#[cfg(feature = \"dev-evidence\")]\n    #[serde(default, skip_serializing_if = \"Option::is_none\")]\n    pub debug: Option<MeshDebugEvidence>,"),
        "debug evidence field must remain feature-gated and omitted from default artifacts"
    );

    let constrained_cavity_mod = read_source(
        &crate_root
            .join("tetrahedron")
            .join("src")
            .join("cavity")
            .join("constrained")
            .join("mod.rs"),
    );
    assert!(
        constrained_cavity_mod.contains("#[cfg(test)]\nmod diagnostic_metrics;"),
        "constrained-cavity diagnostic metrics should stay out of release builds"
    );
    assert!(
        constrained_cavity_mod.contains("#[cfg(test)]\nmod diagnostics;"),
        "constrained-cavity diagnostics should stay out of release builds"
    );

    let constrained_cavity_exact_cover_mod = read_source(
        &crate_root
            .join("tetrahedron")
            .join("src")
            .join("cavity")
            .join("constrained")
            .join("exact_cover")
            .join("mod.rs"),
    );
    assert!(
        constrained_cavity_exact_cover_mod.contains("#[cfg(test)]\nmod diagnostics;"),
        "exact-cover diagnostics should stay out of release builds"
    );
    assert!(
        constrained_cavity_exact_cover_mod.contains("#[cfg(test)]\npub(crate) use diagnostics::*;"),
        "exact-cover diagnostic exports should stay test-only"
    );
}

#[test]
fn retired_synthetic_preparation_contracts_are_absent() {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("meshing crate should be nested beneath the workspace root");
    let mut files = Vec::new();
    for relative_root in [".github", "crates", "docs", "scripts"] {
        collect_active_source_files(&workspace_root.join(relative_root), &mut files);
    }

    let retired_fragments = [
        ["analysis", "_prep"].concat(),
        ["geometry", ".prep_for_analysis"].concat(),
        ["geometry", ".prep_artifact_health"].concat(),
        ["geometry", "_prep_"].concat(),
        ["prep", "_artifact_id"].concat(),
        ["Meshing", "PrepResult"].concat(),
        ["Fea", "PrepContext"].concat(),
    ];

    let mut violations = Vec::new();
    for path in files {
        let source = read_source(&path);
        for retired in &retired_fragments {
            if source.contains(retired) {
                violations.push(format!("{} contains {retired}", path.display()));
            }
        }
    }
    assert!(
        violations.is_empty(),
        "retired synthetic preparation contracts remain:\n{}",
        violations.join("\n")
    );
}

#[test]
fn retired_public_meshing_contracts_are_absent() {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("meshing crate should be nested beneath the workspace root");
    let mut files = Vec::new();
    for relative_root in [".github", "crates", "docs", "scripts"] {
        collect_active_source_files(&workspace_root.join(relative_root), &mut files);
    }

    let retired_fragments = [
        ["StructuredGrid", "Tetrahedron"].concat(),
        ["structured_grid", "_tetrahedron"].concat(),
        ["prepare_geometry", "_for_analysis"].concat(),
        ["MeshingPrep", "Result"].concat(),
        ["geometry.prep", "_for_analysis/v1"].concat(),
        ["geometry-prep", "-for-analysis/v1"].concat(),
        ["deterministic_topology", "_seed/v1"].concat(),
    ];

    let mut violations = Vec::new();
    for path in files {
        let source = read_source(&path);
        for retired in &retired_fragments {
            if source.contains(retired) {
                violations.push(format!("{} contains {retired}", path.display()));
            }
        }
    }
    assert!(
        violations.is_empty(),
        "retired public meshing contracts remain:\n{}",
        violations.join("\n")
    );
}

#[test]
fn meshing_owned_and_bridge_code_has_no_lint_suppressions() {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("meshing crate should be nested beneath the workspace root");
    let mut files = Vec::new();
    for relative_root in [
        "crates/runmat-meshing",
        "crates/runmat-execution-runner-native",
        "crates/runmat-runtime/src/analysis",
        "crates/runmat-runtime/tests/analysis",
    ] {
        collect_active_source_files(&workspace_root.join(relative_root), &mut files);
    }

    let forbidden_fragments = [
        ["#[", "allow("].concat(),
        ["#[", "expect("].concat(),
        ["#![", "allow("].concat(),
        ["#![", "expect("].concat(),
    ];
    let mut violations = Vec::new();
    for path in files {
        let source = read_source(&path);
        for forbidden in &forbidden_fragments {
            if source.contains(forbidden) {
                violations.push(format!("{} contains {forbidden}", path.display()));
            }
        }
    }
    assert!(
        violations.is_empty(),
        "lint suppressions are forbidden in meshing-owned and bridge code:\n{}",
        violations.join("\n")
    );
}
