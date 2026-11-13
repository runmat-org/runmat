#![cfg(feature = "occt-native")]

use std::fs;
use std::process::Command;

use runmat_meshing_core::CanonicalMeshingContract;
use runmat_runtime::analysis::storage::{AnalysisArtifactStore, FilesystemAnalysisArtifactStore};
use serde_json::Value;
use tempfile::TempDir;

#[path = "support/local_offline.rs"]
mod local_offline;
#[path = "support/fea_admission.rs"]
mod fea_admission;

#[path = "typed_fea_geometry_cli/acoustic.rs"]
mod acoustic;

#[path = "typed_fea_geometry_cli/closure.rs"]
mod closure;

#[path = "typed_fea_geometry_cli/assessments.rs"]
mod assessments;

#[path = "typed_fea_geometry_cli/electrical.rs"]
mod electrical;
#[path = "typed_fea_geometry_cli/harmonic.rs"]
mod harmonic;

fn runmat(project: &TempDir, args: &[&str]) -> std::process::Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_runmat"));
    local_offline::configure(&mut command, project.path());
    command
        .current_dir(project.path())
        .args(args)
        .env_remove("RUNMAT_CONFIG")
        .env(
            "RUNMAT_PACKAGE_CACHE_DIR",
            project.path().join("package-cache"),
        )
        .env(
            "RUNMAT_EXECUTION_STATE_DIR",
            project.path().join("execution-state"),
        )
        .env("RUNMAT_FEA_ARTIFACT_ROOT", project.path().join("artifacts"))
        .env(
            "RUNMAT_FEA_STUDY_ARTIFACT_ROOT",
            project.path().join("study-artifacts"),
        );
    let _admission = fea_admission::AdmissionServer::for_command(&mut command, project.path());
    command.output().expect("run typed FEA geometry command")
}

#[test]
fn generated_and_derived_geometry_complete_the_typed_study_pipeline() {
    let project = TempDir::new().unwrap();
    let source_root = project.path().join("src");
    fs::create_dir(&source_root).unwrap();
    fs::write(
        project.path().join("runmat.toml"),
        "[package]\nname = \"typed-fea-geometry\"\n\n[sources]\nroots = [\"src\"]\n",
    )
    .unwrap();
    fs::write(
        source_root.join("makePart.m"),
        r#"function part = makePart(width, depth, height)
part = geometry.finish(geometry.box(width, depth, height));
end
"#,
    )
    .unwrap();
    fs::write(source_root.join("main.fea"), STUDY).unwrap();

    let checked = runmat(
        &project,
        &["check", "src/main.fea", "--study", "response", "--json"],
    );
    assert!(
        checked.status.success(),
        "generated/derived study failed check:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&checked.stdout),
        String::from_utf8_lossy(&checked.stderr)
    );
    let checked: Value = serde_json::from_slice(&checked.stdout).unwrap();
    assert_eq!(checked["status"], "valid");
    assert_eq!(checked["rootModule"], "src/main.fea");
    assert_eq!(checked["program"]["studies"].as_array().unwrap().len(), 1);
    assert_eq!(checked["program"]["studies"][0]["study_id"], "response");

    let first = runmat(
        &project,
        &["run", "src/main.fea", "--study", "response", "--json"],
    );
    assert!(
        first.status.success(),
        "generated/derived study failed run:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&first.stdout),
        String::from_utf8_lossy(&first.stderr)
    );
    let first: Value = serde_json::from_slice(&first.stdout).unwrap();
    assert_eq!(first["root_id"], "response");
    assert_eq!(first["procedure"], "structural_linear_sequence");
    assert!(first["study_definition_digest"]
        .as_str()
        .is_some_and(|digest| digest.starts_with("sha256:")));
    assert!(first["nodes"].as_u64().unwrap() > 0);
    assert!(first["elements"].as_u64().unwrap() > 0);
    let step_run_ids = first["step_run_ids"]
        .as_array()
        .expect("typed publication must expose its persisted step runs");
    assert_eq!(step_run_ids.len(), 1);
    let result_manifest = first["result_manifest_artifact_path"]
        .as_str()
        .expect("typed publication must expose its result manifest");
    assert!(
        project.path().join(result_manifest).is_file(),
        "typed result manifest was not persisted at {result_manifest}"
    );
    let first_manifest_bytes = fs::read(project.path().join(result_manifest)).unwrap();

    let repeated = runmat(
        &project,
        &["run", "src/main.fea", "--study", "response", "--json"],
    );
    assert!(
        repeated.status.success(),
        "repeated generated/derived study failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&repeated.stdout),
        String::from_utf8_lossy(&repeated.stderr)
    );
    let repeated: Value = serde_json::from_slice(&repeated.stdout).unwrap();
    assert_eq!(repeated["mesh_digest"], first["mesh_digest"]);
    assert_eq!(
        repeated["study_definition_digest"],
        first["study_definition_digest"]
    );
    assert_eq!(repeated["nodes"], first["nodes"]);
    assert_eq!(repeated["elements"], first["elements"]);
    let repeated_ids = repeated["step_run_ids"].as_array().unwrap();
    assert_eq!(repeated_ids.len(), step_run_ids.len());
    for (original, repeated) in step_run_ids.iter().zip(repeated_ids) {
        let (original_definition, original_attempt) =
            runmat_runtime::analysis::split_attempt_run_id(original.as_str().unwrap())
                .expect("original execution has an attempt-owned run ID");
        let (repeated_definition, repeated_attempt) =
            runmat_runtime::analysis::split_attempt_run_id(repeated.as_str().unwrap())
                .expect("repeated execution has an attempt-owned run ID");
        assert_eq!(original_definition, repeated_definition);
        assert_ne!(original_attempt, repeated_attempt);
    }
    let repeated_manifest = repeated["result_manifest_artifact_path"]
        .as_str()
        .expect("repeat publication must expose its result manifest");
    assert!(project.path().join(repeated_manifest).is_file());
    assert_ne!(result_manifest, repeated_manifest);
    assert_eq!(
        fs::read(project.path().join(result_manifest)).unwrap(),
        first_manifest_bytes,
        "a fresh execution must not overwrite its predecessor's publication"
    );
}

#[test]
fn authored_thermal_orders_and_procedures_publish_cold_uniform_equilibrium() {
    for quadratic in [false, true] {
        for transient in [false, true] {
            let project = TempDir::new().unwrap();
            let source_root = project.path().join("src");
            fs::create_dir(&source_root).unwrap();
            fs::write(
                project.path().join("runmat.toml"),
                "[package]\nname = \"typed-thermal\"\n\n[sources]\nroots = [\"src\"]\n\n[runtime.fea]\nartifact_root = \"artifacts\"\nstudy_artifact_root = \"study-artifacts\"\n",
            )
            .unwrap();
            fs::write(
                source_root.join("makePart.m"),
                "function part = makePart()\npart = geometry.finish(geometry.box(1, 1, 1));\nend\n",
            )
            .unwrap();
            let mut source = include_str!("fixtures/typed_box_thermal.fea").to_owned();
            if quadratic {
                source = source.replace("order = linear", "order = quadratic");
            }
            if transient {
                source = source.replace("thermal steady {}", "thermal transient {\nstart_time = 0 s\nend_time = 1 s\ntime_step = 0.25 s\ninitial_conditions = [model.initial.uniform_start]\n}");
            }
            fs::write(source_root.join("main.fea"), source).unwrap();
            let output = runmat(
                &project,
                &["run", "src/main.fea", "--study", "response", "--json"],
            );
            assert!(
                output.status.success(),
                "thermal quadratic={quadratic} transient={transient}: stdout={} stderr={}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            let summary: Value = serde_json::from_slice(&output.stdout).unwrap();
            assert_eq!(summary["root_id"], "response");
            assert_eq!(
                summary["procedure"],
                if transient {
                    "thermal_transient"
                } else {
                    "thermal_steady"
                }
            );
            let run_ids = summary["step_run_ids"].as_array().unwrap();
            assert_eq!(run_ids.len(), 1);
            assert!(
                summary["nodes"].as_u64().unwrap() > 0 && summary["elements"].as_u64().unwrap() > 0
            );
            let manifest_path = summary["result_manifest_artifact_path"].as_str().unwrap();
            let manifest: Value =
                serde_json::from_slice(&fs::read(project.path().join(manifest_path)).unwrap())
                    .unwrap();
            assert!(
                manifest.is_object(),
                "persisted thermal manifest must reopen as typed JSON"
            );
            let meshes = fs::read_dir(project.path().join("artifacts/meshes"))
                .unwrap()
                .map(|entry| entry.unwrap().path())
                .filter(|path| path.to_string_lossy().ends_with(".solver-mesh.cbor"))
                .collect::<Vec<_>>();
            assert_eq!(meshes.len(), 1);
            let mesh = runmat_meshing_core::SolverMeshArtifact::canonical_decode(
                &fs::read(&meshes[0]).unwrap(),
            )
            .unwrap();
            mesh.validate().unwrap();
            let mesh_digest = mesh
                .canonical_digest
                .bytes()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>();
            assert_eq!(summary["mesh_digest"], mesh_digest);
            assert!(mesh
                .topology
                .volume_elements
                .iter()
                .all(|element| element.order
                    == if quadratic {
                        runmat_meshing_core::ElementOrder::Tet10
                    } else {
                        runmat_meshing_core::ElementOrder::Tet4
                    }));
            assert_eq!(
                summary["nodes"].as_u64().unwrap() as usize,
                mesh.topology.nodes.len()
            );
            assert_eq!(
                summary["elements"].as_u64().unwrap() as usize,
                mesh.topology.volume_elements.len()
            );
            // Fresh store instance after child-process exit: no in-memory producer fields survive.
            let store = FilesystemAnalysisArtifactStore::new(project.path().join("artifacts"));
            let retained = store
                .load_run(run_ids[0].as_str().unwrap())
                .unwrap()
                .unwrap();
            let render = retained.render_topology.as_ref().unwrap();
            assert_eq!(
                render.volume_element_count,
                mesh.topology.volume_elements.len()
            );
            for render_mesh in &render.meshes {
                for (vertex, index) in render_mesh
                    .vertices
                    .iter()
                    .zip(&render_mesh.vertex_volume_node_indices)
                {
                    if let Some(index) = index {
                        assert_eq!(
                            *vertex,
                            mesh.topology.nodes[*index].coordinates_m.canonical_value()
                        );
                    }
                }
            }
            assert!(retained.modal_results.is_none() && retained.nonlinear_results.is_none());
            let thermal = retained.thermal_results.unwrap();
            assert_eq!(
                thermal.temperature_snapshots.len(),
                if transient { 5 } else { 1 }
            );
            if transient {
                assert_eq!(thermal.time_points_s, vec![0.0, 0.25, 0.5, 0.75, 1.0]);
            }
            for field in &thermal.temperature_snapshots {
                assert_eq!(
                    field.shape.iter().product::<usize>(),
                    mesh.topology.nodes.len()
                );
                assert!(field
                    .as_host_f64()
                    .unwrap()
                    .iter()
                    .all(|t| t.is_finite() && (t - 300.0).abs() < 1e-8));
            }
            for field in thermal
                .temperature_gradient_snapshots
                .iter()
                .chain(&thermal.heat_flux_snapshots)
            {
                assert!(field
                    .as_host_f64()
                    .unwrap()
                    .iter()
                    .all(|value| value.is_finite() && value.abs() < 1e-8));
            }
            let repeated = runmat(
                &project,
                &["run", "src/main.fea", "--study", "response", "--json"],
            );
            assert!(
                repeated.status.success(),
                "repeat thermal failed: {}",
                String::from_utf8_lossy(&repeated.stderr)
            );
            let repeated: Value = serde_json::from_slice(&repeated.stdout).unwrap();
            for key in [
                "mesh_digest",
                "study_definition_digest",
                "nodes",
                "elements",
            ] {
                assert_eq!(repeated[key], summary[key], "thermal repeat changed {key}");
            }
            let repeated_manifest: Value = serde_json::from_slice(
                &fs::read(
                    project
                        .path()
                        .join(repeated["result_manifest_artifact_path"].as_str().unwrap()),
                )
                .unwrap(),
            )
            .unwrap();
            assert_eq!(
                repeated_manifest["workload_digest"],
                manifest["workload_digest"]
            );
            // Distinct execution evidence must not collide with the immutable first run.
            if repeated_manifest["numerical_evidence"]["root"]
                != manifest["numerical_evidence"]["root"]
            {
                assert_ne!(repeated["step_run_ids"], summary["step_run_ids"]);
            } else {
                assert_eq!(repeated["step_run_ids"], summary["step_run_ids"]);
            }
            assert!(store
                .load_run(run_ids[0].as_str().unwrap())
                .unwrap()
                .is_some());
        }
    }
}

const STUDY: &str = r#"
import function makePart from "./makePart.m"

geometry generated_part {
    input width : length = 1 m
    input depth : length = 1 m
    input height : length = 1 m
    generated by function.makePart {
        arguments {
            width = input.width
            depth = input.depth
            height = input.height
        }
    }
}

geometry relieved from geometry.generated_part {
    derive {
        feature hole = cylinder(
            radius = 0.1 m,
            height = 1 m,
            axis = global([0, 0, 1]),
            origin = [0.5, 0.5, 0] m,
        )
        result = subtract(source.result, feature.hole)
    }
    region body : solid {
        select result
        expect exactly_one
    }
    region support : face {
        select result
        expect one_or_more
    }
}

mesh volume on geometry.relieved {
    tetrahedra {
        order = linear
        target_size = 0.4 m
        maximum_deviation = 0.02 m
    }
}

material steel {
    mass_density = 7850 kg/m^3
    elastic {
        youngs_modulus = 200 GPa
        poisson_ratio = 0.3
    }
}

model structure {
    assign material.steel to geometry.relieved.region.body
    constraint base {
        structural.fixed { target = geometry.relieved.region.support }
    }
}

study response {
    geometry = geometry.relieved
    mesh = mesh.volume
    model = model.structure
    analysis solve {
        structural static {}
    }
    output displacement = analysis.solve.field.node.displacement
}
"#;
