//! SIM-24: the same study gives the same result on every run and for every local worker and
//! thread count. Execution placement and timing are evidence, never result identity.
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;
use std::process::Command;
use tempfile::TempDir;

const STUDY: &str = include_str!("fixtures/typed_box_loaded_static.fea");

/// The identity of one run's results: definition, mesh, step run ids, the bytes of every
/// persisted solver state, and the assessed scalar.
#[derive(Debug, PartialEq)]
struct ResultIdentity {
    definition: String,
    mesh: String,
    step_run_ids: Value,
    state_digests: Vec<String>,
    assessed: Vec<f64>,
}

fn run(policy: Option<(u32, u32)>) -> ResultIdentity {
    let project = TempDir::new().unwrap();
    fs::create_dir_all(project.path().join("geometry")).unwrap();
    fs::write(
        project.path().join("geometry/box.brep"),
        include_bytes!("../../runmat-geometry/io/tests/fixtures/box.brep"),
    )
    .unwrap();
    let source = match policy {
        None => STUDY.to_string(),
        Some((workers, threads)) => format!(
            "execution local {{\n    placement = local\n    resources {{\n        workers = {workers}..{workers}\n        cpu_threads_per_worker = {threads}\n    }}\n}}\n{}",
            STUDY.replace("study box_static {\n", "study box_static {\n    execution = execution.local\n")
        ),
    };
    fs::write(project.path().join("study.fea"), source).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_runmat"))
        .current_dir(project.path())
        .args(["--offline", "fea", "run", "study.fea", "--json"])
        .env_remove("RUNMAT_CONFIG")
        .env(
            "RUNMAT_PACKAGE_CACHE_DIR",
            project.path().join("package-cache"),
        )
        .env(
            "RUNMAT_EXECUTION_STATE_DIR",
            project.path().join("execution-state"),
        )
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{policy:?}: stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let run: Value = serde_json::from_slice(&output.stdout).unwrap();
    let mut state_digests = Vec::new();
    collect_state_digests(&project.path().join(".artifacts/studies"), &mut state_digests);
    state_digests.sort();
    assert!(!state_digests.is_empty(), "the run persisted no solver state");
    ResultIdentity {
        definition: run["study_definition_digest"].as_str().unwrap().to_string(),
        mesh: run["mesh_digest"].as_str().unwrap().to_string(),
        step_run_ids: run["step_run_ids"].clone(),
        state_digests,
        assessed: run["assessments"]
            .as_array()
            .unwrap()
            .iter()
            .map(|assessment| assessment["evidence"]["value"].as_f64().unwrap())
            .collect(),
    }
}

fn collect_state_digests(directory: &Path, digests: &mut Vec<String>) {
    let Ok(entries) = fs::read_dir(directory) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_state_digests(&path, digests);
        } else if path.extension().is_some_and(|extension| extension == "state") {
            digests.push(format!("{:x}", Sha256::digest(fs::read(&path).unwrap())));
        }
    }
}

#[test]
fn a_study_gives_identical_results_across_runs_workers_and_threads() {
    let reference = run(None);
    assert_eq!(reference.assessed.len(), 1);
    assert!(reference.assessed[0] > 0.0);
    // Repeated runs, then explicit worker and thread counts. Placement differs; results do not.
    for policy in [None, None, Some((1, 1)), Some((4, 1)), Some((2, 4))] {
        let identity = run(policy);
        // The definition digest binds the authored execution policy, so compare it only for
        // identical sources; every result identity below must match regardless.
        if policy.is_none() {
            assert_eq!(identity.definition, reference.definition);
        }
        assert_eq!(identity.mesh, reference.mesh, "{policy:?}");
        assert_eq!(identity.state_digests, reference.state_digests, "{policy:?}");
        assert_eq!(
            identity.assessed.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
            reference.assessed.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
            "{policy:?}"
        );
        if policy.is_none() {
            assert_eq!(identity.step_run_ids, reference.step_run_ids);
        }
    }
}
