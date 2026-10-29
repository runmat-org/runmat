//! SIM-24: the same study gives the same result on every run and for every local worker and
//! thread count. Execution placement and timing are evidence, never result identity.
use serde_json::Value;
use std::fs;
use std::path::Path;
use std::process::Command;
use tempfile::TempDir;

#[path = "support/local_offline.rs"]
mod local_offline;
#[path = "support/fea_admission.rs"]
mod fea_admission;

const STUDY: &str = include_str!("fixtures/typed_box_loaded_static.fea");

/// The identity of one run's results: definition, mesh, step run ids, the content digest of every
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
    let mut command = Command::new(env!("CARGO_BIN_EXE_runmat"));
    command
        .current_dir(project.path())
        .args(["--offline", "run", "study.fea", "--json"])
        .env_remove("RUNMAT_CONFIG")
        .env(
            "RUNMAT_PACKAGE_CACHE_DIR",
            project.path().join("package-cache"),
        )
        .env(
            "RUNMAT_EXECUTION_STATE_DIR",
            project.path().join("execution-state"),
        );
    local_offline::configure(&mut command, project.path());
    let _admission = fea_admission::AdmissionServer::for_command(&mut command, project.path());
    // Preserve the authored solver-state location after admission config sets its defaults.
    let output = command
        .env(
            "RUNMAT_FEA_STUDY_ARTIFACT_ROOT",
            project.path().join(".artifacts/studies"),
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
    collect_state_digests(
        &project.path().join(".artifacts/studies"),
        &mut state_digests,
    );
    state_digests.sort();
    assert!(
        !state_digests.is_empty(),
        "the run persisted no solver state"
    );
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
        } else if path
            .extension()
            .is_some_and(|extension| extension == "state")
        {
            // The state header records its numerical content digest; its execution identity
            // also binds the resource plan the step ran under, which is placement evidence.
            let bytes = fs::read(&path).unwrap();
            let marker = b"content_digest";
            let at = bytes
                .windows(marker.len())
                .position(|window| window == marker)
                .expect("state header records its content digest");
            let digest = &bytes[at + marker.len()..];
            let start = digest
                .windows(7)
                .position(|window| window == b"sha256:")
                .unwrap();
            digests.push(String::from_utf8_lossy(&digest[start..start + 71]).into_owned());
        }
    }
}

// witness: FEA-PUBLIC-RESULTS-001/result_digest_determinism @
//   shared_infrastructure/canonical_mixed_topology/public_study; coverage: partial;
//   boundary: A loaded structural static study run through the CLI repeatedly and with explicit
//   worker and thread counts (1x1, 4x1, 2x4) gives identical mesh identity, solver-state content
//   digests and assessed scalar bits; one study, one platform.
// witness: FEA-SCALE-PARALLEL-001/thread_count_determinism @
//   linear_structural/canonical_tetrahedra/parallel_cpu; coverage: partial;
//   boundary: The loaded static box study gives bit-identical state digests and assessed values at
//   serial and host-admissible parallel layouts (up to 8 total threads) through the public
//   command; other families are not run.
#[test]
fn a_study_gives_identical_results_across_runs_workers_and_threads() {
    let reference = run(None);
    assert_eq!(reference.assessed.len(), 1);
    assert!(reference.assessed[0] > 0.0);
    // Match the local admission authority's available-parallelism discovery.
    // The fixture must vary placement without asking a six-core host for eight
    // simultaneous cores; larger hosts still exercise the original 2x4 layout.
    let capacity = std::thread::available_parallelism().map_or(1, usize::from);
    let parallel_workers = capacity.min(4) as u32;
    let threaded_workers = capacity.min(2) as u32;
    let threads_per_worker = (capacity / threaded_workers as usize).clamp(1, 4) as u32;
    // Repeated runs, then explicit worker and thread counts. Placement differs; results do not.
    for policy in [
        None,
        None,
        None,
        None,
        Some((1, 1)),
        Some((parallel_workers, 1)),
        Some((threaded_workers, threads_per_worker)),
    ] {
        let identity = run(policy);
        // The definition digest binds the authored execution policy, so compare it only for
        // identical sources; every result identity below must match regardless.
        if policy.is_none() {
            assert_eq!(identity.definition, reference.definition);
        }
        assert_eq!(identity.mesh, reference.mesh, "{policy:?}");
        assert_eq!(
            identity.state_digests, reference.state_digests,
            "{policy:?}"
        );
        assert_eq!(
            identity
                .assessed
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            reference
                .assessed
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "{policy:?}"
        );
        // Every run is a fresh attempt of the same definition: the step run ids share their
        // definition part and differ only in the attempt suffix.
        if policy.is_none() {
            let definitions = |ids: &serde_json::Value| {
                ids.as_array()
                    .expect("step run ids")
                    .iter()
                    .map(|id| {
                        let id = id.as_str().expect("step run id");
                        id.rsplit_once('_').expect("attempt run id").0.to_string()
                    })
                    .collect::<Vec<_>>()
            };
            assert_eq!(
                definitions(&identity.step_run_ids),
                definitions(&reference.step_run_ids)
            );
            assert_ne!(identity.step_run_ids, reference.step_run_ids);
        }
    }
}
