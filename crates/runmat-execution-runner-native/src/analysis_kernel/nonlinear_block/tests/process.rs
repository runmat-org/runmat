//! Real producer termination with the shared native program host and immutable store.
use super::*;
use runmat_analysis_execution::import_nonlinear_solid_block;
use runmat_execution_artifact::{
    cache::FilesystemObjectStore, ProgramExecutionRequest, ProgramExecutionResponse,
};
use std::{path::Path, process::Command};

const ROOT_ENV: &str = "RUNMAT_J2_PROCESS_WITNESS_ROOT";

#[test]
#[ignore = "subprocess entrypoint, invoked by original-history and publication process witnesses"]
fn child_native_program() {
    let root = std::path::PathBuf::from(std::env::var_os(ROOT_ENV).expect("witness root"));
    let request: ProgramExecutionRequest =
        serde_json::from_slice(&std::fs::read(root.join("request.json")).unwrap()).unwrap();
    let limits = crate::NativeAnalysisHostLimits {
        chunk_bytes: 4096,
        inventory: ObjectInventoryLimits::default(),
    };
    let mut store =
        FilesystemObjectStore::open(root.join("store"), limits.inventory.max_object_bytes).unwrap();
    let response = crate::execute_analysis_program_request(
        &request,
        &mut store,
        &crate::NativeAnalysisKernel,
        limits,
    );
    response.validate_against(&request).unwrap();
    std::fs::write(
        root.join("response.json"),
        serde_json::to_vec(&response).unwrap(),
    )
    .unwrap();
}

pub(super) fn run_terminated(
    root: &Path,
    program: &runmat_analysis_execution::PreparedAnalysisProgram,
    policy: AnalysisOutputPolicy,
) -> ProgramExecutionResponse {
    assert_eq!(policy.chunk_bytes(), 4096);
    let limits = policy.inventory_limits();
    let mut store =
        FilesystemObjectStore::open(root.join("store"), limits.max_object_bytes).unwrap();
    program.write_to(&mut store).unwrap();
    drop(store);
    std::fs::write(
        root.join("request.json"),
        serde_json::to_vec(&program.request).unwrap(),
    )
    .unwrap();
    let response_path = root.join("response.json");
    if response_path.exists() {
        std::fs::remove_file(&response_path).unwrap();
    }
    let child_name = format!(
        "{}::child_native_program",
        module_path!().split_once("::").unwrap().1
    );
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", &child_name, "--ignored", "--nocapture"])
        .env(ROOT_ENV, root)
        .spawn()
        .unwrap();
    let producer_pid = child.id();
    assert_ne!(producer_pid, std::process::id());
    // wait() confirms the actual producer exited before any field/history import.
    assert!(
        child.wait().unwrap().success(),
        "producer {producer_pid} failed"
    );
    serde_json::from_slice(&std::fs::read(response_path).unwrap()).unwrap()
}

// witness: FEA-NONLINEAR-STATE-001/producer_exit @ small_strain_j2/tet4|tet10/native_process;
//   boundary: Native producer actually exits before cold consumers resume; homogeneous uncoupled J2
//   only.
#[test]
fn terminated_producers_resume_original_history() {
    for quadratic in [false, true] {
        let (workload, mesh) = fixture(quadratic);
        let request = NonlinearSolidBlockRequest {
            workload: workload.identity().unwrap(),
            topology: Digest::from_bytes(*mesh.canonical_digest.bytes()),
            block_ordinal: 0,
            first_output_ordinal: 1,
            accepted_block_limit: 9,
            predecessor: None,
        };
        let first = prepare(&workload, &mesh, request.clone(), None);
        let directory = tempfile::tempdir().unwrap();
        let response = run_terminated(directory.path(), &first.program, first.output_policy);
        let limits = first.output_policy.inventory_limits();
        let store =
            FilesystemObjectStore::open(directory.path().join("store"), limits.max_object_bytes)
                .unwrap();
        let cold = import_nonlinear_solid_block(&store, response, &first).unwrap();
        assert!(!cold.block.complete);
        let retained = cold
            .retained_storage_bytes(
                workload.model.sequence.steps[0]
                    .accuracy
                    .resources
                    .resident_byte_budget(),
            )
            .unwrap();
        let payload = cold.block.canonical_bytes().unwrap().len() as u64
            + cold.numerical_receipt.canonical_bytes().unwrap().len() as u64;
        assert!(retained > payload);
        assert!(cold.retained_storage_bytes(1).is_err());
        let alpha = (0.9 * 5.6 - 5.0) / 30.0;
        assert!(cold
            .block
            .result
            .accepted_state
            .point_history()
            .all(|(_, _, p)| (p.equivalent_plastic_strain - alpha).abs() < 2e-9));
        let next = NonlinearSolidBlockRequest {
            block_ordinal: 1,
            first_output_ordinal: cold.block.last_output_ordinal().unwrap() + 1,
            accepted_block_limit: 3,
            predecessor: Some(NonlinearSolidBlockReference {
                artifact_digest: cold.block.identity,
                controller_digest: cold.block.checkpoint.content_digest(),
            }),
            ..request
        };
        let initial_alpha = cold
            .block
            .result
            .equivalent_plastic_strain_snapshots
            .last()
            .unwrap()
            .clone();
        drop(first);
        drop(store);
        let resumed = prepare(&workload, &mesh, next, Some(cold.objects));
        let response = run_terminated(directory.path(), &resumed.program, resumed.output_policy);
        let store =
            FilesystemObjectStore::open(directory.path().join("store"), limits.max_object_bytes)
                .unwrap();
        let terminal = import_nonlinear_solid_block(&store, response, &resumed).unwrap();
        assert!(terminal.block.complete);
        assert_eq!(
            terminal.block.result.initial_equivalent_plastic_strain,
            initial_alpha
        );
        for (_, _, p) in terminal.block.result.accepted_state.point_history() {
            assert!((p.equivalent_plastic_strain - 0.02).abs() < 2e-9);
            for (actual, exact) in p
                .engineering_plastic_strain
                .iter()
                .zip([0.02, -0.01, -0.01, 0.0, 0.0, 0.0])
            {
                assert!((actual - exact).abs() < 2e-9);
            }
        }
    }
}
