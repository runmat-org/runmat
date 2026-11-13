//! Shared program, host response and reopened immutable store, not kernel-only transport.
use super::*;
use runmat_analysis_execution::{
    execute_serial_analysis, import_nonlinear_solid_block, AnalysisHostResponse,
};
use runmat_execution_artifact::{cache::FilesystemObjectStore, ProgramExecutionResponse};

fn run(
    store: &mut FilesystemObjectStore,
    task: &PreparedNonlinearSolidBlockTask,
) -> ProgramExecutionResponse {
    task.program.write_to(store).unwrap();
    let done = execute_serial_analysis(
        &task.program.request,
        store,
        &super::super::super::NativeAnalysisKernel,
        task.output_policy.chunk_bytes(),
        task.output_policy.inventory_limits(),
    )
    .unwrap();
    AnalysisHostResponse::completed(&task.host, &done)
        .unwrap()
        .program_response()
}

#[test]
fn generic_native_program_and_cold_store_preserve_original_checkpoint_lineage() {
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
        let path = directory.path().join("original-j2");
        let limits = first.output_policy.inventory_limits();
        let mut store = FilesystemObjectStore::open(&path, limits.max_object_bytes).unwrap();
        let response = run(&mut store, &first);
        drop(store);
        let store = FilesystemObjectStore::open(&path, limits.max_object_bytes).unwrap();
        let cold = import_nonlinear_solid_block(&store, response, &first).unwrap();
        cold.numerical_receipt
            .validate_against(&workload, &cold.block)
            .unwrap();
        assert!(!cold.block.complete);
        assert!(
            runmat_analysis_fea::execution::nonlinear_solid_state_publication_requests(
                &workload,
                &cold.block
            )
            .is_err()
        );
        let next =
            NonlinearSolidBlockRequest::continue_original(&workload, &cold.block, 3).unwrap();
        assert_eq!(next.block_ordinal, 1);
        assert_eq!(next.first_output_ordinal, 10);
        let original_alpha = cold
            .block
            .result
            .equivalent_plastic_strain_snapshots
            .last()
            .unwrap()
            .clone();
        let prefix_residual = cold
            .numerical_receipt
            .receipt()
            .accepted_relative_residual_maximum;
        let prefix_increment = cold
            .numerical_receipt
            .receipt()
            .accepted_relative_increment_maximum;
        drop(first);
        drop(store);
        let resumed = prepare(&workload, &mesh, next, Some(cold.objects));
        let mut store = FilesystemObjectStore::open(&path, limits.max_object_bytes).unwrap();
        let response = run(&mut store, &resumed);
        drop(store);
        let store = FilesystemObjectStore::open(&path, limits.max_object_bytes).unwrap();
        let terminal = import_nonlinear_solid_block(&store, response, &resumed).unwrap();
        terminal
            .numerical_receipt
            .validate_against(&workload, &terminal.block)
            .unwrap();
        assert!(terminal.block.complete);
        assert!(
            NonlinearSolidBlockRequest::continue_original(&workload, &terminal.block, 3).is_err()
        );
        assert!(
            terminal
                .numerical_receipt
                .receipt()
                .accepted_relative_residual_maximum
                >= prefix_residual
        );
        assert!(
            terminal
                .numerical_receipt
                .receipt()
                .accepted_relative_increment_maximum
                >= prefix_increment
        );
        assert_eq!(
            terminal.block.result.initial_equivalent_plastic_strain,
            original_alpha
        );
        assert!(terminal
            .block
            .result
            .accepted_state
            .point_history()
            .all(|(_, _, p)| (p.equivalent_plastic_strain - 0.02).abs() < 2e-9));
    }
}
