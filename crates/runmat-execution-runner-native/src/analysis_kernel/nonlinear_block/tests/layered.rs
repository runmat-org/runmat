//! Independent nonuniform original-history continuation after actual producer exits.
//! An aligned material interface is a separate reference, not a replacement moving-front gate.
use super::*;
use runmat_analysis_execution::import_nonlinear_solid_block;
use runmat_execution_artifact::cache::FilesystemObjectStore;
mod fixture;
mod publication;
pub(super) mod reference;
mod runtime;

// witness: FEA-NONLINEAR-STATE-001/producer_exit @ small_strain_j2/tet4|tet10/native_process;
//   boundary: Native exited-producer aligned nonuniform J2 reference; no uniform moving-front
//   accuracy or contact credit.
#[test]
fn terminated_native_blocks_preserve_nonuniform_material_history() {
    for quadratic in [false, true] {
        let (workload, mesh) = fixture::prepare(quadratic);
        let request = NonlinearSolidBlockRequest {
            workload: workload.identity().unwrap(),
            topology: Digest::from_bytes(*mesh.canonical_digest.bytes()),
            block_ordinal: 0,
            first_output_ordinal: 1,
            accepted_block_limit: 9,
            predecessor: None,
        };
        let first = prepare(&workload, &mesh, request, None);
        let policy = first.output_policy;
        let limits = policy.inventory_limits();
        let directory = tempfile::tempdir().unwrap();
        let response = process::run_terminated(directory.path(), &first.program, policy);
        let store =
            FilesystemObjectStore::open(directory.path().join("store"), limits.max_object_bytes)
                .unwrap();
        let cold = import_nonlinear_solid_block(&store, response, &first).unwrap();
        assert!(!cold.block.complete);
        reference::verify(&mesh, &cold.block, 0.9);
        cold.numerical_receipt
            .validate_against(&workload, &cold.block)
            .unwrap();
        let initial_history = cold
            .block
            .result
            .equivalent_plastic_strain_snapshots
            .last()
            .unwrap()
            .clone();
        let next =
            NonlinearSolidBlockRequest::continue_original(&workload, &cold.block, 3).unwrap();
        assert_eq!(next.first_output_ordinal, 10);
        assert_eq!(next.block_ordinal, 1);
        let prefix_work = cold.block.checkpoint.work();
        drop(first);
        drop(store);
        let resumed = prepare(&workload, &mesh, next, Some(cold.objects));
        cancellation::reject_cancelled(&resumed);
        let response = process::run_terminated(directory.path(), &resumed.program, policy);
        let store =
            FilesystemObjectStore::open(directory.path().join("store"), limits.max_object_bytes)
                .unwrap();
        let terminal = import_nonlinear_solid_block(&store, response, &resumed).unwrap();
        assert!(terminal.block.complete);
        reference::verify(&mesh, &terminal.block, 1.0);
        assert_eq!(
            terminal.block.result.initial_equivalent_plastic_strain,
            initial_history
        );
        assert!(
            terminal.block.checkpoint.work().accepted_increments()
                > prefix_work.accepted_increments()
        );
        terminal
            .numerical_receipt
            .validate_against(&workload, &terminal.block)
            .unwrap();
        assert!(
            NonlinearSolidBlockRequest::continue_original(&workload, &terminal.block, 3).is_err()
        );
        drop(store);
        publication::verify(directory.path(), &resumed, &workload, &terminal);
    }
}
