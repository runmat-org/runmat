//! Nonuniform original history through every native phase and the cold runtime store.
use super::*;

#[test]
fn layered_original_history_survives_whole_runtime_publication_after_publisher_exit() {
    for quadratic in [false, true] {
        let (workload, mesh) = fixture::prepare(quadratic);
        let preparation = super::super::coordinator::preparation_from(workload, mesh.clone(), true);
        let policy = preparation.output_policy;
        let (completed, directory) = super::super::coordinator::execute_preparation(preparation);
        assert_eq!(completed.closures.len(), 13);
        assert_eq!(completed.states.len(), 3);
        assert_eq!(completed.terminal_usage.evidence.attempts.len(), 13);
        assert_eq!(completed.baseline_usage.evidence.attempts.len(), 12);
        reference::verify(&mesh, &completed.terminal_block.block, 1.0);
        let store = FilesystemObjectStore::open(
            directory.path().join("store"),
            policy.inventory_limits().max_object_bytes,
        )
        .unwrap();
        super::super::durable::verify_original_root_with(
            &completed,
            &store,
            policy,
            super::super::durable::runtime_publication::history::Reference::Layered,
        );
    }
}
