//! Recover original partitions through the shared field owner, after producer termination.
use super::*;
use runmat_analysis_fea::results::field::{
    reconstruct_recovered_field, RecoveredField, RecoveredFieldEntityCounts,
    RecoveredFieldPartition,
};
pub(super) fn original(
    done: &CompletedThermalStudy,
    mesh: &SolverMeshArtifact,
    ordinal: usize,
    index: usize,
) -> RecoveredField {
    let entry = &done.result.manifest.fields[ordinal][index];
    let partitions = entry
        .partitions
        .iter()
        .map(|d| {
            RecoveredFieldPartition::from_canonical_bytes(
                &done
                    .result
                    .objects
                    .partition_payload(d.partition.digest())
                    .unwrap(),
            )
            .unwrap()
        })
        .collect();
    let counts = RecoveredFieldEntityCounts {
        nodes: mesh.topology.nodes.len(),
        volume_elements: mesh.topology.volume_elements.len(),
        oriented_faces: 0,
        volume_integration_points: 0,
        oriented_edges: 0,
        beam_elements: 0,
        shell_elements: 0,
        connectors: 0,
        constraint_groups: 0,
    };
    reconstruct_recovered_field(
        entry.field,
        entry.definition.clone(),
        entry.sample,
        partitions,
        entry.definition.stored_entities(counts),
    )
    .unwrap()
}
pub(super) fn acceptance(done: &CompletedThermalStudy, solved: usize) {
    let m = &done.result.manifest;
    assert_eq!(m.nonlinear_evidence.len(), solved);
    assert_eq!(m.residual_target, TARGET);
    for e in &m.nonlinear_evidence {
        assert!((1..=32).contains(&e.iteration_count));
        assert!(e.backtracking_count <= 32 * 24);
        assert!(e.relative_equilibrium_residual <= TARGET);
        assert!(e.relative_increment <= TARGET);
    }
    assert!(m
        .energy_balances
        .iter()
        .all(|b| b.relative_residual <= TARGET));
    done.result
        .numerical_receipt
        .validate_against(&done.workload, m)
        .unwrap();
}
