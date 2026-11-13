use super::*;
use crate::algebra::{
    anchored_energy::{AnchoredEnergySource, EnergyVerdict},
    MatrixProperties, OperatorRole, RealSparseMatrix,
};

#[test]
fn original_ten_vertex_energy_proves_the_one_dof_selection_with_bounded_work() {
    let matrix = RealSparseMatrix::diagonal(
        vec![9.],
        OperatorRole::Stiffness,
        MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
    )
    .unwrap();
    let base = CoercivityWorkEnvelope::derive(matrix.pattern()).unwrap();
    let rows = [
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        Some(0),
    ];
    let demand = SourceEnergyDemand::derive(
        base,
        rows.len(),
        std::mem::size_of_val(&rows) as u64,
        [10].into_iter(),
    )
    .unwrap();
    let source_bytes = AnchoredEnergySource::resident_bytes_for_widths([10].into_iter()).unwrap();
    let mut source = AnchoredEnergySource::new([10].into_iter(), source_bytes).unwrap();
    let values = source
        .capture_next(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        .unwrap();
    for (index, value) in values.iter_mut().enumerate() {
        *value = if index / 10 == index % 10 { 9. } else { -1. };
    }
    let EnergyVerdict::Verified(proof) = source.prove_selection(&matrix, &rows).unwrap() else {
        panic!("exact complete-graph local energy must prove its grounded selection")
    };
    // Independent exact reduced eigenvalue is9. This demand does not establish that fact.
    assert!(proof.lower_bound() > 0. && proof.lower_bound() <= 9.);
    assert!(proof.resident_bytes() > base.resident_bytes_limit());
    assert!(proof.resident_bytes() <= demand.resident_bytes());
    assert!(proof.scalar_operations() <= demand.scalar_operations());
    assert_eq!(
        proof.operator_identity(),
        matrix.numerical_identity().unwrap()
    );
}

#[test]
fn source_inventory_rejects_invalid_overflow_and_cancelled_inputs() {
    let matrix = RealSparseMatrix::diagonal(
        vec![1.],
        OperatorRole::Stiffness,
        MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
    )
    .unwrap();
    let base = CoercivityWorkEnvelope::derive(matrix.pattern()).unwrap();
    assert!(matches!(
        SourceEnergyDemand::derive(base, 10, 0, [33].into_iter()),
        Err(EnergyError::InvalidBlock)
    ));
    assert!(matches!(
        SourceEnergyDemand::derive(base, 10, u64::MAX, [10].into_iter()),
        Err(EnergyError::Arithmetic)
    ));
    let token = crate::progress::FeaCancellationToken::new();
    token.cancel();
    let _guard =
        crate::progress::replace_fea_progress_context(crate::progress::FeaProgressContext {
            cancellation: token,
            ..Default::default()
        });
    assert!(matches!(
        SourceEnergyDemand::derive(base, 10, 0, [10].into_iter()),
        Err(EnergyError::Cancelled)
    ));
}
