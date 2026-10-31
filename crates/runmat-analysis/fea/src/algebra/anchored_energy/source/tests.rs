use super::*;
use crate::algebra::{MatrixProperties, OperatorRole};

fn chain() -> RealSparseMatrix {
    RealSparseMatrix::from_dense(
        &[1., 0., 0., 0., 2., -1., 0., -1., 1.],
        3,
        OperatorRole::Stiffness,
        MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
    )
    .unwrap()
}
fn source() -> AnchoredEnergySource {
    let widths = [2, 2].into_iter();
    let bytes = AnchoredEnergySource::resident_bytes_for_widths(widths.clone()).unwrap();
    AnchoredEnergySource::new(widths, bytes).unwrap()
}
#[test]
fn inventory_admits_before_allocation_and_requires_complete_original_source() {
    let widths = [2, 2].into_iter();
    let bytes = AnchoredEnergySource::resident_bytes_for_widths(widths.clone()).unwrap();
    assert!(matches!(
        AnchoredEnergySource::new(widths, bytes - 1),
        Err(EnergyError::Admission { .. })
    ));
    assert!(AnchoredEnergySource::resident_bytes_for_widths([1, 33].into_iter()).is_err());
    let mut incomplete = source();
    assert!(incomplete.capture_next(&[0, 1, 2]).is_err());
    incomplete
        .capture_next(&[0, 1])
        .unwrap()
        .copy_from_slice(&[1., -1., -1., 1.]);
    assert_eq!(
        incomplete
            .prove(&chain(), &[true, false, false], 1.)
            .unwrap_err(),
        EnergyError::InvalidBlock
    );
}
#[test]
fn original_capture_is_normalized_once_and_its_residency_is_charged_to_proof() {
    let mut captured = source();
    let bytes = captured.resident_bytes;
    for nodes in [[0, 1], [1, 2]] {
        captured
            .capture_next(&nodes)
            .unwrap()
            .copy_from_slice(&[4., -4., -4., 4.]);
    }
    assert!(captured.capture_next(&[0, 1]).is_err());
    let EnergyVerdict::Verified(certificate) = captured
        .prove(&chain(), &[true, false, false], 0.25)
        .unwrap()
    else {
        panic!("exact chain")
    };
    assert!(certificate.resident_bytes() > bytes);
    assert!(certificate.lower_bound() > 0. && certificate.lower_bound() < (3. - 5_f64.sqrt()) / 2.);
    let mut corrupt = source();
    for nodes in [[0, 1], [1, 2]] {
        corrupt
            .capture_next(&nodes)
            .unwrap()
            .copy_from_slice(&[4., -4., -4., 4.]);
    }
    assert!(matches!(
        corrupt.prove(&chain(), &[true, false, false], 0.5).unwrap(),
        EnergyVerdict::Indeterminate { .. }
    ));
}
#[test]
fn cancelled_capture_and_proof_do_not_publish_a_certificate() {
    let mut captured = source();
    let mut complete = source();
    for nodes in [[0, 1], [1, 2]] {
        complete
            .capture_next(&nodes)
            .unwrap()
            .copy_from_slice(&[1., -1., -1., 1.]);
    }
    let token = crate::progress::FeaCancellationToken::new();
    token.cancel();
    let _guard =
        crate::progress::replace_fea_progress_context(crate::progress::FeaProgressContext {
            cancellation: token,
            ..Default::default()
        });
    assert_eq!(
        complete
            .prove(&chain(), &[true, false, false], 1.)
            .unwrap_err(),
        EnergyError::Cancelled
    );
    assert_eq!(
        captured.capture_next(&[0, 1]).unwrap_err(),
        EnergyError::Cancelled
    );
    assert!(matches!(
        AnchoredEnergySource::new([2].into_iter(), 4096),
        Err(EnergyError::Cancelled)
    ));
    assert!(matches!(
        AnchoredEnergyCheck::new(
            &chain(),
            &[true, false, false],
            CoercivityWorkEnvelope::derive(chain().pattern()).unwrap()
        ),
        Err(EnergyError::Cancelled)
    ));
}

#[test]
fn retained_source_cannot_be_hidden_inside_the_original_operator_workspace() {
    let mut captured = AnchoredEnergySource::new(std::iter::repeat_n(2, 100), 10000).unwrap();
    for _ in 0..100 {
        captured
            .capture_next(&[0, 1])
            .unwrap()
            .copy_from_slice(&[1., -1., -1., 1.]);
    }
    let matrix = RealSparseMatrix::diagonal(
        vec![1., 100.],
        OperatorRole::Stiffness,
        MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
    )
    .unwrap();
    let envelope = CoercivityWorkEnvelope::derive(matrix.pattern()).unwrap();
    let prescribed = [true, false];
    let Err(EnergyError::Admission {
        resident,
        resident_limit,
        ..
    }) = AnchoredEnergyCheck::new_projected(
        &matrix,
        EnergyProjection::Identity(&prescribed),
        envelope.into(),
        captured.resident_bytes + std::mem::size_of_val(&prescribed) as u64,
        captured.values.len() as u64,
    )
    else {
        panic!("full source residency must be admitted")
    };
    assert!(resident > resident_limit);
    // A source-aware reservation admits the actual contributions beside that
    // workspace; it does not hide them inside the smaller operator-only limit.
    let demand = SourceEnergyDemand::derive(
        envelope,
        prescribed.len(),
        std::mem::size_of_val(&prescribed) as u64,
        std::iter::repeat_n(2, 100),
    )
    .unwrap();
    let EnergyVerdict::Verified(proof) = captured.prove(&matrix, &prescribed, 1.).unwrap() else {
        panic!("the complete grounded source must certify its assembled operator")
    };
    assert!(proof.resident_bytes() > envelope.energy_resident_limit());
    assert!(proof.resident_bytes() <= demand.resident_bytes());
    assert!(proof.scalar_operations() <= demand.scalar_operations());
    assert!(proof.lower_bound() > 0. && proof.lower_bound() <= 100.);
}
