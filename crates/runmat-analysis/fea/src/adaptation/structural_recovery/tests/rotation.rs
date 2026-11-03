//! Coordinate-frame invariance of the physical symmetric-stress tensor norm.
use runmat_analysis_core::{AnalysisFieldLayout, ObservationQuantity, FieldLocation};
use super::*;

#[test]
fn recovery_indicators_and_marking_preserve_stress_under_rotation() {
    for quadratic in [false, true] {
        let mesh = if quadratic {
            crate::topology::test_meshes::quadratic_slab_mesh(1)
        } else {
            two_tetrahedron_artifact()
        };
        let count = mesh.topology.volume_elements.len();
        for integration_points in [false, true] {
            let points = if integration_points { 4 } else { 1 };
            let name = if integration_points {
                FEA_FIELD_STRUCTURAL_INTEGRATION_POINT_STRESS
            } else {
                FEA_FIELD_STRUCTURAL_STRESS
            };
            for axis in 0..3 {
                let other = (axis + 1) % 3;
                let shear = match (axis.min(other), axis.max(other)) {
                    (0, 1) => 5,
                    (0, 2) => 4,
                    (1, 2) => 3,
                    _ => unreachable!(),
                };
                let mut original = Vec::new();
                let mut rotated = Vec::new();
                for element in 0..count {
                    for point in 0..points {
                        let amplitude = (element + 1) as f64 + point as f64 / 5.0;
                        let mut before = [0.0; 6];
                        before[axis] = 2.0 * amplitude;
                        // Q diag(2*a,0) Q^T at45degrees has [a,a;a,a].
                        // This exact tensor transformation is independent of estimator code.
                        let mut after = [0.0; 6];
                        after[axis] = amplitude;
                        after[other] = amplitude;
                        after[shear] = amplitude;
                        original.extend(before);
                        rotated.extend(after);
                    }
                }
                let shape = if integration_points {
                    vec![count * 4, 6]
                } else {
                    vec![count, 6]
                };
                let before = estimate_structural_recovery_error(
                    &mesh,
                    "elements",
                    &AnalysisField::host_f64(name, if integration_points { AnalysisFieldLayout::symmetric_tensor3(ObservationQuantity::Stress, FieldLocation::IntegrationPoint) } else { AnalysisFieldLayout::symmetric_tensor3(ObservationQuantity::Stress, FieldLocation::Element) }, shape.clone(), original),
                    StructuralRecoveryEstimatorOptions::default(),
                )
                .unwrap();
                let after = estimate_structural_recovery_error(
                    &mesh,
                    "elements",
                    &AnalysisField::host_f64(name, if integration_points { AnalysisFieldLayout::symmetric_tensor3(ObservationQuantity::Stress, FieldLocation::IntegrationPoint) } else { AnalysisFieldLayout::symmetric_tensor3(ObservationQuantity::Stress, FieldLocation::Element) }, shape, rotated),
                    StructuralRecoveryEstimatorOptions::default(),
                )
                .unwrap();
                assert!(before.total_error > 0.0);
                assert!((after.total_error-before.total_error).abs()<1.0e-12*before.total_error,
                    "quadratic={quadratic} points={integration_points} axis={axis} original={} rotated={}",before.total_error,after.total_error);
                assert!(
                    (before.reference_norm - after.reference_norm).abs()
                        < 1e-12 * before.reference_norm
                );
                assert!(
                    (before.relative_error().unwrap() - after.relative_error().unwrap()).abs()
                        < 1e-12
                );
                assert_eq!(
                    before.marked_element_identities,
                    after.marked_element_identities
                );
                for (a, b) in before.indicators.iter().zip(after.indicators) {
                    assert_eq!(a.element_stable_identity, b.element_stable_identity);
                    assert!((a.error - b.error).abs() < 1.0e-12 * before.total_error);
                }
            }
        }
    }
}

#[test]
fn physical_shear_and_cold_norm_identity_are_explicit() {
    let mesh = two_tetrahedron_artifact();
    for shear in 3..6 {
        let mut values = vec![0.0; 12];
        values[6 + shear] = 2.0;
        let estimate = estimate_structural_recovery_error(
            &mesh,
            "elements",
            &AnalysisField::host_f64(FEA_FIELD_STRUCTURAL_STRESS, AnalysisFieldLayout::symmetric_tensor3(ObservationQuantity::Stress, FieldLocation::Element), vec![2, 6], values),
            StructuralRecoveryEstimatorOptions::default(),
        )
        .unwrap();
        // The scalar-normal volume integral is 4/15. Physical shear appears
        // twice in the symmetric 3x3 tensor, giving squared discrepancy 8/15.
        assert!((estimate.total_error - (8.0_f64 / 15.0).sqrt()).abs() < 1.0e-12);
        assert!((estimate.reference_norm - (8.0 / 3.0_f64).sqrt()).abs() < 1e-12);
        assert_eq!(
            estimate.norm,
            StructuralRecoveryNorm::StressTensorFrobeniusL2
        );
        let value = serde_json::to_value(&estimate).unwrap();
        let decoded: StructuralRecoveryEstimate = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(decoded, estimate);
        decoded.validate().unwrap();
        let mut missing = value.clone();
        missing.as_object_mut().unwrap().remove("norm");
        assert!(serde_json::from_value::<StructuralRecoveryEstimate>(missing).is_err());
        let mut foreign = value;
        foreign["norm"] = serde_json::json!("component_vector_l2");
        assert!(serde_json::from_value::<StructuralRecoveryEstimate>(foreign).is_err());
    }
}

#[test]
fn finite_stress_overflow_cannot_become_zero_recovery_error() {
    let mesh = two_tetrahedron_artifact();
    let mut values = vec![0.0; 8 * 6];
    for point in 0..8 {
        values[point * 6] = (point / 4 + 1) as f64 * 1.0e200;
    }
    assert_eq!(
        estimate_structural_recovery_error(
            &mesh,
            "elements",
            &AnalysisField::host_f64(
                FEA_FIELD_STRUCTURAL_INTEGRATION_POINT_STRESS, AnalysisFieldLayout::symmetric_tensor3(ObservationQuantity::Stress, FieldLocation::IntegrationPoint),
                vec![8, 6],
                values
            ),
            StructuralRecoveryEstimatorOptions::default()
        ),
        Err(StructuralRecoveryEstimatorError::InvalidStressField)
    );
}

#[test]
fn recovery_reference_norm_integrates_original_tensor_in_both_layouts_and_orders() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = artifact(order);
        for points in [1, 4] {
            for scale in [1.0e-5, 1.0, 1.0e5] {
                let row = [2.0 * scale, 0.0, 0.0, 0.0, 0.0, 3.0 * scale];
                let field = AnalysisField::host_f64(
                    if points == 1 {
                        FEA_FIELD_STRUCTURAL_STRESS
                    } else {
                        FEA_FIELD_STRUCTURAL_INTEGRATION_POINT_STRESS
                    }, if points == 1 { AnalysisFieldLayout::symmetric_tensor3(ObservationQuantity::Stress, FieldLocation::Element) } else { AnalysisFieldLayout::symmetric_tensor3(ObservationQuantity::Stress, FieldLocation::IntegrationPoint) },
                    vec![points, 6],
                    (0..points).flat_map(|_| row).collect(),
                );
                let estimate = estimate_structural_recovery_error(
                    &mesh,
                    "elements",
                    &field,
                    StructuralRecoveryEstimatorOptions::default(),
                )
                .unwrap();
                let expected = scale * (22.0 / 6.0_f64).sqrt();
                assert!((estimate.reference_norm / expected - 1.0).abs() < 1.0e-12);
                assert!(estimate.relative_error().unwrap() < 1.0e-14);
                let mut missing = serde_json::to_value(&estimate).unwrap();
                missing.as_object_mut().unwrap().remove("reference_norm");
                assert!(serde_json::from_value::<StructuralRecoveryEstimate>(missing).is_err());
            }
        }
    }
}
