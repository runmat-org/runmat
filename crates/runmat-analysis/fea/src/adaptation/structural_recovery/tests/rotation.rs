//! Coordinate-frame invariance of the physical symmetric-stress tensor norm.
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
                    &AnalysisField::host_f64(name, shape.clone(), original),
                    StructuralRecoveryEstimatorOptions::default(),
                )
                .unwrap();
                let after = estimate_structural_recovery_error(
                    &mesh,
                    "elements",
                    &AnalysisField::host_f64(name, shape, rotated),
                    StructuralRecoveryEstimatorOptions::default(),
                )
                .unwrap();
                assert!(before.total_error > 0.0);
                assert!((after.total_error-before.total_error).abs()<1.0e-12*before.total_error,
                    "quadratic={quadratic} points={integration_points} axis={axis} original={} rotated={}",before.total_error,after.total_error);
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
            &AnalysisField::host_f64(FEA_FIELD_STRUCTURAL_STRESS, vec![2, 6], values),
            StructuralRecoveryEstimatorOptions::default(),
        )
        .unwrap();
        // The original scalar-normal discrepancy is0.5. Physical shear appears
        // twice in the symmetric3x3 tensor, giving sqrt(2)*0.5.
        assert!((estimate.total_error - 0.5 * 2.0_f64.sqrt()).abs() < 1.0e-12);
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
                FEA_FIELD_STRUCTURAL_INTEGRATION_POINT_STRESS,
                vec![8, 6],
                values
            ),
            StructuralRecoveryEstimatorOptions::default()
        ),
        Err(StructuralRecoveryEstimatorError::InvalidStressField)
    );
}
