use super::*;

#[test]
fn direct_identity_elimination_matches_dense_reference_for_every_mask() {
    let source = RealSparseMatrix::try_new(
        SparsePattern::try_new(4, 4, vec![0, 2, 4, 5, 6], vec![1, 2, 0, 1, 3, 2]).unwrap(),
        vec![2.0, 0.0, 3.0, 4.0, 5.0, 6.0],
        OperatorRole::Conductivity,
        MatrixProperties::GENERAL,
        vec![ContributionProvenance {
            domain: ContributionDomain::Thermal,
            contribution_id: ContributionId("source".into()),
        }],
    )
    .unwrap();
    for bits in 0..16 {
        let mask = (0..4)
            .map(|index| bits & (1 << index) != 0)
            .collect::<Vec<_>>();
        let mut dense = source.to_dense();
        for row in 0..4 {
            for column in 0..4 {
                if mask[row] || mask[column] {
                    dense[row * 4 + column] = if row == column && mask[row] { 1.0 } else { 0.0 };
                }
            }
        }
        let actual = source.with_identity_constraints(&mask).unwrap();
        assert_eq!(actual.to_dense(), dense, "mask {bits}");
        assert!(actual.values().iter().all(|value| *value != 0.0));
        assert_eq!(actual.contributions[0], source.contributions[0]);
        assert_eq!(
            actual.contributions[1].domain,
            ContributionDomain::Constraint
        );
        let reference =
            RealSparseMatrix::from_dense(&dense, 4, source.role, source.properties).unwrap();
        assert_eq!(
            actual.numerical_identity().unwrap(),
            reference.numerical_identity().unwrap()
        );
    }
    assert_eq!(
        source.with_identity_constraints(&[true]),
        Err(RealSparseMatrixError::ShapeMismatch)
    );
    let empty =
        RealSparseMatrix::diagonal(vec![], OperatorRole::Mass, MatrixProperties::GENERAL).unwrap();
    assert!(empty
        .with_identity_constraints(&[])
        .unwrap()
        .values()
        .is_empty());
}

#[test]
fn sparse_finalization_preserves_cancelled_rows_order_and_provenance() {
    let mut assembler =
        RealSparseAssembler::new(3, 4, OperatorRole::Conductivity, MatrixProperties::GENERAL);
    assembler.add(0, 3, 2.0);
    assembler.add(0, 1, 4.0);
    assembler.add(0, 3, -2.0);
    assembler.add(2, 2, 5.0);
    assembler.add(2, 0, 1.0);
    let provenance = ContributionProvenance {
        domain: ContributionDomain::Thermal,
        contribution_id: ContributionId("conductivity".into()),
    };
    assembler.add_contribution(provenance.clone());
    let matrix = assembler.finish().unwrap();
    let expected = RealSparseMatrix::try_new(
        SparsePattern::try_new(3, 4, vec![0, 1, 1, 3], vec![1, 0, 2]).unwrap(),
        vec![4.0, 1.0, 5.0],
        OperatorRole::Conductivity,
        MatrixProperties::GENERAL,
        vec![provenance],
    )
    .unwrap();
    assert_eq!(matrix, expected);
    assert_eq!(
        matrix.numerical_identity().unwrap(),
        expected.numerical_identity().unwrap()
    );
    assert_eq!(
        matrix.values_identity().unwrap(),
        expected.values_identity().unwrap()
    );
}

#[test]
fn sparse_finalization_handles_empty_payload_and_rejects_sum_overflow() {
    let empty =
        RealSparseAssembler::square(0, OperatorRole::Conductivity, MatrixProperties::GENERAL)
            .finish()
            .unwrap();
    assert!(empty.values.is_empty());
    assert_eq!(empty.pattern.row_offsets(), &[0]);
    let mut overflow =
        RealSparseAssembler::square(1, OperatorRole::Conductivity, MatrixProperties::GENERAL);
    overflow.add(0, 0, f64::MAX);
    overflow.add(0, 0, f64::MAX);
    assert_eq!(
        overflow.finish(),
        Err(RealSparseMatrixError::NonFiniteValue { entry: 0 })
    );
    assert_eq!(
        reserve_payload::<f64>(usize::MAX),
        Err(RealSparseMatrixError::StorageAllocation {
            elements: usize::MAX
        })
    );
}

#[test]
fn value_identity_is_separate_from_pattern_identity() {
    let first = RealSparseMatrix::diagonal(
        vec![1.0, 2.0],
        OperatorRole::Stiffness,
        MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
    )
    .unwrap();
    let changed_values = RealSparseMatrix::diagonal(
        vec![1.0, 3.0],
        OperatorRole::Stiffness,
        MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
    )
    .unwrap();
    let changed_metadata = RealSparseMatrix::diagonal(
        vec![1.0, 2.0],
        OperatorRole::Mass,
        MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
    )
    .unwrap();

    assert_eq!(
        first.pattern_identity().unwrap(),
        changed_values.pattern_identity().unwrap()
    );
    assert_ne!(
        first.values_identity().unwrap(),
        changed_values.values_identity().unwrap()
    );
    assert_ne!(
        first.values_identity().unwrap(),
        changed_metadata.values_identity().unwrap()
    );

    let changed_provenance = first.clone().with_contribution(ContributionProvenance {
        domain: ContributionDomain::TimeIntegration,
        contribution_id: ContributionId("sample-1".to_string()),
    });
    assert_ne!(
        first.values_identity().unwrap(),
        changed_provenance.values_identity().unwrap()
    );
    assert_eq!(
        first.numerical_identity().unwrap(),
        changed_provenance.numerical_identity().unwrap()
    );
}

#[test]
fn assembler_reports_invalid_admission_without_panicking() {
    let mut assembler =
        RealSparseAssembler::square(1, OperatorRole::Stiffness, MatrixProperties::GENERAL);
    assembler.add(1, 0, 1.0);
    assert_eq!(
        assembler.finish(),
        Err(RealSparseMatrixError::ShapeMismatch)
    );
}

#[test]
fn fluid_provenance_roundtrips_without_changing_numerical_identity() {
    let original = RealSparseMatrix::diagonal(
        vec![2.0, 3.0],
        OperatorRole::EffectiveTangent,
        MatrixProperties::GENERAL,
    )
    .unwrap();
    let fluid = original.clone().with_contribution(ContributionProvenance {
        domain: ContributionDomain::Fluid,
        contribution_id: ContributionId("cell-momentum".into()),
    });
    let wire = serde_json::to_value(&fluid).unwrap();
    assert_eq!(wire["contributions"][0]["domain"], "fluid");
    let restored: RealSparseMatrix = serde_json::from_value(wire).unwrap();
    assert_eq!(restored, fluid);
    assert_ne!(
        original.values_identity().unwrap(),
        fluid.values_identity().unwrap()
    );
    assert_eq!(
        original.numerical_identity().unwrap(),
        fluid.numerical_identity().unwrap()
    );
}
#[test]
fn immutable_clones_share_numeric_storage_and_cold_decode_revalidates() {
    let original = RealSparseMatrix::diagonal(
        vec![2.0, 3.0, 4.0],
        OperatorRole::Conductivity,
        MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
    )
    .unwrap();
    let cloned = original.clone();
    assert_eq!(original.values().as_ptr(), cloned.values().as_ptr());
    assert_eq!(
        original.pattern().row_offsets().as_ptr(),
        cloned.pattern().row_offsets().as_ptr()
    );
    assert_eq!(
        original.pattern().column_indices().as_ptr(),
        cloned.pattern().column_indices().as_ptr()
    );
    let bytes = serde_json::to_vec(&original).unwrap();
    let cold: RealSparseMatrix = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(cold, original);
    assert_eq!(
        cold.numerical_identity().unwrap(),
        original.numerical_identity().unwrap()
    );
    assert_eq!(serde_json::to_vec(&cold).unwrap(), bytes);
    let mut corrupt = serde_json::to_value(&original).unwrap();
    corrupt["values"] = serde_json::json!([2.0]);
    assert!(serde_json::from_value::<RealSparseMatrix>(corrupt).is_err());
    drop(original);
    assert_eq!(cloned.values(), &[2.0, 3.0, 4.0]);
}
