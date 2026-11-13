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
// witness: FEA-SHARED-RESOURCES-001/shared_immutable_inventories @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Immutable sparse clones share support and numeric values; cold decoding preserves
//   exact wire identity and rejects malformed shape. Transformed operators, solver workspaces and
//   native heap measurements remain separate obligations.
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

/// Discovered-assembler reference: every term's nonzero scaled entries accumulated term by term.
fn scaled_sum_reference(terms: &[(f64, &RealSparseMatrix)]) -> RealSparseMatrix {
    let first = terms[0].1;
    let mut assembler = RealSparseAssembler::new(
        first.rows(),
        first.columns(),
        OperatorRole::Stiffness,
        MatrixProperties::GENERAL,
    );
    for (_, matrix) in terms {
        for contribution in matrix.contributions.iter() {
            if !assembler.contributions.contains(contribution) {
                assembler.add_contribution(contribution.clone());
            }
        }
    }
    for (scale, matrix) in terms {
        for row in 0..matrix.rows() {
            for entry in matrix.pattern.row_range(row) {
                assembler.add(
                    row,
                    matrix.pattern.column(entry),
                    *scale * matrix.values[entry],
                );
            }
        }
    }
    assembler.finish().unwrap()
}

fn pseudo_random_matrix(n: usize, seed: u64, id: &str) -> RealSparseMatrix {
    let mut state = seed;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut assembler =
        RealSparseAssembler::new(n, n, OperatorRole::Stiffness, MatrixProperties::GENERAL);
    assembler.add_contribution(ContributionProvenance {
        domain: ContributionDomain::Acoustic,
        contribution_id: ContributionId(id.into()),
    });
    for row in 0..n {
        for _ in 0..(next() % 7) {
            let column = (next() % n as u64) as usize;
            // Small integers make exact cancellation between terms reachable.
            let value = (next() % 5) as f64 - 2.0;
            assembler.add(row, column, value);
        }
    }
    assembler.finish().unwrap()
}

/// Scaled operator sums are bitwise the discovered-assembler sum (values, support, exact-zero drop) for one and several admitted threads
#[test]
fn scaled_sum_matches_term_by_term_accumulation_for_every_thread_count() {
    let n = 5000;
    let a = pseudo_random_matrix(n, 0x9e37_79b9_7f4a_7c15, "a");
    let b = pseudo_random_matrix(n, 0x2545_f491_4f6c_dd1d, "b");
    let c = pseudo_random_matrix(n, 0x1234_5678_9abc_def1, "a");
    let terms = [(1.0, &a), (-1.0, &b), (0.37, &c), (0.0, &b)];
    let expected = scaled_sum_reference(&terms);
    let serial =
        RealSparseMatrix::scaled_sum(&terms, OperatorRole::Stiffness, MatrixProperties::GENERAL)
            .unwrap();
    assert_eq!(serial.pattern, expected.pattern);
    assert_eq!(
        serial
            .values
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        expected
            .values
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(*serial.contributions, *expected.contributions);
    assert!(serial.values.iter().all(|value| *value != 0.0));
    let threaded = runmat_cpu_execution::with_cpu_pool(4, || {
        RealSparseMatrix::scaled_sum(&terms, OperatorRole::Stiffness, MatrixProperties::GENERAL)
            .unwrap()
    })
    .unwrap();
    assert_eq!(threaded.pattern, serial.pattern);
    assert_eq!(
        threaded
            .values
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        serial
            .values
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
    let mismatched = pseudo_random_matrix(7, 1, "d");
    assert!(matches!(
        RealSparseMatrix::scaled_sum(
            &[(1.0, &a), (1.0, &mismatched)],
            OperatorRole::Stiffness,
            MatrixProperties::GENERAL
        ),
        Err(RealSparseMatrixError::ShapeMismatch)
    ));
    assert!(matches!(
        RealSparseMatrix::scaled_sum(
            &[(f64::INFINITY, &a)],
            OperatorRole::Stiffness,
            MatrixProperties::GENERAL
        ),
        Err(RealSparseMatrixError::NonFiniteValue { .. })
    ));
}

fn row_matrix(columns: usize) -> RealSparseMatrix {
    RealSparseMatrix::try_new(
        SparsePattern::try_new(
            1,
            columns as u64,
            vec![0, columns as u64],
            (0..columns as u64).collect(),
        )
        .unwrap(),
        (0..columns).map(|i| i as f64 * 0.5).collect(),
        OperatorRole::Stiffness,
        MatrixProperties::GENERAL,
        Vec::new(),
    )
    .unwrap()
}

#[test]
fn streamed_values_identity_is_the_digest_of_the_canonical_encoding() {
    let matrix = row_matrix(1000);
    let encoded = runmat_canonical_codec::encode_contract(
        VALUES_IDENTITY_PREFIX,
        VALUES_IDENTITY_DOMAIN,
        &(
            matrix.pattern.identity().unwrap(),
            &matrix.values,
            matrix.scalar_domain,
            matrix.precision,
            matrix.role,
            matrix.properties,
            &matrix.contributions,
        ),
        VALUES_IDENTITY_LIMITS,
    )
    .unwrap();
    assert_eq!(matrix.values_identity().unwrap(), Digest::sha256(encoded));
}

/// About 1.1 GB resident: run explicitly (`--ignored`). A values contract whose encoding
/// exceeds the former fixed 512 MiB cap has pattern, values and numerical identities.
#[test]
#[ignore]
fn sparse_values_identity_over_512_mib_is_admitted() {
    let columns = (512 * 1024 * 1024) / 8 + 1;
    let matrix = row_matrix(columns);
    assert!(matrix.values.len() * 8 > 512 * 1024 * 1024);
    let values = matrix.values_identity().unwrap();
    let numerical = matrix.numerical_identity().unwrap();
    assert_ne!(values, numerical);
    assert_ne!(matrix.pattern_identity().unwrap(), values);
}
