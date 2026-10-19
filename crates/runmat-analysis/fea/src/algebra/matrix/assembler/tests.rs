use super::*;

#[test]
fn planned_scatter_matches_discovered_and_compacts_without_new_buffers() {
    let pattern = SparsePattern::try_new(
        4,
        5,
        vec![0, 5, 10, 15, 20],
        (0..4).flat_map(|_| 0..5).collect(),
    )
    .unwrap();
    let mut planned = RealSparseAssembler::with_pattern(
        pattern,
        OperatorRole::Conductivity,
        MatrixProperties::GENERAL,
    )
    .unwrap();
    let mut discovered =
        RealSparseAssembler::new(4, 5, OperatorRole::Conductivity, MatrixProperties::GENERAL);
    let (offsets, columns, values) = match &planned.storage {
        AssemblyStorage::Planned { pattern, values } => (
            pattern.row_offsets().as_ptr(),
            pattern.column_indices().as_ptr(),
            values.as_ptr(),
        ),
        _ => unreachable!(),
    };
    for assembler in [&mut planned, &mut discovered] {
        assembler.add_contribution(ContributionProvenance {
            domain: ContributionDomain::Thermal,
            contribution_id: ContributionId("planned".into()),
        });
        // Empty/cancelled rows separate surviving entries; insertion order is deliberately reversed.
        for row in (0..4).rev() {
            for column in (0..5).rev() {
                let value = (row * 5 + column + 1) as f64;
                assembler.add(row, column, value);
                if row == 0 || row == 2 || column % 2 == 0 {
                    assembler.add(row, column, -value);
                }
            }
        }
    }
    let actual = planned.finish().unwrap();
    let expected = discovered.finish().unwrap();
    assert_eq!(actual, expected);
    assert_eq!(
        actual.values_identity().unwrap(),
        expected.values_identity().unwrap()
    );
    assert_eq!(actual.pattern().row_offsets().as_ptr(), offsets);
    assert_eq!(actual.pattern().column_indices().as_ptr(), columns);
    assert_eq!(actual.values().as_ptr(), values);
}

#[test]
fn planned_support_fails_closed_and_handles_empty_patterns() {
    let diagonal = || SparsePattern::try_new(2, 2, vec![0, 1, 2], vec![0, 1]).unwrap();
    for (row, column, value, error) in [
        (
            0,
            1,
            1.0,
            RealSparseMatrixError::OutsidePlannedPattern { row: 0, column: 1 },
        ),
        (2, 0, 1.0, RealSparseMatrixError::ShapeMismatch),
        (
            0,
            0,
            f64::NAN,
            RealSparseMatrixError::NonFiniteValue { entry: 0 },
        ),
    ] {
        let mut assembler = RealSparseAssembler::with_pattern(
            diagonal(),
            OperatorRole::Mass,
            MatrixProperties::GENERAL,
        )
        .unwrap();
        assembler.add(row, column, value);
        assert_eq!(assembler.finish(), Err(error));
    }
    let mut overflow = RealSparseAssembler::with_pattern(
        diagonal(),
        OperatorRole::Mass,
        MatrixProperties::GENERAL,
    )
    .unwrap();
    overflow.add(0, 0, f64::MAX);
    overflow.add(0, 0, f64::MAX);
    assert_eq!(
        overflow.finish(),
        Err(RealSparseMatrixError::NonFiniteValue { entry: 0 })
    );
    let empty = RealSparseAssembler::with_pattern(
        SparsePattern::try_new(0, 0, vec![0], vec![]).unwrap(),
        OperatorRole::Mass,
        MatrixProperties::GENERAL,
    )
    .unwrap()
    .finish()
    .unwrap();
    assert_eq!(empty.pattern().row_offsets(), &[0]);
    assert!(empty.values().is_empty());
}

#[test]
fn shared_planned_support_survives_unchanged_and_compacted_assemblies() {
    let source = SparsePattern::try_new(2, 2, vec![0, 2, 4], vec![0, 1, 0, 1]).unwrap();
    let mut full = RealSparseAssembler::with_pattern(
        source.clone(),
        OperatorRole::Conductivity,
        MatrixProperties::GENERAL,
    )
    .unwrap();
    let mut compacted = RealSparseAssembler::with_pattern(
        source.clone(),
        OperatorRole::Conductivity,
        MatrixProperties::GENERAL,
    )
    .unwrap();
    for row in 0..2 {
        for column in 0..2 {
            full.add(row, column, (row + column + 1) as f64);
            if row != column {
                compacted.add(row, column, 1.0);
            }
        }
    }
    let full = full.finish().unwrap();
    assert_eq!(
        full.pattern().column_indices().as_ptr(),
        source.column_indices().as_ptr()
    );
    let compacted = compacted.finish().unwrap();
    assert_eq!(compacted.pattern().row_offsets(), &[0, 1, 2]);
    assert_eq!(compacted.pattern().column_indices(), &[1, 0]);
    assert_eq!(source.row_offsets(), &[0, 2, 4]);
    assert_eq!(source.column_indices(), &[0, 1, 0, 1]);
    assert_eq!(full.pattern(), &source);
}
