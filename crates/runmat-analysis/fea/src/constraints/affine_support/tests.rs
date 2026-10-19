use super::*;
fn reduction() -> AffineConstraintReduction {
    AffineConstraintReduction {
        full_dof_count: 4,
        reduced_dof_count: 3,
        free_dof_indices: vec![0, 1, 2],
        pivot_dof_indices: vec![3],
        recovery_row_offsets: vec![0, 2, 3, 4, 6],
        recovery_column_indices: vec![2, 0, 1, 2, 0, 1],
        recovery_coefficients: vec![1.; 6],
        offset: vec![0.; 4],
    }
}
fn pattern(dense: &[[bool; 4]; 4]) -> SparsePattern {
    let mut offsets = vec![0];
    let mut columns = Vec::new();
    for row in dense {
        for (j, present) in row.iter().enumerate() {
            if *present {
                columns.push(j as u64);
            }
        }
        offsets.push(columns.len() as u64);
    }
    SparsePattern::try_new(4, 4, offsets, columns).unwrap()
}
fn budget() -> AffineSupportBudget {
    AffineSupportBudget {
        maximum_resident_bytes: 100_000,
        maximum_scalar_operations: 100_000,
    }
}
#[test]
fn exact_support_matches_independent_boolean_congruence_for_every_source_entry() {
    let t = [
        [true, false, true],
        [false, true, false],
        [false, false, true],
        [true, true, false],
    ];
    let r = reduction();
    let original = r.clone();
    for entry in 0..16 {
        let mut a = [[false; 4]; 4];
        a[entry / 4][entry % 4] = true;
        let mut b = [[false; 4]; 4];
        b[(entry + 5) % 4][(entry + 7) % 4] = true;
        let a = pattern(&a);
        let b = pattern(&b);
        let plan = AffineSparseSupportPlan::prepare(&r, &a, &b, budget()).unwrap();
        let statistics = plan.statistics();
        let actual = plan.materialize().unwrap();
        let mut count = 0;
        let mut maximum = 0;
        for i in 0..3 {
            let mut width = 0;
            for j in 0..3 {
                let expected = (0..4).any(|k| {
                    (0..4).any(|l| {
                        t[k][i]
                            && t[l][j]
                            && (a.column_indices()
                                [a.row_offsets()[k] as usize..a.row_offsets()[k + 1] as usize]
                                .contains(&(l as u64))
                                || b.column_indices()
                                    [b.row_offsets()[k] as usize..b.row_offsets()[k + 1] as usize]
                                    .contains(&(l as u64)))
                    })
                });
                let found = actual.column_indices()
                    [actual.row_offsets()[i] as usize..actual.row_offsets()[i + 1] as usize]
                    .contains(&(j as u64));
                assert_eq!(found, expected, "entry {entry}, ({i},{j})");
                width += usize::from(expected);
            }
            count += width;
            maximum = maximum.max(width);
        }
        assert_eq!(statistics.nonzeros, count);
        assert_eq!(statistics.maximum_row_nonzeros, maximum);
        assert_eq!(statistics.resident_bytes().unwrap(), 8 * (4 + count) as u64);
    }
    assert_eq!(r, original);
}
#[test]
fn exact_admission_boundary_and_malformed_offsets_reject_before_materialization() {
    let r = reduction();
    let a = pattern(&[[true; 4]; 4]);
    let plan = AffineSparseSupportPlan::prepare(&r, &a, &a, budget()).unwrap();
    let exact = AffineSupportBudget {
        maximum_resident_bytes: plan.planning_resident_bytes(),
        maximum_scalar_operations: plan.scalar_operations_upper_bound(),
    };
    assert!(AffineSparseSupportPlan::prepare(&r, &a, &a, exact).is_ok());
    for resident in [true, false] {
        let mut short = exact;
        if resident {
            short.maximum_resident_bytes -= 1;
        } else {
            short.maximum_scalar_operations -= 1;
        }
        assert!(matches!(
            AffineSparseSupportPlan::prepare(&r, &a, &a, short),
            Err(AffineSupportError::Budget)
        ));
    }
    for offsets in [vec![0, 7, 3, 4, 6], vec![0, 2, 1, 4, 6]] {
        let mut bad = r.clone();
        bad.recovery_row_offsets = offsets;
        assert!(matches!(
            AffineSparseSupportPlan::prepare(&bad, &a, &a, budget()),
            Err(AffineSupportError::Constraint(_))
        ));
    }
}
#[test]
fn all_fixed_duplicate_entries_and_empty_support_are_canonical() {
    let mut r = reduction();
    r.recovery_column_indices[0] = 0; // Duplicate support must deduplicate.
    let a = pattern(&[[true; 4]; 4]);
    let plan = AffineSparseSupportPlan::prepare(&r, &a, &a, budget()).unwrap();
    assert_eq!(plan.statistics().nonzeros, 9);
    assert_eq!(
        plan.materialize().unwrap().column_indices(),
        &[0, 1, 2, 0, 1, 2, 0, 1, 2]
    );
    r.reduced_dof_count = 0;
    r.free_dof_indices.clear();
    r.pivot_dof_indices = vec![0, 1, 2, 3];
    r.recovery_row_offsets = vec![0; 5];
    r.recovery_column_indices.clear();
    r.recovery_coefficients.clear();
    let actual = AffineSparseSupportPlan::prepare(&r, &a, &a, budget())
        .unwrap()
        .materialize()
        .unwrap();
    assert_eq!(actual.rows(), 0);
    assert_eq!(actual.row_offsets(), &[0]);
}
#[test]
fn cancellation_rejects_both_planning_and_admitted_materialization() {
    let r = reduction();
    let a = pattern(&[[true; 4]; 4]);
    let plan = AffineSparseSupportPlan::prepare(&r, &a, &a, budget()).unwrap();
    let token = crate::progress::FeaCancellationToken::new();
    token.cancel();
    let _guard =
        crate::progress::replace_fea_progress_context(crate::progress::FeaProgressContext {
            handler: None,
            cancellation: token,
            attempt_id: None,
        });
    assert!(matches!(
        AffineSparseSupportPlan::prepare(&r, &a, &a, budget()),
        Err(AffineSupportError::Cancelled)
    ));
    assert!(matches!(
        plan.materialize(),
        Err(AffineSupportError::Cancelled)
    ));
}
