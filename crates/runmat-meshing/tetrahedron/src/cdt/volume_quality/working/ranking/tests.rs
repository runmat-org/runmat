use super::*;
fn counts(values: &[u64], baseline: &[u64]) -> BTreeMap<u64, i64> {
    let mut map = BTreeMap::new();
    for (items, sign) in [(values, 1_i64), (baseline, -1_i64)] {
        for &value in items {
            *map.entry(value).or_default() += sign;
        }
    }
    map.retain(|_, count| *count != 0);
    map
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: A complete-spectrum improvement confined below the selected defect class cannot
//   discharge a repair pass. Reducing that class's multiplicity or priority is admitted; creating a
//   higher defect is rejected. Shared sparse delta comparison needs no copied full spectrum.
//   Physical acceptance still checks every unquantized bound independently.
#[test]
fn selected_class_progress_rejects_tail_only_stagnation_and_accepts_class_reduction() {
    let empty = BTreeMap::new();
    let tail = counts(&[90, 40], &[90, 50]);
    assert_eq!(
        compare_joint_changes(&tail, &empty, &empty, &empty),
        Ordering::Less
    );
    assert_eq!(
        compare_joint_changes_above(&tail, &empty, &empty, &empty, 90),
        Ordering::Equal
    );
    for after in [vec![89, 80], vec![90, 80]] {
        // Two source-class cells become either zero or one source-class cell.
        let changes = counts(&after, &[90, 90]);
        assert_eq!(
            compare_joint_changes_above(&changes, &empty, &empty, &empty, 90),
            Ordering::Less
        );
    }
    let regression = counts(&[91, 1], &[90, 50]);
    assert_eq!(
        compare_joint_changes_above(&regression, &empty, &empty, &empty, 90),
        Ordering::Greater
    );
}
// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: A four-stream sparse-delta merge matches independently sorted complete size/shape
//   spectra and rejects cross-phase undo. The coupled scheduler requires shared strict progress;
//   standalone subdivision retains its separate shape-preservation policy. This ordering proof is
//   not a guarantee that an arbitrary protected boundary admits the requested mesh quality.
#[test]
fn joint_delta_order_matches_complete_spectra_and_rejects_cross_phase_undo() {
    for case in 0..128_u64 {
        let values = |seed| {
            (0..17_u64)
                .map(|index| (index * (case + seed) + seed * 13) % 101)
                .collect::<Vec<_>>()
        };
        let (left_shape, left_refinement, right_shape, right_refinement) =
            (values(1), values(3), values(5), values(7));
        let baseline = values(11);
        let mut left = left_shape
            .iter()
            .chain(&left_refinement)
            .copied()
            .collect::<Vec<_>>();
        let mut right = right_shape
            .iter()
            .chain(&right_refinement)
            .copied()
            .collect::<Vec<_>>();
        left.sort_unstable_by(|a, b| b.cmp(a));
        right.sort_unstable_by(|a, b| b.cmp(a));
        assert_eq!(
            compare_joint_changes(
                &counts(&left_shape, &baseline),
                &counts(&left_refinement, &baseline),
                &counts(&right_shape, &baseline),
                &counts(&right_refinement, &baseline)
            ),
            left.cmp(&right)
        );
    }
    let empty = BTreeMap::new();
    // Shape repair lowers 90 to 80, while transferring the remaining defect to size.
    assert_eq!(
        compare_joint_changes(&counts(&[], &[90]), &counts(&[80], &[]), &empty, &empty),
        Ordering::Less
    );
    // Size repair may not undo that gain by creating a shape defect at 85.
    assert_eq!(
        compare_joint_changes(&counts(&[85], &[]), &counts(&[], &[80]), &empty, &empty),
        Ordering::Greater
    );
}
