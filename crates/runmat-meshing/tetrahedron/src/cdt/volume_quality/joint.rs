//! Canonical reference comparison for joint size and shape progress. Borrowed
//! cell inventories merge without copying background cells into search seeds.
use super::*;
use std::collections::BTreeMap;

pub(in crate::cdt) fn compare(
    before: &DelaunayVolumeQuality,
    after: &DelaunayVolumeQuality,
) -> Ordering {
    compare_entries(before.tetrahedra.iter(), after.tetrahedra.iter())
}

pub(in crate::cdt) fn compare_entries<'a, 'b>(
    before: impl Iterator<Item = &'a DelaunayTetrahedronQuality>,
    after: impl Iterator<Item = &'b DelaunayTetrahedronQuality>,
) -> Ordering {
    let mut changes = BTreeMap::<u64, i64>::new();
    let mut record = |cell: &DelaunayTetrahedronQuality, sign: i64| {
        for ratio in [
            cell.refinement_violation_ratio,
            cell.optimization_violation_ratio,
        ] {
            if !violates_quality_bound(ratio) {
                continue;
            }
            let bin = violation_priority_key(ratio);
            let count = changes.entry(bin).or_default();
            *count += sign;
            if *count == 0 {
                changes.remove(&bin);
            }
        }
    };
    let key = |cell: &DelaunayTetrahedronQuality| {
        let mut key = cell.node_identities;
        key.sort_unstable();
        key
    };
    let mut left = before.peekable();
    let mut right = after.peekable();
    loop {
        match (left.peek(), right.peek()) {
            (Some(a), Some(b)) if key(a) == key(b) => {
                if a.refinement_violation_ratio != b.refinement_violation_ratio
                    || a.optimization_violation_ratio != b.optimization_violation_ratio
                {
                    record(a, -1);
                    record(b, 1);
                }
                left.next();
                right.next();
            }
            (Some(a), Some(b)) if key(a) < key(b) => {
                record(a, -1);
                left.next();
            }
            (Some(_), Some(b)) => {
                record(b, 1);
                right.next();
            }
            (Some(a), None) => {
                record(a, -1);
                left.next();
            }
            (None, Some(b)) => {
                record(b, 1);
                right.next();
            }
            (None, None) => break,
        }
    }
    changes
        .last_key_value()
        .map_or(Ordering::Equal, |(_, count)| count.cmp(&0))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn canonical_join_rejects_a_size_shape_cycle_across_unrelated_inventory() {
        use crate::cdt::test_fixtures::coupled_quality::{fixture, metric};
        for background in [0, 64] {
            let (topology, provenance) = fixture(false, background);
            let request = metric(&provenance, false);
            let mut before = evaluate_delaunay_volume_quality(
                &topology,
                &request,
                &provenance,
                DelaunayVolumeQualityOptions::default(),
                &runmat_meshing_core::NeverCancelled,
            )
            .unwrap();
            // Algebraic progress-policy witness, not a replacement physical-quality result.
            for cell in &mut before.tetrahedra {
                cell.refinement_violation_ratio = 0.5;
                cell.optimization_violation_ratio = 0.5;
            }
            before.tetrahedra[0].optimization_violation_ratio = 9.;
            let mut transfer = before.clone();
            transfer.tetrahedra[0].optimization_violation_ratio = 0.5;
            transfer.tetrahedra[0].refinement_violation_ratio = 8.;
            assert_eq!(compare(&before, &transfer), Ordering::Less);
            let mut undo = transfer.clone();
            undo.tetrahedra[0].refinement_violation_ratio = 0.5;
            undo.tetrahedra[0].optimization_violation_ratio = 8.5;
            assert_eq!(compare(&transfer, &undo), Ordering::Greater);
            assert_eq!(compare(&before, &before), Ordering::Equal);
        }
    }
}
