//! Exact lexicographic comparison of the existing quantized descending spectrum.
//! The largest bin with unequal multiplicity decides the same ordering, including
//! equal prefixes and exhaustion. No full spectrum is copied into a trial.
use super::*;
use std::cmp::Reverse;
pub(super) type Rank = (u64, Reverse<[[u64; 3]; 4]>, Reverse<CellKey>);

#[derive(Default)]
pub(super) struct Ranking {
    pub(super) priorities: BTreeMap<Rank, CellKey>,
    pub(super) delta: BTreeMap<u64, i64>,
}

#[derive(Clone)]
pub(in crate::cdt) struct SpectrumDelta {
    shape: BTreeMap<u64, i64>,
    refinement: BTreeMap<u64, i64>,
}

impl SpectrumDelta {
    pub(super) fn compare_owner(
        &self,
        owner: &WorkingVolumeQuality,
        objective: QualityObjective,
    ) -> Ordering {
        compare_changes(&owner.ranking(objective).delta, self.changes(objective))
    }
    pub(super) fn capture(owner: &WorkingVolumeQuality) -> Self {
        Self {
            shape: owner.shape.delta.clone(),
            refinement: owner.refinement.delta.clone(),
        }
    }
    pub(in crate::cdt) fn compare(&self, other: &Self, objective: QualityObjective) -> Ordering {
        compare_changes(self.changes(objective), other.changes(objective))
    }
    pub(in crate::cdt) fn compare_joint(&self, other: &Self) -> Ordering {
        compare_joint_changes(
            &self.shape,
            &self.refinement,
            &other.shape,
            &other.refinement,
        )
    }
    fn changes(&self, objective: QualityObjective) -> &BTreeMap<u64, i64> {
        match objective {
            QualityObjective::Shape => &self.shape,
            QualityObjective::Refinement => &self.refinement,
        }
    }
}

fn compare_changes(left: &BTreeMap<u64, i64>, right: &BTreeMap<u64, i64>) -> Ordering {
    let mut left = left.iter().rev().peekable();
    let mut right = right.iter().rev().peekable();
    loop {
        let difference = match (left.peek(), right.peek()) {
            (Some((a, _)), Some((b, _))) if a > b => left.next().unwrap().1.cmp(&0),
            (Some((a, _)), Some((b, _))) if a < b => 0.cmp(right.next().unwrap().1),
            (Some(_), Some(_)) => left.next().unwrap().1.cmp(right.next().unwrap().1),
            (Some(_), None) => left.next().unwrap().1.cmp(&0),
            (None, Some(_)) => 0.cmp(right.next().unwrap().1),
            (None, None) => return Ordering::Equal,
        };
        if difference != Ordering::Equal {
            return difference;
        }
    }
}

impl Ranking {
    pub(super) fn update(
        &mut self,
        entry: &Entry,
        key: CellKey,
        objective: QualityObjective,
        add: bool,
        track_delta: bool,
    ) {
        let ratio = match objective {
            QualityObjective::Shape => entry.quality.optimization_violation_ratio,
            QualityObjective::Refinement => entry.quality.refinement_violation_ratio,
        };
        let bin = violation_priority_key(ratio);
        if track_delta && violates_quality_bound(ratio) {
            let count = self.delta.entry(bin).or_default();
            *count += if add { 1 } else { -1 };
            if *count == 0 {
                self.delta.remove(&bin);
            }
        }
        if violates_quality_bound(ratio) {
            let rank = (
                bin,
                Reverse(entry.geometry),
                Reverse(entry.quality.node_identities),
            );
            if add {
                self.priorities.insert(rank, key);
            } else {
                self.priorities.remove(&rank);
            }
        }
    }

    pub(super) fn compare_baseline(&self) -> Ordering {
        self.delta
            .last_key_value()
            .map_or(Ordering::Equal, |(_, count)| count.cmp(&0))
    }
}

/// Compare the union of both normalized violation spectra without materializing
/// either spectrum or adding another whole-mesh ranking. Highest unequal bin
/// multiplicity decides the same order as sorting the two complete spectra.
pub(super) fn compare_joint_changes(
    left_shape: &BTreeMap<u64, i64>,
    left_refinement: &BTreeMap<u64, i64>,
    right_shape: &BTreeMap<u64, i64>,
    right_refinement: &BTreeMap<u64, i64>,
) -> Ordering {
    compare_joint_changes_above(
        left_shape,
        left_refinement,
        right_shape,
        right_refinement,
        0,
    )
}
fn compare_joint_changes_above(
    left_shape: &BTreeMap<u64, i64>,
    left_refinement: &BTreeMap<u64, i64>,
    right_shape: &BTreeMap<u64, i64>,
    right_refinement: &BTreeMap<u64, i64>,
    floor: u64,
) -> Ordering {
    let mut streams = [
        left_shape.iter().rev().peekable(),
        left_refinement.iter().rev().peekable(),
        right_shape.iter().rev().peekable(),
        right_refinement.iter().rev().peekable(),
    ];
    loop {
        let Some(bin) = streams
            .iter_mut()
            .filter_map(|stream| stream.peek().map(|(bin, _)| **bin))
            .max()
        else {
            return Ordering::Equal;
        };
        if bin < floor {
            return Ordering::Equal;
        }
        let mut difference = 0_i128;
        for (index, stream) in streams.iter_mut().enumerate() {
            if stream.peek().is_some_and(|(key, _)| **key == bin) {
                let count = i128::from(*stream.next().unwrap().1);
                difference += if index < 2 { count } else { -count };
            }
        }
        if difference != 0 {
            return difference.cmp(&0);
        }
    }
}

impl WorkingVolumeQuality {
    /// Tail improvements do not discharge the defect class selected for repair.
    pub(in crate::cdt) fn compare_joint_repair_baseline(&self) -> Ordering {
        let Some(floor) = self.repair_floor else {
            return Ordering::Equal;
        };
        let empty = BTreeMap::new();
        compare_joint_changes_above(
            &self.shape.delta,
            &self.refinement.delta,
            &empty,
            &empty,
            floor,
        )
    }
    /// Joint (shape and size) defect histogram against `target`, compared from the highest
    /// violation bin down. `Less` means the defects this change removed outrank every defect it
    /// created: a strict descent in the multiset order over discrete violation bins, which is
    /// well founded, so a sequence of such changes terminates.
    pub(in crate::cdt) fn compare_joint_target(&self, target: &SpectrumDelta) -> Ordering {
        compare_joint_changes(
            &self.shape.delta,
            &self.refinement.delta,
            &target.shape,
            &target.refinement,
        )
    }
}

#[cfg(test)]
mod tests;
