//! Bounded highest-priority frontier. Keep only one batch's source references;
//! do not sort or retain the complete violating-cell inventory every checkpoint.
use super::*;
use crate::cdt::{
    volume_quality::{compare_violation_priority, tetrahedron_geometry_key},
    DelaunayTetrahedronQuality,
};
use std::cmp::Ordering;
use std::{cmp::Reverse, collections::BinaryHeap};

#[derive(Clone, Copy)]
struct SourcePriority {
    violation: f64,
    geometry: [[u64; 3]; 4],
    identities: [StableDigest; 4],
}
impl SourcePriority {
    fn new(input: DelaunayVolumeRefinementInput<'_>, quality: &DelaunayTetrahedronQuality) -> Self {
        Self {
            violation: quality.refinement_violation_ratio,
            geometry: tetrahedron_geometry_key(input.topology, quality.node_identities),
            identities: quality.node_identities,
        }
    }
    fn compare(self, other: Self) -> Ordering {
        compare_violation_priority(
            self.violation,
            self.geometry,
            self.identities,
            other.violation,
            other.geometry,
            other.identities,
        )
    }
}

/// Scheduling position only, never evidence that a skipped cavity is invalid.
/// Each bounded window continues below the last examined source. Wrapping makes
/// changed or formerly blocked sources eligible again without retaining a full
/// mesh-sized queue. Admission still checks the current cavity and full quality.
#[derive(Clone, Copy, Default)]
pub(in crate::cdt::volume_refinement) struct Frontier {
    after: Option<SourcePriority>,
    more: bool,
}
impl Frontier {
    pub(in crate::cdt::volume_refinement) fn advance(
        &mut self,
        input: DelaunayVolumeRefinementInput<'_>,
        source: &DelaunayTetrahedronQuality,
    ) {
        self.after = Some(SourcePriority::new(input, source));
    }
    pub(in crate::cdt::volume_refinement) fn has_more(&self) -> bool {
        self.more
    }
}

struct Source<'a> {
    quality: &'a DelaunayTetrahedronQuality,
    priority: SourcePriority,
}
impl PartialEq for Source<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Source<'_> {}
impl PartialOrd for Source<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Source<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority.compare(other.priority)
    }
}

#[cfg(test)]
pub(super) fn select<'a>(
    input: DelaunayVolumeRefinementInput<'a>,
    maximum: usize,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Vec<&'a DelaunayTetrahedronQuality>, DelaunayVolumeRefinementStepError> {
    select_window(input, maximum, &mut Frontier::default(), cancellation)
}

pub(in crate::cdt::volume_refinement) fn select_window<'a>(
    input: DelaunayVolumeRefinementInput<'a>,
    maximum: usize,
    cursor: &mut Frontier,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Vec<&'a DelaunayTetrahedronQuality>, DelaunayVolumeRefinementStepError> {
    let mut frontier: BinaryHeap<Reverse<Source<'a>>> = BinaryHeap::new();
    let mut eligible = 0usize;
    for (index, quality) in input.quality.tetrahedra.iter().enumerate() {
        if index.is_multiple_of(1024) && cancellation.is_cancelled() {
            return Err(error(
                DelaunayVolumeRefinementStepErrorKind::Cancelled,
                "refinement frontier preparation was cancelled",
            ));
        }
        if maximum == 0 || !quality.requires_refinement() {
            continue;
        }
        let source = Source {
            quality,
            priority: SourcePriority::new(input, quality),
        };
        if cursor
            .after
            .is_some_and(|after| source.priority.compare(after) != Ordering::Less)
        {
            continue;
        }
        eligible += 1;
        if frontier.len() < maximum {
            frontier.push(Reverse(source));
        } else if frontier.peek().is_some_and(|least| source > least.0) {
            frontier.pop();
            frontier.push(Reverse(source));
        }
    }
    if maximum > 0 && frontier.is_empty() && cursor.after.take().is_some() {
        return select_window(input, maximum, cursor, cancellation);
    }
    cursor.more = eligible > maximum;
    let mut sources = frontier
        .into_iter()
        .map(|Reverse(source)| source)
        .collect::<Vec<_>>();
    sources.sort_unstable_by(|left, right| right.cmp(left));
    Ok(sources.into_iter().map(|source| source.quality).collect())
}

#[cfg(test)]
mod tests;
