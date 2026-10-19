//! Exact quality differences against the same immutable beam-search baseline.
//! Unchanged per-cell metadata is shared logically, never cloned into each seed.
use super::*;
use crate::cdt::volume_sliver::{admission, SliverSearchPolicy};

pub(in crate::cdt::volume_sliver) struct QualityPatch {
    removed: Vec<[StableDigest; 4]>,
    changed: Vec<DelaunayTetrahedronQuality>,
    summary: DelaunayVolumeQuality,
}

impl QualityPatch {
    pub(in crate::cdt::volume_sliver) fn between(
        base: &DelaunayVolumeQuality,
        candidate: &DelaunayVolumeQuality,
    ) -> Result<Self, DelaunayVolumeSliverError> {
        for quality in [base, candidate] {
            if quality
                .tetrahedra
                .windows(2)
                .any(|pair| key(&pair[0]) >= key(&pair[1]))
            {
                return Err(error(
                    DelaunayVolumeSliverErrorKind::InvalidQuality,
                    "compact quality seed requires canonical unique cell order",
                ));
            }
        }
        let mut baseline = base.tetrahedra.iter().peekable();
        let mut removed = Vec::new();
        let mut changed = Vec::new();
        for cell in &candidate.tetrahedra {
            while baseline.peek().is_some_and(|old| key(old) < key(cell)) {
                removed.push(key(baseline.next().unwrap()));
            }
            match baseline.peek() {
                Some(old) if key(old) == key(cell) => {
                    if *old != cell {
                        removed.push(key(old));
                        changed.push(cell.clone());
                    }
                    baseline.next();
                }
                _ => changed.push(cell.clone()),
            }
        }
        removed.extend(baseline.map(key));
        Ok(Self {
            removed,
            changed,
            summary: summary(candidate),
        })
    }

    fn entries<'a>(
        &'a self,
        base: &'a DelaunayVolumeQuality,
    ) -> impl Iterator<Item = &'a DelaunayTetrahedronQuality> + 'a {
        let mut retained = base
            .tetrahedra
            .iter()
            .filter(|cell| self.removed.binary_search(&key(cell)).is_err())
            .peekable();
        let mut changed = self.changed.iter().peekable();
        std::iter::from_fn(move || match (retained.peek(), changed.peek()) {
            (Some(left), Some(right)) if key(left) <= key(right) => retained.next(),
            (Some(_), Some(_)) => changed.next(),
            (Some(_), None) => retained.next(),
            (None, Some(_)) => changed.next(),
            (None, None) => None,
        })
    }

    pub(in crate::cdt::volume_sliver) fn spectrum(
        &self,
        base: &DelaunayVolumeQuality,
        objective: crate::cdt::volume_sliver::admission::QualityObjective,
    ) -> Vec<f64> {
        crate::cdt::volume_sliver::admission::quality_spectrum_from_entries(
            self.entries(base),
            objective,
        )
    }

    pub(in crate::cdt::volume_sliver) fn compare_policy(
        &self,
        other: &Self,
        base: &DelaunayVolumeQuality,
        policy: SliverSearchPolicy,
    ) -> std::cmp::Ordering {
        let primary = || {
            admission::compare_spectra(
                &self.spectrum(base, policy.objective),
                &other.spectrum(base, policy.objective),
            )
        };
        if matches!(
            policy.refinement,
            admission::CandidateRefinementPolicy::AllowForCoupledConvergence
        ) {
            crate::cdt::volume_quality::joint::compare_entries(
                self.entries(base),
                other.entries(base),
            )
            .then_with(primary)
        } else {
            primary()
        }
    }

    pub(in crate::cdt::volume_sliver) fn compare_candidate(
        &self,
        candidate: &DelaunayVolumeQuality,
        base: &DelaunayVolumeQuality,
        policy: SliverSearchPolicy,
    ) -> std::cmp::Ordering {
        let primary = || {
            admission::compare_spectra(
                &admission::quality_spectrum(candidate, policy.objective),
                &self.spectrum(base, policy.objective),
            )
        };
        if matches!(
            policy.refinement,
            admission::CandidateRefinementPolicy::AllowForCoupledConvergence
        ) {
            crate::cdt::volume_quality::joint::compare_entries(
                candidate.tetrahedra.iter(),
                self.entries(base),
            )
            .then_with(primary)
        } else {
            primary()
        }
    }

    pub(in crate::cdt::volume_sliver) fn materialize(
        &self,
        base: &DelaunayVolumeQuality,
    ) -> DelaunayVolumeQuality {
        let mut result = summary(&self.summary);
        result.tetrahedra = self.entries(base).cloned().collect();
        result
    }
}

fn key(cell: &DelaunayTetrahedronQuality) -> [StableDigest; 4] {
    let mut key = cell.node_identities;
    key.sort_unstable();
    key
}

fn summary(quality: &DelaunayVolumeQuality) -> DelaunayVolumeQuality {
    DelaunayVolumeQuality {
        tetrahedra: Vec::new(),
        worst_refinement_tetrahedron: quality.worst_refinement_tetrahedron,
        worst_optimization_tetrahedron: quality.worst_optimization_tetrahedron,
        maximum_metric_edge_length: quality.maximum_metric_edge_length,
        maximum_radius_edge_ratio: quality.maximum_radius_edge_ratio,
        minimum_metric_scaled_jacobian: quality.minimum_metric_scaled_jacobian,
    }
}

#[cfg(test)]
#[path = "quality/tests.rs"]
mod tests;
