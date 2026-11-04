//! Local geometric proposals follow the same objective as final admission.
//! Frozen per-cell metrics are only a proposal model: grading and complete
//! affected-cell quality are rederived by the trial owner before acceptance.
use super::*;
use crate::cdt::volume_quality::{
    evaluate_tetrahedron_geometry, violation_priority_key, violation_ratios,
};

#[derive(Clone, Copy)]
pub(in crate::cdt::volume_sliver) enum SmoothingObjective {
    Shape,
    Coupled(crate::cdt::DelaunayVolumeQualityOptions),
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct StarScore(Vec<u64>);

impl SmoothingObjective {
    pub(in crate::cdt::volume_sliver) fn for_policy(
        policy: SliverSearchPolicy,
        options: crate::cdt::DelaunayVolumeQualityOptions,
    ) -> Self {
        match policy.refinement {
            super::super::super::admission::CandidateRefinementPolicy::Preserve => Self::Shape,
            super::super::super::admission::CandidateRefinementPolicy::AllowForCoupledConvergence => {
                Self::Coupled(options)
            }
        }
    }

    pub(super) fn prepare(
        self,
        topology: &(impl crate::cdt::topology::VolumeTopologyRead + ?Sized),
        star: &[u32],
        source_index: usize,
        metrics: &dyn Fn(u32) -> runmat_meshing_size::metric::MetricTensor3,
    ) -> PreparedStar {
        let mut boundary = BTreeMap::new();
        let cells = star
            .iter()
            .map(|cell| {
                let vertices = topology.cell(*cell).vertex_indices;
                let boundary_count = vertices
                    .iter()
                    .filter(|vertex| {
                        *boundary
                            .entry(**vertex)
                            .or_insert_with(|| topology.vertex_is_boundary(**vertex))
                    })
                    .count() as u8;
                StarCell {
                    points: vertices
                        .map(|vertex| topology.node(vertex).coordinates_m.canonical_value()),
                    source_slot: vertices
                        .iter()
                        .position(|vertex| *vertex as usize == source_index)
                        .expect("incident cell contains source"),
                    metric: metrics(*cell),
                    boundary_count,
                }
            })
            .collect();
        PreparedStar {
            objective: self,
            cells,
        }
    }
}

struct StarCell {
    points: [[f64; 3]; 4],
    source_slot: usize,
    metric: runmat_meshing_size::metric::MetricTensor3,
    boundary_count: u8,
}

/// Coordinates, incidence and metric identities cannot change within a fixed-star
/// proposal. Capture them once; no topology query belongs in a numerical probe.
pub(super) struct PreparedStar {
    objective: SmoothingObjective,
    cells: Vec<StarCell>,
}
impl PreparedStar {
    pub(super) fn score(&self, point: [f64; 3]) -> Result<StarScore, DelaunayVolumeSliverError> {
        let mut spectrum = Vec::with_capacity(2 * self.cells.len());
        let mut minimum_jacobian = f64::INFINITY;
        for cell in &self.cells {
            let mut points = cell.points;
            points[cell.source_slot] = point;
            let Self {
                objective: SmoothingObjective::Coupled(options),
                ..
            } = self
            else {
                let Some(jacobian) =
                    crate::cdt::volume_quality::evaluate_tetrahedron_jacobian(points, cell.metric)
                        .map_err(|cause| {
                            error(
                                DelaunayVolumeSliverErrorKind::Accounting(cause),
                                "star score numerical accounting failed",
                            )
                        })?
                else {
                    return Ok(StarScore(vec![u64::MAX]));
                };
                minimum_jacobian = minimum_jacobian.min(jacobian);
                continue;
            };
            let Some(geometry) =
                evaluate_tetrahedron_geometry(points, cell.metric).map_err(|cause| {
                    error(
                        DelaunayVolumeSliverErrorKind::Accounting(cause),
                        "star geometry numerical accounting failed",
                    )
                })?
            else {
                return Ok(StarScore(vec![u64::MAX]));
            };
            let (size, shape) = violation_ratios(
                geometry.maximum_metric_edge_length,
                geometry.metric_radius_edge_ratio,
                geometry.metric_scaled_jacobian,
                cell.boundary_count,
                *options,
            );
            spectrum.extend(
                [size, shape]
                    .into_iter()
                    .filter(|value| *value > 1.)
                    .map(violation_priority_key),
            );
        }
        if matches!(self.objective, SmoothingObjective::Shape) {
            return Ok(StarScore(vec![u64::MAX - minimum_jacobian.to_bits()]));
        }
        spectrum.sort_unstable_by(|left, right| right.cmp(left));
        Ok(StarScore(spectrum))
    }
}

#[cfg(test)]
mod tests;
