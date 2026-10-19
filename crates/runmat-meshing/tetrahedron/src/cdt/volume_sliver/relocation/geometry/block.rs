//! Bounded local maximin smoothing uses all-corner derivatives and bounded search. Preparation
//! visits only the union of selected stars; probes never query the background.
use super::*;
use crate::cdt::volume_quality::PreparedJacobian;
use runmat_meshing_core::quality::predicate::{Point3Construction, PredicateSign};

struct Cell {
    nodes: [DelaunayVolumeNode; 4],
    slots: [Option<usize>; 4],
    metric: PreparedJacobian,
}
pub(in crate::cdt::volume_sliver::relocation) struct PreparedBlock {
    cells: Vec<Cell>,
    start: Vec<[f64; 3]>,
    scale: f64,
}
impl PreparedBlock {
    pub(in crate::cdt::volume_sliver::relocation) fn prepare(
        topology: &(impl VolumeTopologyRead + ?Sized),
        vertices: &[u32],
        cells: &BTreeSet<u32>,
        metrics: &dyn Fn(u32) -> runmat_meshing_size::metric::MetricTensor3,
    ) -> Self {
        let start = vertices
            .iter()
            .map(|vertex| topology.node(*vertex).coordinates_m)
            .collect();
        let mut scale = 0_f64;
        let cells = cells
            .iter()
            .map(|index| {
                let indices = topology.cell(*index).vertex_indices;
                let nodes = indices.map(|vertex| topology.node(vertex));
                for a in 0..4 {
                    for b in a + 1..4 {
                        let length = (0..3)
                            .map(|axis| nodes[a].coordinates_m[axis] - nodes[b].coordinates_m[axis])
                            .fold(0., f64::hypot);
                        if length > 0. {
                            scale = scale.max(length);
                        }
                    }
                }
                Cell {
                    nodes,
                    slots: indices.map(|vertex| vertices.iter().position(|v| *v == vertex)),
                    metric: PreparedJacobian::new(metrics(*index)).expect("admitted metric"),
                }
            })
            .collect();
        Self {
            cells,
            start,
            scale,
        }
    }
    fn score(&self, points: &[[f64; 3]]) -> Result<f64, DelaunayVolumeSliverError> {
        let mut minimum = f64::INFINITY;
        for cell in &self.cells {
            let nodes = std::array::from_fn(|slot| {
                let mut node = cell.nodes[slot];
                if let Some(index) = cell.slots[slot] {
                    node.coordinates_m = points[index];
                    node.construction = Point3Construction::Explicit;
                }
                node
            });
            if orient_volume_nodes(nodes).map_err(|failure| {
                error(
                    DelaunayVolumeSliverErrorKind::InvalidTopology,
                    format!("block proposal orientation failed: {failure:?}"),
                )
            })? != PredicateSign::Positive
            {
                return Ok(0.);
            }
            minimum = minimum.min(
                cell.metric
                    .score(nodes.map(|node| node.coordinates_m))
                    .unwrap_or(0.),
            );
        }
        Ok(minimum)
    }
    pub(in crate::cdt::volume_sliver::relocation) fn optimize(
        &self,
        cancellation: &dyn MeshingCancellationSignal,
    ) -> Result<Option<Vec<[f64; 3]>>, DelaunayVolumeSliverError> {
        if !self.scale.is_finite() || self.scale <= 0. {
            return Ok(None);
        }
        let dimensions = 3 * self.start.len();
        // Coordinate polling is linear in block dimension, unlike a Cartesian
        // stencil whose cost grows exponentially with neighboring point count.
        let directions = (0..dimensions)
            .flat_map(|axis| {
                [-1., 1.].map(move |sign| {
                    let mut direction = vec![0.; dimensions];
                    direction[axis] = sign;
                    direction
                })
            })
            .collect::<Vec<_>>();
        let mut points = self.start.clone();
        let initial = self.score(&points)?;
        let mut score = initial;
        let maximum_radius = self.scale * 0.25;
        let mut radius = maximum_radius;
        for iteration in 0..MAXIMUM_CAVITY_SMOOTHING_ITERATIONS {
            if iteration.is_multiple_of(8) && cancellation.is_cancelled() {
                return Err(error(
                    DelaunayVolumeSliverErrorKind::Cancelled,
                    "local cluster smoothing cancelled",
                ));
            }
            if let Some(direction) = self.ascent_direction(&points, radius) {
                let mut step = radius;
                let mut accepted = None;
                for _ in 0..24 {
                    let trial = displaced(&points, &direction, step);
                    let value = self.score(&trial)?;
                    if value > score {
                        accepted = Some((value, trial, step));
                        break;
                    }
                    step *= 0.5;
                }
                if let Some((value, trial, accepted_step)) = accepted {
                    score = value;
                    points = trial;
                    // A rejected large-radius model must not be proposed again
                    // after only a tiny fraction passed the nonlinear check.
                    // Successful full steps can grow back to the original cap.
                    radius = (2. * accepted_step).min(maximum_radius);
                    continue;
                }
            }
            let mut best = None;
            let results = runmat_cpu_execution::ordered_map(&directions, |direction| {
                let trial = displaced(&points, direction, radius);
                self.score(&trial).map(|value| (value, trial))
            });
            // Preserve canonical first-error and tie behavior across CPU grants.
            for result in results {
                let (value, trial) = result?;
                if value > score
                    && best
                        .as_ref()
                        .is_none_or(|(old, _): &(f64, Vec<[f64; 3]>)| value > *old)
                {
                    best = Some((value, trial));
                }
            }
            if let Some((value, trial)) = best {
                score = value;
                points = trial;
            } else {
                radius *= 0.5;
                if radius < self.scale * 2_f64.powi(-24) {
                    break;
                }
            }
        }
        tracing::debug!(
            point_count = points.len(),
            cell_count = self.cells.len(),
            initial_score = initial,
            proposed_score = score,
            "prepared cluster search finished"
        );
        Ok((score > initial).then_some(points))
    }
}

fn displaced(points: &[[f64; 3]], direction: &[f64], radius: f64) -> Vec<[f64; 3]> {
    points
        .iter()
        .enumerate()
        .map(|(point, p)| {
            std::array::from_fn(|axis| p[axis] + radius * direction[3 * point + axis])
        })
        .collect()
}
mod ascent;

#[cfg(test)]
mod tests;
