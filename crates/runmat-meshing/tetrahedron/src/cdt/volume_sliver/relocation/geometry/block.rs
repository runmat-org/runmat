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
    /// Cells incident to each free point. A coordinate poll moves one point,
    /// so only that star is re-evaluated against the cached complement.
    slot_cells: Vec<Vec<usize>>,
}
/// Cell values at the current accepted points. `None` marks a cell whose
/// orientation is not positive, which scores the whole block as zero.
struct Accepted {
    values: Vec<Option<f64>>,
    /// Per slot: (complement has a non-positive cell, complement minimum).
    complement: Vec<(bool, f64)>,
}
impl PreparedBlock {
    pub(in crate::cdt::volume_sliver::relocation) fn prepare(
        topology: &(impl VolumeTopologyRead + ?Sized),
        vertices: &[u32],
        cells: &BTreeSet<u32>,
        metrics: &dyn Fn(u32) -> runmat_meshing_size::metric::MetricTensor3,
    ) -> Result<Self, DelaunayVolumeSliverError> {
        let start = vertices
            .iter()
            .map(|vertex| topology.node(*vertex).coordinates_m.canonical_value())
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
                            .map(|axis| {
                                nodes[a].coordinates_m.canonical_value()[axis]
                                    - nodes[b].coordinates_m.canonical_value()[axis]
                            })
                            .fold(0., f64::hypot);
                        if length > 0. {
                            scale = scale.max(length);
                        }
                    }
                }
                Ok(Cell {
                    nodes,
                    slots: indices.map(|vertex| vertices.iter().position(|v| *v == vertex)),
                    metric: PreparedJacobian::new(metrics(*index))
                        .map_err(|cause| {
                            error(
                                DelaunayVolumeSliverErrorKind::Accounting(cause),
                                "block metric numerical accounting failed",
                            )
                        })?
                        .expect("admitted metric"),
                })
            })
            .collect::<Result<Vec<Cell>, DelaunayVolumeSliverError>>()?;
        let mut slot_cells = vec![Vec::new(); vertices.len()];
        for (index, cell) in cells.iter().enumerate() {
            for slot in cell.slots.iter().flatten() {
                slot_cells[*slot].push(index);
            }
        }
        Ok(Self {
            cells,
            start,
            scale,
            slot_cells,
        })
    }
    fn cell_value(
        &self,
        cell: &Cell,
        points: &[[f64; 3]],
    ) -> Result<Option<f64>, DelaunayVolumeSliverError> {
        let nodes = std::array::from_fn(|slot| {
            let mut node = cell.nodes[slot];
            if let Some(index) = cell.slots[slot] {
                node.coordinates_m = runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new(points[index]);
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
            return Ok(None);
        }
        Ok(Some(
            cell.metric
                .score(nodes.map(|node| node.coordinates_m.canonical_value()))
                .map_err(|cause| {
                    error(
                        DelaunayVolumeSliverErrorKind::Accounting(cause),
                        "block score numerical accounting failed",
                    )
                })?
                .unwrap_or(0.),
        ))
    }
    fn accept(&self, points: &[[f64; 3]]) -> Result<Accepted, DelaunayVolumeSliverError> {
        let values = self
            .cells
            .iter()
            .map(|cell| self.cell_value(cell, points))
            .collect::<Result<Vec<_>, _>>()?;
        let complement = (0..self.start.len())
            .map(|slot| {
                self.cells
                    .iter()
                    .zip(&values)
                    .filter(|(cell, _)| !cell.slots.contains(&Some(slot)))
                    .fold(
                        (false, f64::INFINITY),
                        |(inverted, minimum), (_, value)| match value {
                            Some(value) => (inverted, minimum.min(*value)),
                            None => (true, minimum),
                        },
                    )
            })
            .collect();
        Ok(Accepted { values, complement })
    }
    /// Exactly `score(points)` when `points` differs from the accepted points
    /// only at `slot`; only that point's star is evaluated.
    fn poll(
        &self,
        accepted: &Accepted,
        slot: usize,
        points: &[[f64; 3]],
    ) -> Result<f64, DelaunayVolumeSliverError> {
        let (inverted, mut minimum) = accepted.complement[slot];
        if inverted {
            return Ok(0.);
        }
        for index in &self.slot_cells[slot] {
            match self.cell_value(&self.cells[*index], points)? {
                Some(value) => minimum = minimum.min(value),
                None => return Ok(0.),
            }
        }
        Ok(minimum)
    }
    fn score(&self, points: &[[f64; 3]]) -> Result<f64, DelaunayVolumeSliverError> {
        let mut minimum = f64::INFINITY;
        for cell in &self.cells {
            match self.cell_value(cell, points)? {
                Some(value) => minimum = minimum.min(value),
                None => return Ok(0.),
            }
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
            .flat_map(|axis| [-1., 1.].map(move |sign| (axis, sign)))
            .collect::<Vec<_>>();
        let mut points = self.start.clone();
        let mut accepted_values = self.accept(&points)?;
        let initial = aggregate(&accepted_values.values);
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
            if let Some(direction) = self.ascent_direction(&points, radius)? {
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
                    accepted_values = self.accept(&trial)?;
                    points = trial;
                    // A rejected large-radius model must not be proposed again
                    // after only a tiny fraction passed the nonlinear check.
                    // Successful full steps can grow back to the original cap.
                    radius = (2. * accepted_step).min(maximum_radius);
                    continue;
                }
            }
            let mut best = None;
            let results = runmat_cpu_execution::ordered_map(&directions, |&(axis, sign)| {
                let mut trial = points.clone();
                trial[axis / 3][axis % 3] += radius * sign;
                self.poll(&accepted_values, axis / 3, &trial)
                    .map(|value| (value, trial))
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
                accepted_values = self.accept(&trial)?;
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

fn aggregate(values: &[Option<f64>]) -> f64 {
    values
        .iter()
        .try_fold(f64::INFINITY, |minimum, value| {
            value.map(|v| minimum.min(v))
        })
        .unwrap_or(0.)
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
