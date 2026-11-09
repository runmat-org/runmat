//! One-sided pressure reconstruction: prescribed velocity does NOT prescribe a
//! zero normal pressure gradient. Original authoritative boundary volume flux is
//! imposed by the pressure equation; the unknown pressure trace is extrapolated
//! from real cell pressure and genuine pressure-outlet samples.
use super::{
    boundary::FaceBoundary, geometry_ops::*, CanonicalFluidError as Error, CanonicalFluidProblem,
};
use crate::topology::FiniteVolumeTopology;
use std::collections::BTreeSet;

#[derive(Clone, Copy)]
enum Sample {
    Cell(usize),
    OutletFace(usize),
}
/// Least-squares weights of every cell, stored contiguously: cell c reads
/// `weights[offsets[c]..offsets[c + 1]]`. One allocation for the whole mesh, sized to the
/// samples actually present rather than the per-cell worst case.
pub(super) struct PressureGradientStencils {
    offsets: Vec<usize>,
    weights: Vec<(Sample, V)>,
}

impl PressureGradientStencils {
    fn cell(&self, c: usize) -> &[(Sample, V)] {
        &self.weights[self.offsets[c]..self.offsets[c + 1]]
    }
}

/// Face-graph depth of the least-squares neighborhood: depth 3 for every cell whose ring spans
/// three dimensions; up to [`MAXIMUM_DEPTH`] where it does not.
const DEPTH: usize = 3;
/// A cell in a corner strip (one face on an inlet and one on a wall, say) can see only cells
/// of its own layer within three face steps, all with centroids in one plane. Their metric is
/// singular in the plane normal, so the neighborhood grows one face ring at a time until it spans.
const MAXIMUM_DEPTH: usize = 6;
/// Cells within `depth` face steps of a tetrahedron (degree ≤ 4): 1 + 4 (3^depth − 1) / 2.
const fn cell_bound(depth: usize) -> usize {
    1 + 2 * (3usize.pow(depth as u32) - 1)
}
/// Samples within `depth` face steps: each cell plus at most four outlet faces.
const fn sample_bound(depth: usize) -> usize {
    5 * cell_bound(depth)
}

/// The cells one face step beyond `front` that are not yet in `visited` (added to it).
fn next_ring(
    t: &FiniteVolumeTopology,
    front: &[usize],
    visited: &mut BTreeSet<usize>,
) -> Vec<usize> {
    let mut next = Vec::new();
    for &c in front {
        for incidence in t.cells[c].faces {
            let f = &t.faces[incidence.face_index];
            let other = if c == f.owner {
                f.neighbor
            } else {
                Some(f.owner)
            };
            if let Some(other) = other {
                if visited.insert(other) {
                    next.push(other);
                }
            }
        }
    }
    next
}

pub(super) fn derive(
    t: &FiniteVolumeTopology,
    b: &[FaceBoundary],
) -> Result<PressureGradientStencils, Error> {
    let mut offsets = Vec::with_capacity(t.cells.len() + 1);
    offsets.push(0);
    let mut all_weights = Vec::new();
    let mut samples = Vec::with_capacity(sample_bound(DEPTH));
    let mut displacements = Vec::with_capacity(sample_bound(DEPTH));
    for (root, cell) in t.cells.iter().enumerate() {
        let mut visited = BTreeSet::from([root]);
        let mut front = vec![root];
        let mut depth = 0;
        let metric = loop {
            front = next_ring(t, &front, &mut visited);
            depth += 1;
            if depth < DEPTH {
                continue;
            }
            if visited.len() > cell_bound(depth) {
                return Err(Error::Invalid("pressure stencil graph degree"));
            }
            samples.clear();
            for &c in &visited {
                if c != root {
                    samples.push((Sample::Cell(c), t.cells[c].centroid_m));
                }
                for incidence in t.cells[c].faces {
                    let i = incidence.face_index;
                    if matches!(b[i], FaceBoundary::PressureOutlet { .. }) {
                        samples.push((Sample::OutletFace(i), t.faces[i].centroid_m));
                    }
                }
            }
            if samples.len() > sample_bound(depth) {
                return Err(Error::Invalid("pressure stencil sample bound"));
            }
            if samples.is_empty() {
                return Err(Error::Unsupported(
                    "insufficient one-sided pressure samples",
                ));
            }
            displacements.clear();
            displacements.extend(samples.iter().map(|(_, x)| sub(*x, cell.centroid_m)));
            let metric = super::reconstruction::GradientMetric::derive(&displacements)
                .map_err(|e| super::reconstruction::at_cell(e, cell, displacements.len()))?;
            if metric.spans_three_dimensions() || depth >= MAXIMUM_DEPTH || front.is_empty() {
                break metric;
            }
        };
        let at = |e| super::reconstruction::at_cell(e, cell, displacements.len());
        for ((sample, _), d) in samples.iter().zip(&displacements) {
            all_weights.push((*sample, metric.weight(*d).map_err(at)?));
        }
        offsets.push(all_weights.len());
    }
    all_weights.shrink_to_fit();
    Ok(PressureGradientStencils {
        offsets,
        weights: all_weights,
    })
}

/// Reconstruct the gradient of original component-relative pressure.
pub(super) fn evaluate(p: &CanonicalFluidProblem<'_>, values: &[f64]) -> Result<Vec<V>, Error> {
    let mut result = Vec::with_capacity(values.len());
    for c in 0..p.pressure_stencils.offsets.len() - 1 {
        let mut gradient = [0.0; 3];
        for (sample, weight) in p.pressure_stencils.cell(c) {
            let value = match *sample {
                Sample::Cell(n) => values[n],
                Sample::OutletFace(i) => {
                    let FaceBoundary::PressureOutlet { pressure, .. } = p.boundaries[i] else {
                        return Err(Error::Invalid("pressure outlet stencil authority"));
                    };
                    pressure - p.pressure_reference_pa.canonical_values()[c]
                }
            };
            gradient = add(gradient, scale(*weight, value - values[c]));
        }
        if gradient.iter().any(|x| !x.is_finite()) {
            return Err(Error::Invalid("nonfinite pressure reconstruction"));
        }
        result.push(gradient);
    }
    Ok(result)
}

#[cfg(test)]
mod tests;
