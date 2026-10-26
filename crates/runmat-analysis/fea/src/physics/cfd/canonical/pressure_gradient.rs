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
pub(super) struct PressureGradientStencil {
    weights: Vec<(Sample, V)>,
}

pub(super) fn derive(
    t: &FiniteVolumeTopology,
    b: &[FaceBoundary],
) -> Result<Vec<PressureGradientStencil>, Error> {
    let mut stencils = Vec::with_capacity(t.cells.len());
    for (root, cell) in t.cells.iter().enumerate() {
        // Degree≤4 tetrahedral face graph: depth3 has at most1+4+12+36=53 cells.
        let mut visited = BTreeSet::from([root]);
        let mut front = vec![root];
        for _ in 0..3 {
            let mut next = Vec::new();
            for c in front {
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
            front = next;
        }
        if visited.len() > 53 {
            return Err(Error::Invalid("pressure stencil graph degree"));
        }
        let mut samples = Vec::with_capacity(265);
        for c in visited {
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
        if samples.len() > 265 {
            return Err(Error::Invalid("pressure stencil sample bound"));
        }
        let displacements = samples
            .iter()
            .map(|(_, x)| sub(*x, cell.centroid_m))
            .collect::<Vec<_>>();
        let metric = super::reconstruction::GradientMetric::derive(&displacements)?;
        let weights = samples
            .into_iter()
            .zip(displacements)
            .map(|((sample, _), d)| metric.weight(d).map(|weight| (sample, weight)))
            .collect::<Result<Vec<_>, _>>()?;
        if weights.is_empty() {
            return Err(Error::Unsupported(
                "insufficient one-sided pressure samples",
            ));
        }
        stencils.push(PressureGradientStencil { weights });
    }
    Ok(stencils)
}

/// Reconstruct the gradient of original component-relative pressure.
pub(super) fn evaluate(p: &CanonicalFluidProblem<'_>, values: &[f64]) -> Result<Vec<V>, Error> {
    let mut result = Vec::with_capacity(values.len());
    for (c, stencil) in p.pressure_stencils.iter().enumerate() {
        let mut gradient = [0.0; 3];
        for (sample, weight) in &stencil.weights {
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
