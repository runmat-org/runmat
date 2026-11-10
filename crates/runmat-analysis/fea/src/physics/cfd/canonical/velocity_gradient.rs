//! Frozen four-face geometry; each evaluation reads the current original field/law.
use super::{
    boundary::FaceBoundary, geometry_ops::*, reconstruction::GradientMetric,
    CanonicalFluidError as Error, CanonicalFluidProblem,
};
use crate::topology::FiniteVolumeTopology;

#[derive(Clone, Copy)]
enum Sample {
    Cell(usize),
    VelocityFace(usize),
    /// Mirror trace u − (u·n̂)n̂ at a symmetry plane, read from the owner cell.
    SymmetryFace(usize),
    ZeroNormalDerivative,
}
#[derive(Clone, Copy)]
pub(super) struct VelocityGradientStencil {
    weights: [(Sample, V); 4],
}
impl VelocityGradientStencil {
    pub(super) const RESIDENT_BYTES: u64 = std::mem::size_of::<Self>() as u64;
}

pub(super) fn derive(
    t: &FiniteVolumeTopology,
    boundaries: &[FaceBoundary],
) -> Result<Vec<VelocityGradientStencil>, Error> {
    let mut stencils = Vec::with_capacity(t.cells.len());
    for (c, cell) in t.cells.iter().enumerate() {
        let mut samples = [Sample::ZeroNormalDerivative; 4];
        let mut displacements = [[0.0; 3]; 4];
        for (j, incidence) in cell.faces.iter().enumerate() {
            let f = &t.faces[incidence.face_index];
            let other = if c == f.owner {
                f.neighbor
            } else {
                Some(f.owner)
            };
            if let Some(n) = other {
                samples[j] = Sample::Cell(n);
                displacements[j] = sub(t.cells[n].centroid_m, cell.centroid_m);
            } else if matches!(
                boundaries[incidence.face_index],
                FaceBoundary::Velocity { .. }
            ) {
                samples[j] = Sample::VelocityFace(incidence.face_index);
                displacements[j] = sub(f.centroid_m, cell.centroid_m);
            } else if matches!(
                boundaries[incidence.face_index],
                FaceBoundary::Symmetry { .. }
            ) {
                samples[j] = Sample::SymmetryFace(incidence.face_index);
                displacements[j] = sub(f.centroid_m, cell.centroid_m);
            } else {
                displacements[j] = scale(f.owner_area_vector_m2, 1.0 / f.area_m2.canonical_value());
            }
        }
        let at = |e| super::reconstruction::at_cell(e, cell, 4);
        let metric = GradientMetric::derive(&displacements).map_err(at)?;
        let mut weights = [(Sample::ZeroNormalDerivative, [0.0; 3]); 4];
        for j in 0..4 {
            weights[j] = (samples[j], metric.weight(displacements[j]).map_err(at)?);
        }
        stencils.push(VelocityGradientStencil { weights });
    }
    Ok(stencils)
}

pub(super) fn evaluate(p: &CanonicalFluidProblem<'_>, values: &[V]) -> Result<Vec<[V; 3]>, Error> {
    if values.len() != p.velocity_stencils.len() || values.iter().flatten().any(|x| !x.is_finite())
    {
        return Err(Error::Invalid("velocity gradient original field"));
    }
    super::geometry_ops::indexed(values.len(), |c| {
        let stencil = &p.velocity_stencils[c];
        let mut gradient = [[0.0; 3]; 3];
        for &(sample, weight) in &stencil.weights {
            let delta = match sample {
                Sample::Cell(n) => sub(values[n], values[c]),
                Sample::VelocityFace(i) => {
                    let FaceBoundary::Velocity { value, .. } = p.boundaries[i] else {
                        return Err(Error::Invalid("velocity gradient boundary authority"));
                    };
                    sub(value, values[c])
                }
                Sample::SymmetryFace(i) => {
                    let f = &p.topology.faces[i];
                    let normal = scale(f.owner_area_vector_m2, 1.0 / f.area_m2.canonical_value());
                    scale(normal, -dot(values[c], normal))
                }
                Sample::ZeroNormalDerivative => [0.0; 3],
            };
            for k in 0..3 {
                gradient[k] = add(gradient[k], scale(weight, delta[k]));
            }
        }
        if gradient.iter().flatten().any(|x| !x.is_finite()) {
            return Err(Error::Invalid("nonfinite velocity reconstruction"));
        }
        Ok(gradient)
    })
}

/// Linear part of the reconstruction: per cell, the cells it reads and the coefficients
/// T[k][j][d] = ∂(∂u_k/∂x_d)/∂u_j(m). Boundary values are constants and drop out; a symmetry
/// plane's mirror trace couples the owner's components through the plane normal.
pub(super) fn linear_terms(
    t: &FiniteVolumeTopology,
    stencils: &[VelocityGradientStencil],
) -> Vec<Vec<(usize, [[V; 3]; 3])>> {
    stencils
        .iter()
        .enumerate()
        .map(|(c, stencil)| {
            let mut terms: Vec<(usize, [[V; 3]; 3])> = Vec::with_capacity(5);
            let mut add_term = |m: usize, k: usize, j: usize, weight: V, factor: f64| {
                let at = match terms.iter().position(|(cell, _)| *cell == m) {
                    Some(at) => at,
                    None => {
                        terms.push((m, [[[0.0; 3]; 3]; 3]));
                        terms.len() - 1
                    }
                };
                for d in 0..3 {
                    terms[at].1[k][j][d] += factor * weight[d];
                }
            };
            for &(sample, weight) in &stencil.weights {
                match sample {
                    Sample::Cell(n) => {
                        for k in 0..3 {
                            add_term(n, k, k, weight, 1.0);
                            add_term(c, k, k, weight, -1.0);
                        }
                    }
                    Sample::VelocityFace(_) => {
                        for k in 0..3 {
                            add_term(c, k, k, weight, -1.0);
                        }
                    }
                    Sample::SymmetryFace(i) => {
                        let f = &t.faces[i];
                        let normal =
                            scale(f.owner_area_vector_m2, 1.0 / f.area_m2.canonical_value());
                        for k in 0..3 {
                            for j in 0..3 {
                                add_term(c, k, j, weight, -normal[k] * normal[j]);
                            }
                        }
                    }
                    Sample::ZeroNormalDerivative => {}
                }
            }
            terms.sort_by_key(|(cell, _)| *cell);
            terms
        })
        .collect()
}
