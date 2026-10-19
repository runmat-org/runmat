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
            } else {
                displacements[j] = scale(f.owner_area_vector_m2, 1.0 / f.area_m2);
            }
        }
        let metric = GradientMetric::derive(&displacements)?;
        let mut weights = [(Sample::ZeroNormalDerivative, [0.0; 3]); 4];
        for j in 0..4 {
            weights[j] = (samples[j], metric.weight(displacements[j])?);
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
    let mut gradients = Vec::with_capacity(values.len());
    for (c, stencil) in p.velocity_stencils.iter().enumerate() {
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
                Sample::ZeroNormalDerivative => [0.0; 3],
            };
            for k in 0..3 {
                gradient[k] = add(gradient[k], scale(weight, delta[k]));
            }
        }
        if gradient.iter().flatten().any(|x| !x.is_finite()) {
            return Err(Error::Invalid("nonfinite velocity reconstruction"));
        }
        gradients.push(gradient);
    }
    Ok(gradients)
}
