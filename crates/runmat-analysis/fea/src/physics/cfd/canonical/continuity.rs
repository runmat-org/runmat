//! Original oriented incidence and physical boundary flux, shared by initial
//! state admission and solve/recovery. No divergence repair or projection.
use super::{
    boundary::FaceBoundary, geometry_ops::dot, CanonicalFluidError as Error, CanonicalFluidProblem,
};

pub(super) struct Continuity {
    pub cell: Vec<f64>,
    pub relative: Vec<f64>,
    pub global: f64,
}

pub(super) fn evaluate(p: &CanonicalFluidProblem<'_>, q: &[f64]) -> Result<Continuity, Error> {
    for (i, face) in p.topology.faces.iter().enumerate() {
        let Some(value) = q.get(i) else {
            return Err(Error::Invalid("original face flux shape/value"));
        };
        if face.neighbor.is_none() {
            match p.boundaries[i] {
                FaceBoundary::Velocity {
                    value: velocity, ..
                } if *value != dot(velocity, face.owner_area_vector_m2) => {
                    return Err(Error::Invalid("retained prescribed boundary flux"))
                }
                FaceBoundary::Symmetry { .. } if *value != 0.0 => {
                    return Err(Error::Invalid("retained symmetry-plane flux"))
                }
                FaceBoundary::PressureOutlet { .. } if *value < 0.0 => {
                    return Err(Error::Unsupported("pressure outlet backflow"))
                }
                _ => {}
            }
        }
    }
    incidence(p, q)
}

/// Previous boundary laws belong to the predecessor, not the receiving step.
pub(super) fn incidence(p: &CanonicalFluidProblem<'_>, q: &[f64]) -> Result<Continuity, Error> {
    if q.len() != p.topology.faces.len() || q.iter().any(|v| !v.is_finite()) {
        return Err(Error::Invalid("original face flux shape/value"));
    }
    let mut cell = vec![0.0; p.topology.cells.len()];
    let mut scale = vec![0.0; cell.len()];
    let mut global = 0.0;
    for (i, face) in p.topology.faces.iter().enumerate() {
        cell[face.owner] += q[i];
        scale[face.owner] += q[i].abs();
        if let Some(neighbor) = face.neighbor {
            cell[neighbor] -= q[i];
            scale[neighbor] += q[i].abs();
        } else {
            global += q[i];
        }
    }
    let relative = cell
        .iter()
        .zip(&scale)
        .map(|(r, s)| super::recovery::relative(r.abs(), *s))
        .collect();
    Ok(Continuity {
        cell,
        relative,
        global,
    })
}
