//! Original geometric face momentum fluxes, all forces in newtons.
use super::{
    boundary::FaceBoundary, geometry_ops::*, CanonicalFluidError as Error, CanonicalFluidProblem,
};

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalFluidFaceForces {
    /// ρ q u_upwind, oriented from owner; paired incidence reverses this vector.
    pub advective_momentum_n: Vec<V>,
    /// Original μ(∇u+∇uᵀ)·S, including αΔu and geometric nonorthogonal remainder.
    pub viscous_traction_n: Vec<V>,
    /// −p_f S, physical pressure Pa × area m², force exerted on owner fluid.
    pub pressure_traction_n: Vec<V>,
    /// Pressure fluctuation relative to the authored component outlet or gauge, for
    /// gauge-independent convergence scaling, NOT a substituted physical pressure.
    pub pressure_fluctuation_traction_n: Vec<V>,
}

pub(super) fn evaluate(
    p: &CanonicalFluidProblem<'_>,
    u: &[V],
    pressure_fluctuation: &[f64],
    q: &[f64],
) -> Result<CanonicalFluidFaceForces, Error> {
    let gu = velocity_gradients(p, u)?;
    let gp = pressure_gradient(p, pressure_fluctuation)?;
    let mut advective = Vec::with_capacity(q.len());
    let mut viscous = Vec::with_capacity(q.len());
    let mut pressure_force = Vec::with_capacity(q.len());
    let mut fluctuation = Vec::with_capacity(q.len());
    for (i, f) in p.topology.faces.iter().enumerate() {
        let (diffusion, remainder) = viscous_parts(p, i, &gu)?;
        let delta = if let Some(n) = f.neighbor {
            sub(u[n], u[f.owner])
        } else {
            match p.boundaries[i] {
                FaceBoundary::Velocity { value, .. } => sub(value, u[f.owner]),
                FaceBoundary::PressureOutlet { .. } => [0.0; 3],
                _ => return Err(Error::Invalid("missing traction boundary")),
            }
        };
        let pf = face_pressure_offset(p, i, pressure_fluctuation, &gp)?;
        advective.push(scale(
            face_upwind_velocity(p, i, u, q[i], &gu)?,
            p.density * q[i],
        ));
        viscous.push(add(scale(delta, diffusion), remainder));
        pressure_force.push(scale(
            f.owner_area_vector_m2,
            -(pf + p.pressure_reference_pa[f.owner]),
        ));
        fluctuation.push(scale(f.owner_area_vector_m2, -pf));
    }
    if advective
        .iter()
        .chain(&viscous)
        .chain(&pressure_force)
        .chain(&fluctuation)
        .flatten()
        .any(|x| !x.is_finite())
    {
        return Err(Error::Invalid("nonfinite original face force"));
    }
    Ok(CanonicalFluidFaceForces {
        advective_momentum_n: advective,
        viscous_traction_n: viscous,
        pressure_traction_n: pressure_force,
        pressure_fluctuation_traction_n: fluctuation,
    })
}
