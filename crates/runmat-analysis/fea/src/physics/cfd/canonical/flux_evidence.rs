//! Original geometric face momentum fluxes, all forces in newtons.
use super::{
    boundary::FaceBoundary, geometry_ops::*, CanonicalFluidError as Error, CanonicalFluidProblem,
};

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalFluidFaceForces {
    #[serde(with = "runmat_types::physical::canonical_storage")]
    /// ρ q u_upwind, oriented from owner; paired incidence reverses this vector.
    pub advective_momentum_n:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Force, Vec<V>>,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    /// Original μ(∇u+∇uᵀ)·S, including αΔu and geometric nonorthogonal remainder.
    pub viscous_traction_n:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Force, Vec<V>>,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    /// −p_f S, physical pressure Pa × area m², force exerted on owner fluid.
    pub pressure_traction_n:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Force, Vec<V>>,
    #[serde(with = "runmat_types::physical::canonical_storage")]
    /// Pressure fluctuation relative to the authored component outlet or gauge, for
    /// gauge-independent convergence scaling, NOT a substituted physical pressure.
    pub pressure_fluctuation_traction_n:
        runmat_types::physical::Quantity<runmat_types::physical::kind::Force, Vec<V>>,
}

/// Face forces and the cell vorticity ∇×u, both from the same boundary-aware velocity
/// gradient that defines the viscous stress.
pub(super) fn evaluate(
    p: &CanonicalFluidProblem<'_>,
    u: &[V],
    pressure_fluctuation: &[f64],
    q: &[f64],
    mesh_flux: Option<&[f64]>,
) -> Result<(CanonicalFluidFaceForces, Vec<V>), Error> {
    let gu = velocity_gradients(p, u)?;
    // Row k of a gradient is ∇u_k: ω = (∂w/∂y − ∂v/∂z, ∂u/∂z − ∂w/∂x, ∂v/∂x − ∂u/∂y).
    let vorticity = gu
        .iter()
        .map(|g| [g[2][1] - g[1][2], g[0][2] - g[2][0], g[1][0] - g[0][1]])
        .collect::<Vec<_>>();
    let gp = pressure_gradient(p, pressure_fluctuation)?;
    let mut advective = Vec::with_capacity(q.len());
    let mut viscous = Vec::with_capacity(q.len());
    let mut pressure_force = Vec::with_capacity(q.len());
    let mut fluctuation = Vec::with_capacity(q.len());
    for (i, f) in p.topology.faces.iter().enumerate() {
        let pf = face_pressure_offset(p, i, pressure_fluctuation, &gp)?;
        if let FaceBoundary::Symmetry { .. } = p.boundaries[i] {
            advective.push([0.0; 3]);
            viscous.push(symmetry_traction(p, i, u[f.owner], &gu[f.owner])?.1);
        } else {
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
            let transported = q[i] - mesh_flux.map_or(0.0, |w| w[i]);
            advective.push(scale(
                face_upwind_velocity(p, i, u, transported, &gu)?,
                p.density * transported,
            ));
            viscous.push(add(scale(delta, diffusion), remainder));
        }
        pressure_force.push(scale(
            f.owner_area_vector_m2,
            -(pf + p.pressure_reference_pa.canonical_values()[f.owner]),
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
    Ok((
        CanonicalFluidFaceForces {
            advective_momentum_n: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Force,
                _,
            >::new(advective),
            viscous_traction_n: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Force,
                _,
            >::new(viscous),
            pressure_traction_n: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Force,
                _,
            >::new(pressure_force),
            pressure_fluctuation_traction_n: runmat_types::physical::Quantity::<
                runmat_types::physical::kind::Force,
                _,
            >::new(fluctuation),
        },
        vorticity,
    ))
}
