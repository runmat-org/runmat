//! Conservative upwind advection and full Newtonian symmetric-gradient traction.
use super::{
    boundary::FaceBoundary, geometry_ops::*, linear, CanonicalFluidError as Error,
    CanonicalFluidProblem,
};
use crate::algebra::RealSparseMatrix;

pub(super) struct Momentum {
    pub matrix: RealSparseMatrix,
    pub rhs: Vec<V>,
    pub diagonal: Vec<f64>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum MomentumMode {
    /// Original unrelaxed equations; physical outlet backflow is unsupported.
    Original,
    /// Bounded SIMPLE trial only, never a publishable physical boundary state.
    SolverTrial,
    /// Unrelaxed operator of the coupled solve's Picard step; trial outlet backflow is
    /// tolerated as in a SIMPLE trial and never published.
    Coupled,
}

#[cfg(test)]
pub(super) fn assemble(
    p: &CanonicalFluidProblem<'_>,
    u: &[V],
    pressure_fluctuation: &[f64],
    q: &[f64],
    mode: MomentumMode,
) -> Result<Momentum, Error> {
    assemble_with_time(p, u, pressure_fluctuation, q, mode, None)
}

pub(super) fn assemble_with_time(
    p: &CanonicalFluidProblem<'_>,
    u: &[V],
    pressure_fluctuation: &[f64],
    q: &[f64],
    mode: MomentumMode,
    time: Option<super::transient::TimeDerivative<'_>>,
) -> Result<Momentum, Error> {
    let mut a = Some(linear::assembler(p, "incompressible-newtonian-momentum")?);
    let (rhs, diagonal) = parts(
        p,
        u,
        Some(pressure_fluctuation),
        q,
        mode,
        time,
        None,
        &mut a,
    )?;
    let matrix = a
        .expect("assembled operator")
        .finish()
        .map_err(|e| Error::Operator(e.to_string()))?;
    Ok(Momentum {
        matrix,
        rhs,
        diagonal,
    })
}

/// Right-hand side of the momentum equations at `u`, without the pressure force, with the same
/// coefficients (flux q, viscosity) as the assembled operator and without building it: the explicit
/// corrections (non-orthogonal diffusion, higher-order advection, symmetry traction) evaluated
/// at a trial velocity. Viscosity comes from `viscosity_gradients` (the frozen iterate), so a
/// shear-rate dependent law stays Picard-lagged and the map stays linear in `u`.
#[allow(clippy::too_many_arguments)]
pub(super) fn rhs_with_time(
    p: &CanonicalFluidProblem<'_>,
    u: &[V],
    q: &[f64],
    mode: MomentumMode,
    time: Option<super::transient::TimeDerivative<'_>>,
    viscosity_gradients: &[[V; 3]],
) -> Result<Vec<V>, Error> {
    Ok(parts(
        p,
        u,
        None,
        q,
        mode,
        time,
        Some(viscosity_gradients),
        &mut None,
    )?
    .0)
}

#[allow(clippy::too_many_arguments)]
fn parts(
    p: &CanonicalFluidProblem<'_>,
    u: &[V],
    pressure_fluctuation: Option<&[f64]>,
    q: &[f64],
    mode: MomentumMode,
    time: Option<super::transient::TimeDerivative<'_>>,
    viscosity_gradients: Option<&[[V; 3]]>,
    a: &mut Option<crate::algebra::RealSparseAssembler>,
) -> Result<(Vec<V>, Vec<f64>), Error> {
    let n = u.len();
    let mut entry = |i: usize, j: usize, v: f64| {
        if let Some(a) = a.as_mut() {
            a.add(i, j, v);
        }
    };
    let gu = velocity_gradients(p, u)?;
    let vgu = viscosity_gradients.unwrap_or(&gu);
    // Without a pressure the right-hand side omits the pressure force (the coupled step adds
    // its own at the trial pressure).
    let pf = match pressure_fluctuation {
        Some(values) => pressure_force(p, values, &pressure_gradient(p, values)?)?,
        None => vec![[0.0; 3]; n],
    };
    let mut rhs = (0..n)
        .map(|c| {
            sub(
                scale(p.body_force_n_per_m3[c], p.topology.cells[c].volume_m3),
                pf[c],
            )
        })
        .collect::<Vec<_>>();
    let mut diagonal = vec![0.0; n];
    for (i, f) in p.topology.faces.iter().enumerate() {
        if let FaceBoundary::Symmetry { .. } = p.boundaries[i] {
            // Impermeable and shear-free: no advective flux. The normal viscous
            // stress is implicit on every component; the tangential part of the
            // same coefficient is returned explicitly so only u·n̂ is penalized.
            let (coefficient, traction) = symmetry_traction(p, i, u[f.owner], &vgu[f.owner])?;
            entry(f.owner, f.owner, coefficient);
            diagonal[f.owner] += coefficient;
            rhs[f.owner] = add(rhs[f.owner], add(scale(u[f.owner], coefficient), traction));
            continue;
        }
        let (diffusion, correction) = viscous_parts_with(p, i, &gu, vgu)?;
        // Convection carries the flux relative to a moving mesh; continuity keeps q itself.
        let transported = q[i] - time.map_or(0.0, |t| t.mesh_flux(i));
        let mass_flux = p.density * transported;
        let provisional_backflow =
            matches!(p.boundaries[i], FaceBoundary::PressureOutlet { .. }) && transported < 0.0;
        let correction = if provisional_backflow {
            if mode == MomentumMode::Original {
                return Err(Error::Unsupported("pressure outlet backflow"));
            }
            // No clipped q or fabricated inlet value. The actual owner unknown
            // and its zero-normal-gradient outlet trace supply a fully explicit
            // signed trial transport. Original recovery never admits this branch.
            sub(
                correction,
                scale(face_velocity_trace(p, i, f.owner, u, &gu), mass_flux),
            )
        } else {
            sub(correction, advection_remainder(p, i, u, transported, &gu)?)
        };
        if let Some(other) = f.neighbor {
            let owner = diffusion + mass_flux.max(0.0);
            let neighbor = diffusion + (-mass_flux).max(0.0);
            entry(f.owner, f.owner, owner);
            entry(f.owner, other, -neighbor);
            entry(other, other, neighbor);
            entry(other, f.owner, -owner);
            diagonal[f.owner] += owner;
            diagonal[other] += neighbor;
            rhs[f.owner] = add(rhs[f.owner], correction);
            rhs[other] = sub(rhs[other], correction);
        } else {
            match &p.boundaries[i] {
                FaceBoundary::Velocity { value, .. } => {
                    let owner = diffusion + mass_flux.max(0.0);
                    entry(f.owner, f.owner, owner);
                    diagonal[f.owner] += owner;
                    rhs[f.owner] = add(
                        rhs[f.owner],
                        add(scale(*value, diffusion + (-mass_flux).max(0.0)), correction),
                    );
                }
                FaceBoundary::PressureOutlet { .. } => {
                    // Homogeneous normal velocity gradient; τ still includes transpose-gradient.
                    if !provisional_backflow {
                        entry(f.owner, f.owner, mass_flux);
                        diagonal[f.owner] += mass_flux;
                    }
                    rhs[f.owner] = add(rhs[f.owner], correction);
                }
                FaceBoundary::Symmetry { .. } | FaceBoundary::Interior => {
                    return Err(Error::Invalid("missing exterior law"))
                }
            }
        }
    }
    // Porous-zone resistance, implicit in each component: V (μ/K + ½ρC₂|u|), the quadratic
    // part Picard-lagged on the current speed.
    for c in 0..n {
        if let Some((linear, quadratic)) = p.porous[c] {
            let coefficient = p.topology.cells[c].volume_m3 * (linear + quadratic * norm(u[c]));
            entry(c, c, coefficient);
            diagonal[c] += coefficient;
        }
    }
    if let Some(time) = time {
        for c in 0..n {
            let coefficient = time.coefficient(p, c)?;
            entry(c, c, coefficient);
            diagonal[c] += coefficient;
            rhs[c] = add(
                rhs[c],
                scale(time.previous_velocity[c], time.previous_coefficient(p, c)?),
            );
        }
    }
    if mode == MomentumMode::SolverTrial {
        for c in 0..n {
            let extra = diagonal[c] * (1.0 / p.policy.momentum_relaxation - 1.0);
            entry(c, c, extra);
            rhs[c] = add(rhs[c], scale(u[c], extra));
            diagonal[c] += extra;
        }
    }
    if diagonal.iter().any(|v| !v.is_finite() || *v <= 0.0)
        || rhs.iter().flatten().any(|v| !v.is_finite())
    {
        return Err(Error::Invalid("momentum operator values"));
    }
    Ok((rhs, diagonal))
}
