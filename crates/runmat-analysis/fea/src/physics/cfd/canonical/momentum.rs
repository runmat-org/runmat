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
    let n = u.len();
    let mut a = linear::assembler(p, "incompressible-newtonian-momentum")?;
    let gu = velocity_gradients(p, u)?;
    let gp = pressure_gradient(p, pressure_fluctuation)?;
    let pf = pressure_force(p, pressure_fluctuation, &gp)?;
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
            let (coefficient, traction) = symmetry_traction(p, i, u[f.owner], &gu[f.owner])?;
            a.add(f.owner, f.owner, coefficient);
            diagonal[f.owner] += coefficient;
            rhs[f.owner] = add(rhs[f.owner], add(scale(u[f.owner], coefficient), traction));
            continue;
        }
        let (diffusion, correction) = viscous_parts(p, i, &gu)?;
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
            a.add(f.owner, f.owner, owner);
            a.add(f.owner, other, -neighbor);
            a.add(other, other, neighbor);
            a.add(other, f.owner, -owner);
            diagonal[f.owner] += owner;
            diagonal[other] += neighbor;
            rhs[f.owner] = add(rhs[f.owner], correction);
            rhs[other] = sub(rhs[other], correction);
        } else {
            match &p.boundaries[i] {
                FaceBoundary::Velocity { value, .. } => {
                    let owner = diffusion + mass_flux.max(0.0);
                    a.add(f.owner, f.owner, owner);
                    diagonal[f.owner] += owner;
                    rhs[f.owner] = add(
                        rhs[f.owner],
                        add(scale(*value, diffusion + (-mass_flux).max(0.0)), correction),
                    );
                }
                FaceBoundary::PressureOutlet { .. } => {
                    // Homogeneous normal velocity gradient; τ still includes transpose-gradient.
                    if !provisional_backflow {
                        a.add(f.owner, f.owner, mass_flux);
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
            let coefficient =
                p.topology.cells[c].volume_m3 * (linear + quadratic * norm(u[c]));
            a.add(c, c, coefficient);
            diagonal[c] += coefficient;
        }
    }
    if let Some(time) = time {
        for c in 0..n {
            let coefficient = time.coefficient(p, c)?;
            a.add(c, c, coefficient);
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
            a.add(c, c, extra);
            rhs[c] = add(rhs[c], scale(u[c], extra));
            diagonal[c] += extra;
        }
    }
    let matrix = a.finish().map_err(|e| Error::Operator(e.to_string()))?;
    if diagonal.iter().any(|v| !v.is_finite() || *v <= 0.0)
        || rhs.iter().flatten().any(|v| !v.is_finite())
    {
        return Err(Error::Invalid("momentum operator values"));
    }
    Ok(Momentum {
        matrix,
        rhs,
        diagonal,
    })
}
