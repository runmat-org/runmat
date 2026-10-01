use super::{
    geometry_ops::*, linear, momentum, pressure, recovery, CanonicalFluidError as Error,
    CanonicalFluidProblem, CanonicalFluidResult,
};

pub(super) fn solve(p: &CanonicalFluidProblem<'_>) -> Result<CanonicalFluidResult, Error> {
    let n = p.topology.cells.len();
    let mut u = vec![[0.0; 3]; n];
    let mut physical_pressure = p.pressure_reference_pa.clone();
    let mut q = pressure::h_flux(p, &u, &vec![[0.0; 3]; n])?;
    let mut linear_iterations = 0;
    let mut last = (f64::INFINITY, f64::INFINITY, f64::INFINITY);
    for iteration in 1..=p.policy.maximum_outer_iterations {
        let m = momentum::assemble(
            p,
            &u,
            &physical_pressure,
            &q,
            momentum::MomentumMode::SolverTrial,
        )?;
        let mut predicted = vec![[0.0; 3]; n];
        for k in 0..3 {
            let b = m.rhs.iter().map(|v| v[k]).collect::<Vec<_>>();
            let (x, count) = linear::solve(p, &m.matrix, &b)?;
            linear_iterations += count;
            for c in 0..n {
                predicted[c][k] = x[c];
            }
        }
        let rau = (0..n)
            .map(|c| p.topology.cells[c].volume_m3 / m.diagonal[c])
            .collect::<Vec<_>>();
        let gp = gradient(p, &physical_pressure, None)?;
        let old_force = pressure_force(p, &physical_pressure, &gp)?;
        let pressure_term = (0..n)
            .map(|c| scale(old_force[c], rau[c] / p.topology.cells[c].volume_m3))
            .collect::<Vec<_>>();
        let h = (0..n)
            .map(|c| add(predicted[c], pressure_term[c]))
            .collect::<Vec<_>>();
        let qh = pressure::h_flux(p, &predicted, &pressure_term)?;
        let rf = pressure::coefficients(p, &rau)?;
        let (new_pressure, new_q, count) = pressure::solve(p, &qh, &rf, &physical_pressure)?;
        linear_iterations += count;
        for c in 0..n {
            physical_pressure[c] +=
                p.policy.pressure_relaxation * (new_pressure[c] - physical_pressure[c]);
        }
        let gp = gradient(p, &physical_pressure, None)?;
        let force = pressure_force(p, &physical_pressure, &gp)?;
        u = (0..n)
            .map(|c| {
                sub(
                    h[c],
                    scale(force[c], rau[c] / p.topology.cells[c].volume_m3),
                )
            })
            .collect();
        q = new_q;
        // A trial may temporarily cross the physical outlet's admissible set.
        // Continue the bounded iteration without evaluating physical acceptance;
        // no field with outlet backflow can be recovered or published.
        if p.topology.faces.iter().enumerate().any(|(i, f)| {
            f.neighbor.is_none()
                && matches!(
                    p.boundaries[i],
                    super::boundary::FaceBoundary::PressureOutlet { .. }
                )
                && q[i] < 0.0
        }) {
            continue;
        }
        let residual = recovery::evaluate(p, &u, &physical_pressure, &q)?;
        last = (
            residual.momentum_relative,
            residual.continuity_relative,
            residual.flux_constitutive_relative,
        );
        if residual.accepted(p.policy.physical_relative_tolerance) {
            return Ok(CanonicalFluidResult {
                numerical_input_identity: p.numerical_input_identity,
                authored_authority_identity: p.authored_authority_identity,
                velocity_m_per_s: u,
                pressure_pa: physical_pressure,
                face_volume_flux_m3_per_s: q,
                outer_iterations: iteration,
                linear_iterations,
                residual,
            });
        }
    }
    Err(Error::IterationExhausted {
        outer_iterations: p.policy.maximum_outer_iterations,
        linear_iterations,
        momentum_relative: last.0,
        continuity_relative: last.1,
        flux_constitutive_relative: last.2,
    })
}
