use super::{
    geometry_ops::*, linear, momentum, pressure, recovery, CanonicalFluidError as Error,
    CanonicalFluidProblem, CanonicalFluidResult,
};

pub(super) fn solve(p: &CanonicalFluidProblem<'_>) -> Result<CanonicalFluidResult, Error> {
    solve_with_time(p, None, None, p.numerical_input_identity)
}

pub(super) fn solve_with_time(
    p: &CanonicalFluidProblem<'_>,
    initial: Option<super::transient::CanonicalFluidFields<'_>>,
    time: Option<super::transient::TimeDerivative<'_>>,
    numerical_input_identity: runmat_execution::Digest,
) -> Result<CanonicalFluidResult, Error> {
    if crate::is_fea_cancelled() {
        return Err(Error::Cancelled);
    }
    if p.policy.pressure_velocity_coupling == super::PressureVelocityCoupling::Coupled {
        return super::coupled::solve_with_time(p, initial, time, numerical_input_identity);
    }
    let n = p.topology.cells.len();
    let (mut u, mut pressure_fluctuation, mut q) = match initial {
        Some(fields) => (
            fields.velocity_m_per_s.canonical_values().to_vec(),
            fields.pressure_fluctuation_pa.canonical_values().to_vec(),
            fields.face_volume_flux_m3_per_s.canonical_values().to_vec(),
        ),
        None => {
            let u = vec![[0.0; 3]; n];
            let q = pressure::h_flux(p, &u, &vec![[0.0; 3]; n])?;
            (u, vec![0.0; n], q)
        }
    };
    let mut linear_iterations = 0;
    let mut last = [f64::INFINITY; 5];
    let mut worst = String::from("not evaluated");
    let mut first_worst: Option<f64> = None;
    let mut acceleration_history = None;
    for iteration in 1..=p.policy.maximum_outer_iterations {
        if crate::is_fea_cancelled() {
            return Err(Error::Cancelled);
        }
        let m = momentum::assemble_with_time(
            p,
            &u,
            &pressure_fluctuation,
            &q,
            momentum::MomentumMode::SolverTrial,
            time,
        )?;
        let mut predicted = vec![[0.0; 3]; n];
        for k in 0..3 {
            let b = m.rhs.iter().map(|v| v[k]).collect::<Vec<_>>();
            let current = u.iter().map(|v| v[k]).collect::<Vec<_>>();
            let (x, count) =
                linear::solve(p, &m.matrix, &b, &current, linear::LinearSystem::Momentum)?;
            linear_iterations += count;
            for c in 0..n {
                predicted[c][k] = x[c];
            }
        }
        let rau = (0..n)
            .map(|c| p.topology.cells[c].volume_m3 / m.diagonal[c])
            .collect::<Vec<_>>();
        let gp = pressure_gradient(p, &pressure_fluctuation)?;
        let old_force = pressure_force(p, &pressure_fluctuation, &gp)?;
        let pressure_term = (0..n)
            .map(|c| scale(old_force[c], rau[c] / p.topology.cells[c].volume_m3))
            .collect::<Vec<_>>();
        let h = (0..n)
            .map(|c| add(predicted[c], pressure_term[c]))
            .collect::<Vec<_>>();
        let qh = pressure::h_flux(p, &predicted, &pressure_term)?;
        let rf = pressure::coefficients(p, &rau)?;
        let (new_pressure, new_q, count) = pressure::solve(p, &qh, &rf, &pressure_fluctuation)?;
        linear_iterations += count;
        let mut pressure_defect = Vec::new();
        if p.policy.fixed_point_acceleration.is_some() {
            pressure_defect.reserve(n);
        }
        for c in 0..n {
            let delta = p.policy.pressure_relaxation * (new_pressure[c] - pressure_fluctuation[c]);
            pressure_fluctuation[c] += delta;
            if p.policy.fixed_point_acceleration.is_some() {
                pressure_defect.push(delta);
            }
        }
        p.pressure_reference_plan
            .normalize(p, &mut pressure_fluctuation)?;
        let gp = pressure_gradient(p, &pressure_fluctuation)?;
        let force = pressure_force(p, &pressure_fluctuation, &gp)?;
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
            acceleration_history = None;
            continue;
        }
        let residual = recovery::evaluate_with_time(p, &u, &pressure_fluctuation, &q, time)?;
        let residual = super::acceleration::consider(
            p,
            super::acceleration::Fields {
                velocity: &mut u,
                pressure: &mut pressure_fluctuation,
                flux: &mut q,
            },
            pressure_defect,
            &mut acceleration_history,
            residual,
            time,
        )?;
        last = [
            residual.momentum_relative,
            residual.continuity_relative,
            residual.flux_constitutive_relative,
            residual.global_momentum_relative,
            residual
                .cell_momentum_relative
                .iter()
                .copied()
                .fold(0.0, f64::max),
        ];
        let (worst_relative, description) = recovery::describe_worst_cell(p, &residual);
        let first = *first_worst.get_or_insert(worst_relative);
        worst = format!("{description}; it was {first:e} at the first evaluated step");
        if residual.accepted(p.policy.physical_relative_tolerance) {
            let pressure_pa = pressure_fluctuation
                .iter()
                .zip(p.pressure_reference_pa.canonical_values())
                .map(|(value, reference)| {
                    let absolute = value + reference;
                    if absolute.is_finite() {
                        Ok(absolute)
                    } else {
                        Err(Error::Invalid("nonfinite absolute pressure projection"))
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(CanonicalFluidResult {
                numerical_input_identity,
                authored_authority_identity: p.authored_authority_identity,
                velocity_m_per_s: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Velocity,
                    _,
                >::new(u),
                pressure_pa: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Pressure,
                    _,
                >::new(pressure_pa),
                pressure_fluctuation_pa: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Pressure,
                    _,
                >::new(pressure_fluctuation),
                pressure_reference_pa: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Pressure,
                    _,
                >::new(
                    p.pressure_reference_pa.canonical_values().clone()
                ),
                face_volume_flux_m3_per_s: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::VolumeFlowRate,
                    _,
                >::new(q),
                outer_iterations: iteration,
                linear_iterations,
                residual,
            });
        }
    }
    Err(Error::IterationExhausted {
        outer_iterations: p.policy.maximum_outer_iterations,
        linear_iterations,
        momentum_relative: last[0],
        continuity_relative: last[1],
        flux_constitutive_relative: last[2],
        global_momentum_relative: last[3],
        maximum_cell_momentum_relative: last[4],
        worst_cell: worst,
    })
}
