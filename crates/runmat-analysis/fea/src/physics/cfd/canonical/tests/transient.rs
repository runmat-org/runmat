use super::super::{
    geometry_ops::{dot, V},
    transient::CanonicalFluidFields,
    CanonicalFluidError as Error,
};
use super::closed_reference::closed;
use super::*;

fn original_uniform(
    p: &CanonicalFluidProblem<'_>,
    velocity: V,
    gradient: V,
) -> (Vec<V>, Vec<f64>, Vec<f64>) {
    let u = vec![velocity; p.topology.cells.len()];
    let mut pressure = p
        .topology
        .cells
        .iter()
        .map(|c| dot(gradient, c.centroid_m))
        .collect::<Vec<_>>();
    p.pressure_reference_plan
        .normalize(p, &mut pressure)
        .unwrap();
    let flux = p
        .topology
        .faces
        .iter()
        .map(|f| dot(velocity, f.owner_area_vector_m2))
        .collect();
    (u, pressure, flux)
}
fn fields<'a>(
    problem: &'a CanonicalFluidProblem<'_>,
    u: &'a [V],
    p: &'a [f64],
    q: &'a [f64],
) -> CanonicalFluidFields<'a> {
    CanonicalFluidFields {
        pressure_reference_pa: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Pressure,
            _,
        >::new(problem.pressure_reference_pa.canonical_values()),
        velocity_m_per_s: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Velocity,
            _,
        >::new(u),
        pressure_fluctuation_pa: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Pressure,
            _,
        >::new(p),
        face_volume_flux_m3_per_s: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::VolumeFlowRate,
            _,
        >::new(q),
    }
}

// witness: FEA-FLUID-TRANSIENT-001/inertia @
//   incompressible_laminar_transient/canonical_tetrahedra/numerical_kernel; coverage:
//   partial;
//   check: uniform-acceleration; boundary: A uniformly accelerating field has the exact inertia-driven pressure and binds cold;
//   a nontrivial unsteady reference is not covered here.
#[test]
fn original_uniform_acceleration_has_exact_inertia_pressure_and_cold_binding() {
    let acceleration = [0.01, -0.005, 0.002];
    let initial_velocity = [0.001, 0.002, -0.001];
    let duration = 0.2;
    let final_velocity = std::array::from_fn(|k| initial_velocity[k] + duration * acceleration[k]);
    for reference in [0.0, 101325.0] {
        let f = closed(2, final_velocity, reference);
        let p = CanonicalFluidProblem::prepare(f.input()).unwrap();
        let exact_gradient = acceleration.map(|a| -p.density * a);
        let (u, pressure, flux) = original_uniform(&p, initial_velocity, exact_gradient);
        let step = p
            .prepare_backward_euler(3.0, 3.2, fields(&p, &u, &pressure, &flux))
            .unwrap();
        assert!(
            step.reservation().resident_bytes_upper_bound
                > p.reservation().resident_bytes_upper_bound
        );
        let result = step.solve().unwrap();
        let cold: CanonicalFluidResult =
            serde_json::from_slice(&serde_json::to_vec(&result).unwrap()).unwrap();
        assert_eq!(step.recover(&cold).unwrap(), result.residual);
        assert!(
            p.recover(&cold).is_err(),
            "transient result cannot claim steady equations"
        );
        for velocity in result.velocity_m_per_s.canonical_values() {
            assert!(norm(sub(*velocity, final_velocity)) < 1e-8);
        }
        for gradient in super::super::geometry_ops::pressure_gradient(
            &p,
            result.pressure_fluctuation_pa.canonical_values(),
        )
        .unwrap()
        {
            assert!(
                norm(sub(gradient, exact_gradient)) < 1e-6,
                "independent rho du/dt + gradp = 0: {gradient:?}"
            );
        }
        assert!(p
            .prepare_backward_euler(3.0, 3.4, fields(&p, &u, &pressure, &flux))
            .unwrap()
            .recover(&cold)
            .is_err());
        let mut foreign = u.clone();
        foreign[0][0] += 0.001;
        assert!(p
            .prepare_backward_euler(3.0, 3.2, fields(&p, &foreign, &pressure, &flux))
            .unwrap()
            .recover(&cold)
            .is_err());
        let mut changed = cold;
        changed.velocity_m_per_s = changed
            .velocity_m_per_s
            .clone()
            .try_map_canonical_storage(|mut values| {
                values[0][0] += 0.01;
                values
            })
            .unwrap();
        assert!(step.recover(&changed).is_err());
    }
}

// witness: FEA-FLUID-TRANSIENT-001/convergence @
//   incompressible_laminar_transient/canonical_tetrahedra/numerical_kernel;
//   check: exponential-translation; boundary: Uniform exponential translation pressure on three time steps; no vortex or
//   independent spatial sweep.
#[test]
fn exponential_translation_pressure_shows_first_order_time_convergence() {
    // Independent continuum solution: u=exp(-t)e_x, gradp=rho exp(-t)e_x.
    // There is no body force, convection divergence or viscous strain. Exact
    // boundary velocity drives each fresh physical interval, not a solver callback.
    let mut errors = Vec::new();
    for dt in [0.1, 0.05, 0.025] {
        let end = 0.2_f64;
        let count = (end / dt).round() as usize;
        let mut retained: Option<CanonicalFluidResult> = None;
        for ordinal in 0..count {
            let start = ordinal as f64 * dt;
            let finish = (ordinal + 1) as f64 * dt;
            let f = closed(2, [(-finish).exp(), 0.0, 0.0], 101325.0);
            let p = CanonicalFluidProblem::prepare(f.input()).unwrap();
            let initial = original_uniform(&p, [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
            let previous = retained
                .as_ref()
                .map(CanonicalFluidResult::original_fields)
                .unwrap_or_else(|| fields(&p, &initial.0, &initial.1, &initial.2));
            let step = p.prepare_backward_euler(start, finish, previous).unwrap();
            let result = step.solve().unwrap();
            step.recover(&result).unwrap();
            if ordinal + 1 == count {
                let exact = [(-end).exp(), 0.0, 0.0];
                let mut error = 0.0;
                let mut measure = 0.0;
                for (cell, gradient) in p.topology.cells.iter().zip(
                    super::super::geometry_ops::pressure_gradient(
                        &p,
                        result.pressure_fluctuation_pa.canonical_values(),
                    )
                    .unwrap(),
                ) {
                    error += cell.volume_m3 * norm(sub(gradient, exact)).powi(2);
                    measure += cell.volume_m3 * norm(exact).powi(2);
                }
                errors.push((error / measure).sqrt());
            }
            retained = Some(result);
        }
    }
    eprintln!("independent exponential pressure time errors: {errors:?}");
    assert!(errors[1] < 0.6 * errors[0] && errors[2] < 0.6 * errors[1] && errors[2] < 0.02);
}

#[test]
fn interval_admission_rejects_order_nonfinite_divergence_budget_and_cancellation() {
    let f = closed(2, [0.0; 3], 0.0);
    let p = CanonicalFluidProblem::prepare(f.input()).unwrap();
    let (u, pressure, mut q) = original_uniform(&p, [0.0; 3], [0.0; 3]);
    for (start, end) in [
        (1.0, 1.0),
        (2.0, 1.0),
        (0.0, f64::INFINITY),
        (-0.0, 1.0),
        (-1.0, -0.0),
        (0.0, f64::MIN_POSITIVE * f64::EPSILON),
    ] {
        assert!(p
            .prepare_backward_euler(start, end, fields(&p, &u, &pressure, &q))
            .is_err());
    }
    q[0] = 1.0;
    assert!(p
        .prepare_backward_euler(0.0, 1.0, fields(&p, &u, &pressure, &q))
        .is_err());
    q[0] = 0.0;
    let mut limited_fixture = closed(2, [0.0; 3], 0.0);
    limited_fixture.policy.maximum_resident_bytes =
        Some(p.reservation().resident_bytes_upper_bound);
    let limited = CanonicalFluidProblem::prepare(limited_fixture.input()).unwrap();
    assert!(matches!(
        limited.prepare_backward_euler(0.0, 1.0, fields(&p, &u, &pressure, &q)),
        Err(Error::Resource(_))
    ));
    let mut work_fixture = closed(2, [0.0; 3], 0.0);
    work_fixture.policy.maximum_operator_work = Some(p.reservation().operator_work_upper_bound);
    let work_limited = CanonicalFluidProblem::prepare(work_fixture.input()).unwrap();
    assert!(matches!(
        work_limited.prepare_backward_euler(0.0, 1.0, fields(&p, &u, &pressure, &q)),
        Err(Error::Resource(_))
    ));
    let mut malformed = u.clone();
    malformed[0][0] = f64::NAN;
    assert!(p
        .prepare_backward_euler(0.0, 1.0, fields(&p, &malformed, &pressure, &q))
        .is_err());
    assert!(p
        .prepare_backward_euler(0.0, 1.0, fields(&p, &u[1..], &pressure, &q))
        .is_err());
    assert!(p
        .prepare_backward_euler(0.0, 1.0, fields(&p, &u, &pressure, &q[1..]))
        .is_err());
    let step = p
        .prepare_backward_euler(0.0, 1.0, fields(&p, &u, &pressure, &q))
        .unwrap();
    let cancellation = crate::progress::FeaCancellationToken::new();
    cancellation.cancel();
    let _guard =
        crate::progress::replace_fea_progress_context(crate::progress::FeaProgressContext {
            cancellation,
            ..Default::default()
        });
    assert!(matches!(step.solve(), Err(Error::Cancelled)));
    assert!(matches!(
        p.prepare_backward_euler(0.0, 1.0, fields(&p, &u, &pressure, &q)),
        Err(Error::Cancelled)
    ));
}

// witness: FEA-FLUID-TRANSIENT-001/inertia @
//   incompressible_laminar_transient/canonical_tetrahedra/numerical_kernel;
//   boundary: Independent rho*volume*delta-U/dt cell action on every axis; not temporal field
//   convergence.
#[test]
fn original_mass_term_matches_independent_cell_integral_on_every_axis() {
    use super::super::{
        momentum::{self, MomentumMode},
        transient::TimeDerivative,
    };
    use crate::algebra::spmv;
    let f = closed(2, [0.0; 3], 0.0);
    let p = CanonicalFluidProblem::prepare(f.input()).unwrap();
    let (u, pressure, q) = original_uniform(&p, [0.0; 3], [0.0; 3]);
    let old = p
        .topology
        .cells
        .iter()
        .map(|c| c.centroid_m.map(|v| 0.01 * v))
        .collect::<Vec<_>>();
    let current = p
        .topology
        .cells
        .iter()
        .map(|c| c.centroid_m.map(|v| -0.02 * v))
        .collect::<Vec<_>>();
    let dt = 0.125;
    let steady = momentum::assemble(&p, &u, &pressure, &q, MomentumMode::Original).unwrap();
    let transient = momentum::assemble_with_time(
        &p,
        &u,
        &pressure,
        &q,
        MomentumMode::Original,
        Some(TimeDerivative {
            previous_velocity: &old,
            duration_s: dt,
            motion: None,
        }),
    )
    .unwrap();
    for k in 0..3 {
        let component = current.iter().map(|v| v[k]).collect::<Vec<_>>();
        let base = spmv(&steady.matrix, &component).unwrap();
        let timed = spmv(&transient.matrix, &component).unwrap();
        for (c, cell) in p.topology.cells.iter().enumerate() {
            let difference = (timed[c] - transient.rhs[c][k]) - (base[c] - steady.rhs[c][k]);
            let exact = p.density * cell.volume_m3 * (current[c][k] - old[c][k]) / dt;
            assert!((difference - exact).abs() < 1e-14);
        }
    }
}
