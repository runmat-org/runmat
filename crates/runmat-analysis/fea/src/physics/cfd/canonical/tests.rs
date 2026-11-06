//! Independent continuum evidence on canonical fixture meshes. Boundary authority
//! is frozen triangle-centroid samples; this is not authored CAD profile support.
mod acceleration;
mod authority;
mod boundary_balances;
mod closed_reference;
mod kinematic;
use super::qualification::fixture;
mod linear_correction;
mod mass_flow;
mod moving;
mod plan_cache;
mod porous;
mod preconditioner_envelope;
mod pressure_reference;
mod rheology;
mod reconstruction;
mod symmetry;
use super::qualification::reference;
mod time_signals;
mod transient;

use super::qualification::{cell_centroid_velocity_error as l2, wall_shear_error};
use super::*;
use fixture::{case, material, policy, ReferenceCase};
use geometry_ops::{norm, sub};

#[test]
fn exhausted_iteration_reports_every_physical_acceptance_residual() {
    let mut f = case(1, ReferenceCase::Convective, false);
    f.policy.maximum_outer_iterations = 1;
    let p = CanonicalFluidProblem::prepare(f.input()).unwrap();
    let error = p.solve().unwrap_err();
    let CanonicalFluidError::IterationExhausted {
        outer_iterations,
        momentum_relative,
        continuity_relative,
        flux_constitutive_relative,
        global_momentum_relative,
        maximum_cell_momentum_relative,
        ..
    } = &error
    else {
        panic!("expected exhausted original equations: {error}");
    };
    assert_eq!(*outer_iterations, 1);
    let residuals = [
        *momentum_relative,
        *continuity_relative,
        *flux_constitutive_relative,
        *global_momentum_relative,
        *maximum_cell_momentum_relative,
    ];
    assert!(residuals.iter().all(|r| r.is_finite() && *r >= 0.0));
    assert!(residuals
        .iter()
        .any(|r| *r > f.policy.physical_relative_tolerance));
    let message = error.to_string();
    assert!(message.contains("global_momentum="));
    assert!(message.contains("maximum_cell_momentum="));
}

#[test]
fn provisional_outlet_backflow_is_solver_only_not_original_physics() {
    let f = case(2, ReferenceCase::Couette, false);
    let p = CanonicalFluidProblem::prepare(f.input()).unwrap();
    let u = vec![[0.1, 0.0, 0.0]; f.topology.cells.len()];
    let pressure = vec![0.0; u.len()];
    let mut q = pressure::h_flux(&p, &u, &vec![[0.0; 3]; u.len()]).unwrap();
    let outlet = p
        .boundaries
        .iter()
        .position(|b| matches!(b, boundary::FaceBoundary::PressureOutlet { .. }))
        .unwrap();
    q[outlet] = -1e-5;
    assert!(momentum::assemble(&p, &u, &pressure, &q, momentum::MomentumMode::SolverTrial).is_ok());
    assert!(matches!(
        momentum::assemble(&p, &u, &pressure, &q, momentum::MomentumMode::Original),
        Err(CanonicalFluidError::Unsupported("pressure outlet backflow"))
    ));
    assert!(matches!(
        recovery::evaluate(&p, &u, &pressure, &q),
        Err(CanonicalFluidError::Unsupported("pressure outlet backflow"))
    ));
    // The ordinary positive-outlet Couette solve/recovery test exercises the
    // converged path without any additional authored inflow boundary authority.
}

// witness: FEA-FLUID-EQUATIONS-001/flux @
//   incompressible_laminar_steady/canonical_tetrahedra/numerical_kernel;
//   boundary: Independent affine skew-face flux and integrated nonzero advection; no general
//   nonlinear field-convergence credit.
#[test]
fn affine_skew_face_velocity_and_nonzero_convection_are_consistent() {
    let f = case(2, ReferenceCase::Convective, false);
    let p = CanonicalFluidProblem::prepare(f.input()).unwrap();
    let u = f
        .topology
        .cells
        .iter()
        .map(|c| reference::velocity(f.reference, c.centroid_m, false))
        .collect::<Vec<_>>();
    let gu = geometry_ops::velocity_gradients(&p, &u).unwrap();
    let q = pressure::h_flux(&p, &u, &vec![[0.0; 3]; u.len()]).unwrap();
    let mut convection = vec![[0.0; 3]; u.len()];
    for (i, face) in f.topology.faces.iter().enumerate() {
        let exact = reference::velocity(f.reference, face.centroid_m, false);
        let s = face.owner_area_vector_m2;
        assert!((q[i] - geometry_ops::dot(exact, s)).abs() < 1e-12);
        // Independent constant advecting field makes q*U affine, so centroid
        // quadrature integrates the analytic nonzero convection exactly. This
        // tests face transport, not a residual-generated manufactured source.
        let advecting_flux = s[0] + 0.2 * s[1];
        let trace = geometry_ops::face_upwind_velocity(&p, i, &u, advecting_flux, &gu).unwrap();
        assert!(
            norm(sub(trace, exact)) < 1e-12,
            "skew face trace {trace:?} vs {exact:?}"
        );
        let action = geometry_ops::scale(trace, advecting_flux);
        convection[face.owner] = geometry_ops::add(convection[face.owner], action);
        if let Some(n) = face.neighbor {
            convection[n] = geometry_ops::sub(convection[n], action);
        }
    }
    for (c, action) in f.topology.cells.iter().zip(convection) {
        assert!(norm(sub(action, [0.2 * c.volume_m3, 0.0, 0.0])) < 1e-12);
    }
}

#[test]
fn unknown_pressure_trace_reconstructs_affine_inlet_gradient_without_fake_neumann() {
    let f = case(2, ReferenceCase::Convective, false);
    let p = CanonicalFluidProblem::prepare(f.input()).unwrap();
    let samples = f
        .topology
        .cells
        .iter()
        .enumerate()
        .map(|(i, c)| {
            reference::pressure(f.reference, c.centroid_m, false)
                - p.pressure_reference_pa.canonical_values()[i]
        })
        .collect::<Vec<_>>();
    let gradients = geometry_ops::pressure_gradient(&p, &samples).unwrap();
    for g in gradients {
        assert!(
            norm(sub(g, [-0.2, 0.0, 0.0])) < 1e-12,
            "independent affine pressure stencil {g:?}"
        );
    }
}

// witness: FEA-FLUID-ACCURACY-001/steady_references @
//   incompressible_laminar_steady/canonical_tetrahedra/numerical_kernel;
//   boundary: Independent affine Couette velocity, wall shear and original balance; not
//   cavity/backward-facing step.
#[test]
fn affine_couette_uses_original_geometry_and_balanced_physical_fields() {
    let fixture = case(2, ReferenceCase::Couette, false);
    let p = CanonicalFluidProblem::prepare(fixture.input()).unwrap();
    let result = p.solve().unwrap();
    let recovered = p.recover(&result).unwrap();
    let error = l2(&fixture, &result);
    eprintln!(
        "canonical Couette cells={} relative L2={} momentum={} continuity={} flux={}",
        result.velocity_m_per_s.canonical_values().len(),
        error,
        recovered.momentum_relative,
        recovered.continuity_relative,
        recovered.flux_constitutive_relative
    );
    assert!(error < 0.25, "independent affine velocity error {error}");
    assert!(norm(recovered.global_momentum_n.canonical_value()) < 1e-6);
    assert!(
        recovered
            .global_boundary_flux_m3_per_s
            .canonical_value()
            .abs()
            < 1e-8
    );
    let top = fixture
        .topology
        .faces
        .iter()
        .enumerate()
        .filter(|(_, f)| f.neighbor.is_none() && (f.centroid_m[1] - 1.0).abs() < 1e-12)
        .map(|(i, _)| recovered.face_forces.viscous_traction_n.canonical_values()[i][0])
        .sum::<f64>();
    let bottom = fixture
        .topology
        .faces
        .iter()
        .enumerate()
        .filter(|(_, f)| f.neighbor.is_none() && f.centroid_m[1].abs() < 1e-12)
        .map(|(i, _)| recovered.face_forces.viscous_traction_n.canonical_values()[i][0])
        .sum::<f64>();
    assert!(
        (top - 1.0).abs() < 0.25 && (bottom + 1.0).abs() < 0.25,
        "independent Couette µ shear: top={top}, bottom={bottom}"
    );
    assert!(recovered
        .cell_momentum_relative
        .iter()
        .all(|r| *r <= policy().physical_relative_tolerance));
    let mut changed = result.clone();
    changed.residual.face_forces.viscous_traction_n = changed
        .residual
        .face_forces
        .viscous_traction_n
        .clone()
        .try_map_canonical_storage(|mut values| {
            values[0][0] += 1.0;
            values
        })
        .unwrap();
    assert!(p.recover(&changed).is_err());
    let mut changed = result.clone();
    changed.velocity_m_per_s = changed
        .velocity_m_per_s
        .clone()
        .try_map_canonical_storage(|mut values| {
            values[0][0] += 0.1;
            values
        })
        .unwrap();
    assert!(p.recover(&changed).is_err());
    let mut changed = result.clone();
    changed.face_volume_flux_m3_per_s = changed
        .face_volume_flux_m3_per_s
        .clone()
        .try_map_canonical_storage(|mut values| {
            values[0] += 0.1;
            values
        })
        .unwrap();
    assert!(p.recover(&changed).is_err());
    let mut changed = result.clone();
    changed.pressure_pa = changed
        .pressure_pa
        .clone()
        .try_map_canonical_storage(|mut values| {
            values[0] += 0.1;
            values
        })
        .unwrap();
    assert!(p.recover(&changed).is_err());
    let mut changed = result.clone();
    changed.residual.momentum_relative = 1.0;
    assert!(p.recover(&changed).is_err());
    let mut changed = result.clone();
    changed.linear_iterations = p.reservation().linear_iterations_upper_bound + 1;
    assert!(p.recover(&changed).is_err());
    let mut changed = result;
    changed.numerical_input_identity =
        runmat_execution::Digest::sha256(b"different original input");
    assert!(p.recover(&changed).is_err());
}

#[test]
fn affine_nonzero_convection_reproduces_independent_velocity_and_pressure() {
    let mut errors = Vec::new();
    let mut pressure_errors = Vec::new();
    for n in [2, 3, 4] {
        let fixture = case(n, ReferenceCase::Convective, false);
        let p = CanonicalFluidProblem::prepare(fixture.input()).unwrap();
        let result = p.solve().unwrap();
        let r = p.recover(&result).unwrap();
        let error = l2(&fixture, &result);
        errors.push(error);
        let pe = pressure_l2(&fixture, &result);
        pressure_errors.push(pe);
        eprintln!("canonical convective MMS n={n} independent physicalpressure L2={pe}");
        eprintln!(
            "canonical convective MMS n={n} velocity L2={error} momentum={} mass={} flux={}",
            r.momentum_relative, r.continuity_relative, r.flux_constitutive_relative
        );
        assert!(r
            .cell_continuity_relative
            .iter()
            .all(|x| *x <= policy().physical_relative_tolerance));
        assert!(r.global_boundary_flux_m3_per_s.canonical_value().abs() < 1e-7);
    }
    // Both analytic fields are affine and are reproduced by this reconstruction.
    // Their remaining error is iterative, not a spatial truncation error: demanding
    // monotonic refinement at the solver floor would not establish an order of
    // convergence. Require substantially tighter accuracy on EVERY mesh instead.
    // The non-affine Fourier duct below retains its strict refinement gates.
    assert!(
        errors.iter().all(|error| *error < 1e-6),
        "independent affine convective velocity error {errors:?}"
    );
    assert!(
        pressure_errors.iter().all(|error| *error < 1e-4),
        "independent affine physical pressure error {pressure_errors:?}"
    );
}

// witness: FEA-FLUID-ACCURACY-001/spatial @
//   incompressible_laminar_steady/canonical_tetrahedra/numerical_kernel;
//   boundary: Controlled-tail non-affine Fourier duct: velocity, flow and wall shear refinement;
//   not pressure/cavity/backward-facing-step envelope.
#[test]
fn rectangular_duct_compares_fourier_reference_with_controlled_tail() {
    assert!(reference::duct_tail(255) < 1e-5);
    let mut errors = Vec::new();
    let mut flow_errors = Vec::new();
    let mut shear_errors = Vec::new();
    for n in [2, 3, 4] {
        let fixture = case(n, ReferenceCase::Duct, false);
        let p = CanonicalFluidProblem::prepare(fixture.input()).unwrap();
        let result = p.solve().unwrap();
        let error = l2(&fixture, &result);
        errors.push(error);
        eprintln!("canonical duct n={n} independent Fourier L2={error}");
        let r = p.recover(&result).unwrap();
        assert!(r.global_boundary_flux_m3_per_s.canonical_value().abs() < 1e-8);
        let flow = fixture
            .topology
            .faces
            .iter()
            .enumerate()
            .filter(|(_, f)| f.neighbor.is_none() && reference::outlet(f.centroid_m, false))
            .map(|(i, _)| result.face_volume_flux_m3_per_s.canonical_values()[i])
            .sum::<f64>();
        let exact_flow = reference::duct_flow(255);
        let flow_error = (flow - exact_flow).abs() / exact_flow;
        flow_errors.push(flow_error);
        let shear_error = wall_shear_error(&fixture, &r);
        shear_errors.push(shear_error);
        eprintln!("canonical duct n={n} independent flowrel={flow_error} shearrel={shear_error}, globalphysicalmomentumrel={}",r.global_momentum_relative);
        let viscous = fixture
            .topology
            .faces
            .iter()
            .enumerate()
            .filter(|(_, f)| f.neighbor.is_none())
            .map(|(i, _)| r.face_forces.viscous_traction_n.canonical_values()[i])
            .fold([0.0; 3], geometry_ops::add);
        let pressure = fixture
            .topology
            .faces
            .iter()
            .enumerate()
            .filter(|(_, f)| f.neighbor.is_none())
            .map(|(i, _)| r.face_forces.pressure_traction_n.canonical_values()[i])
            .fold([0.0; 3], geometry_ops::add);
        let advective = fixture
            .topology
            .faces
            .iter()
            .enumerate()
            .filter(|(_, f)| f.neighbor.is_none())
            .map(|(i, _)| r.face_forces.advective_momentum_n.canonical_values()[i])
            .fold([0.0; 3], geometry_ops::add);
        assert!(
            norm(sub(
                geometry_ops::add(geometry_ops::add(viscous, pressure), [1.0, 0.0, 0.0]),
                advective
            )) < 1e-6,
            "independent original body-force/stress closure"
        );
    }
    assert!(
        errors[1] < errors[0] && errors[2] < errors[1],
        "independent duct refinement {errors:?}"
    );
    assert!(
        errors[2] < 0.35,
        "independent duct absolute error {errors:?}"
    );
    assert!(
        flow_errors[1] < flow_errors[0] && flow_errors[2] < flow_errors[1] && flow_errors[2] < 0.25,
        "independent duct flow refinement {flow_errors:?}"
    );
    assert!(
        shear_errors[1] < shear_errors[0]
            && shear_errors[2] < shear_errors[1]
            && shear_errors[2] < 0.35,
        "independent duct shear refinement {shear_errors:?}"
    );
}

// witness: FEA-FLUID-INPUTS-001/boundaries @
//   incompressible_laminar_steady/canonical_tetrahedra/numerical_kernel; coverage: partial;
//   boundary: A tangential moving wall is supported and normal wall motion is rejected unless an
//   ALE formulation governs it; velocity components, symmetry, periodic and pressure-reference
//   boundaries are other cases.
#[test]
fn tangential_moving_wall_supported_but_normal_motion_requires_ale() {
    let mut f = case(1, ReferenceCase::Couette, false);
    let face = f
        .source
        .boundary_faces
        .iter()
        .find(|face| (face.centroid_m[1] - 1.0).abs() < 1e-12)
        .unwrap();
    let id = face.provenance[0].clone();
    let constraint=f.constraints.iter_mut().find(|c|matches!(&c.kind,runmat_analysis_core::FluidConstraintKind::PrescribedVelocity{target,..} if target.resolved_entities().contains(&id))).unwrap();
    let runmat_analysis_core::FluidConstraintKind::PrescribedVelocity { target, .. } =
        &constraint.kind
    else {
        unreachable!()
    };
    let target = target.clone();
    constraint.kind = runmat_analysis_core::FluidConstraintKind::MovingWall {
        target: target.clone(),
        velocity: runmat_analysis_core::PrescribedValue::Constant(
            [1.0, 0.0, 0.0].map(runmat_analysis_core::Velocity::new),
        ),
    };
    assert!(CanonicalFluidProblem::prepare(f.input()).is_ok());
    let constraint = f
        .constraints
        .iter_mut()
        .find(|c| {
            matches!(
                &c.kind,
                runmat_analysis_core::FluidConstraintKind::MovingWall { .. }
            )
        })
        .unwrap();
    constraint.kind = runmat_analysis_core::FluidConstraintKind::MovingWall {
        target,
        velocity: runmat_analysis_core::PrescribedValue::Constant(
            [0.0, 1e-20, 0.0].map(runmat_analysis_core::Velocity::new),
        ),
    };
    assert!(matches!(
        CanonicalFluidProblem::prepare(f.input()),
        Err(CanonicalFluidError::Unsupported(
            "normal moving-wall motion requires ALE geometry"
        ))
    ));
}

#[test]
fn rigid_rotation_covariance_preserves_velocity_pressure_and_flux() {
    let a = case(2, ReferenceCase::Couette, false);
    let b = case(2, ReferenceCase::Couette, true);
    let pa = CanonicalFluidProblem::prepare(a.input()).unwrap();
    let pb = CanonicalFluidProblem::prepare(b.input()).unwrap();
    let ra = pa.solve().unwrap();
    let rb = pb.solve().unwrap();
    for (u, v) in ra
        .velocity_m_per_s
        .canonical_values()
        .iter()
        .zip(rb.velocity_m_per_s.canonical_values())
    {
        assert!(norm(sub(reference::rotate(*u), *v)) < 1e-6);
    }
    for (p, q) in ra
        .pressure_pa
        .canonical_values()
        .iter()
        .zip(rb.pressure_pa.canonical_values())
    {
        assert!((p - q).abs() < 1e-6);
    }
    for (p, q) in ra
        .face_volume_flux_m3_per_s
        .canonical_values()
        .iter()
        .zip(rb.face_volume_flux_m3_per_s.canonical_values())
    {
        assert!((p - q).abs() < 1e-6);
    }
    assert!(pb.recover(&ra).is_err());
}

#[test]
fn closed_gauge_overlap_nonfinite_and_resource_inputs_fail_closed() {
    let mut f = case(1, ReferenceCase::Couette, false);
    f.operators.clear();
    assert!(CanonicalFluidProblem::prepare(f.input()).is_err());
    let mut f = case(1, ReferenceCase::Couette, false);
    let runmat_analysis_core::FluidConstraintKind::PrescribedVelocity { target, .. } =
        &mut f.constraints[0].kind
    else {
        unreachable!()
    };
    let mut entities = target.resolved_entities().to_vec();
    let mut foreign = entities[0].clone();
    foreign.source_topology_id = "missing-original-face".into();
    entities.push(foreign);
    target.bind_resolved_entities(entities).unwrap();
    assert!(CanonicalFluidProblem::prepare(f.input()).is_err());
    let mut f = case(1, ReferenceCase::Couette, false);
    f.constraints.push(f.constraints[0].clone());
    assert!(CanonicalFluidProblem::prepare(f.input()).is_err());
    let mut f = case(1, ReferenceCase::Couette, false);
    f.policy.maximum_resident_bytes = Some(1);
    assert!(CanonicalFluidProblem::prepare(f.input()).is_err());
    let mut f = case(1, ReferenceCase::Couette, false);
    f.policy.maximum_operator_work = Some(1);
    assert!(CanonicalFluidProblem::prepare(f.input()).is_err());
    let mut f = case(1, ReferenceCase::Couette, false);
    f.material = material();
    f.material.dynamic_viscosity = runmat_types::physical::Quantity::new(f64::NAN);
    assert!(CanonicalFluidProblem::prepare(f.input()).is_err());
}

#[test]
fn weak_and_strong_original_forcing_preserve_relative_accuracy() {
    let mut normalized = Vec::new();
    for amplitude in [1e-6, 1.0, 1e6] {
        let mut f = case(2, ReferenceCase::Couette, false);
        f.material.density = runmat_types::physical::Quantity::new(1.0 / amplitude);
        for c in &mut f.constraints {
            if let runmat_analysis_core::FluidConstraintKind::PrescribedVelocity {
                value: runmat_analysis_core::PrescribedValue::Constant(v),
                ..
            } = &mut c.kind
            {
                for k in v {
                    *k = runmat_types::physical::Quantity::new(k.canonical_value() * (amplitude));
                }
            }
        }
        let p = CanonicalFluidProblem::prepare(f.input()).unwrap();
        let r = p.solve().unwrap();
        p.recover(&r).unwrap();
        normalized.push(
            r.velocity_m_per_s
                .canonical_values()
                .iter()
                .map(|v| v.map(|x| x / amplitude))
                .collect::<Vec<_>>(),
        );
    }
    for v in [&normalized[0], &normalized[2]] {
        for (a, b) in v.iter().zip(&normalized[1]) {
            assert!(norm(sub(*a, *b)) < 1e-6);
        }
    }
}

fn pressure_l2(f: &fixture::Fixture, r: &CanonicalFluidResult) -> f64 {
    let mut error = 0.0;
    let mut exact = 0.0;
    for (c, p) in f
        .topology
        .cells
        .iter()
        .zip(r.pressure_pa.canonical_values())
    {
        let reference = reference::pressure(f.reference, c.centroid_m, f.rotated);
        error += c.volume_m3 * (p - reference).powi(2);
        exact += c.volume_m3 * reference.powi(2);
    }
    (error / exact).sqrt()
}

mod time_block;

// witness: FEA-FLUID-OUTPUTS-001/wall_shear_stress_fields @ incompressible_laminar_steady/tet4/numerical_kernel;
//   coverage: partial; boundary: Couette vorticity and wall shear stress equal the linear profile's;
//   time-averaged WSS and the oscillatory shear index are not covered.
#[test]
fn couette_vorticity_and_wall_shear_match_the_linear_profile() {
    // u = (y, 0, 0) on the unit slab with μ = fixture viscosity: ω = (0, 0, −1),
    // wall shear on the stationary y = 0 wall is μ (1, 0, 0) along the flow.
    let f = case(2, ReferenceCase::Couette, false);
    let p = CanonicalFluidProblem::prepare(f.input()).unwrap();
    let r = p.solve().unwrap();
    for w in r.residual.cell_vorticity_per_s.canonical_values() {
        assert!(norm(sub(*w, [0.0, 0.0, -1.0])) < 1e-6, "vorticity {w:?}");
    }
    let mu = f.material.dynamic_viscosity.canonical_value();
    let mut walls = 0;
    for (i, face) in p.topology.faces.iter().enumerate() {
        if face.neighbor.is_some() || face.centroid_m[1].abs() > 1e-12 {
            continue;
        }
        let n = face
            .owner_area_vector_m2
            .map(|x| x / face.area_m2.canonical_value());
        let t = r.residual.face_forces.viscous_traction_n.canonical_values()[i];
        let tn = geometry_ops::dot(t, n);
        let shear: [f64; 3] =
            std::array::from_fn(|k| -(t[k] - tn * n[k]) / face.area_m2.canonical_value());
        assert!(
            norm(sub(shear, [mu, 0.0, 0.0])) < 1e-6 * mu.max(1.0),
            "{shear:?}"
        );
        walls += 1;
    }
    assert!(walls > 0);
}

#[test]
#[ignore = "convergence measurement, run with --ignored --nocapture"]
fn measure_outer_iterations_against_refinement() {
    for coupling in [
        PressureVelocityCoupling::Coupled,
        PressureVelocityCoupling::Segregated,
    ] {
        for cells in [4, 8] {
            let mut f = case(cells, ReferenceCase::Duct, false);
            f.policy = CanonicalFluidPolicy::default();
            f.policy.pressure_velocity_coupling = coupling;
            f.policy.maximum_outer_iterations = 5000;
            let p = CanonicalFluidProblem::prepare(f.input()).unwrap();
            let start = std::time::Instant::now();
            match p.solve() {
                Ok(r) => eprintln!(
                    "{coupling:?} n={} outer={} linear={} {:?}",
                    p.topology.cells.len(),
                    r.outer_iterations,
                    r.linear_iterations,
                    start.elapsed()
                ),
                Err(e) => eprintln!("{coupling:?} cells={cells} failed {e}"),
            }
        }
    }
}

// witness: FEA-FLUID-SOLVERS-001/coupled_newton_continuation @ incompressible_laminar_steady/tet4/numerical_kernel;
//   coverage: partial; boundary: the coupled Picard solve (FGMRES, SIMPLE block preconditioner)
//   reaches the segregated solver's discrete solution in a fraction of the outer iterations;
//   Newton linearization, pseudo-transient continuation and the Re 1000 cavity are not covered.
#[test]
fn coupled_solve_reaches_the_segregated_fixed_point_in_far_fewer_outer_iterations() {
    let mut f = case(4, ReferenceCase::Duct, false);
    f.policy = CanonicalFluidPolicy::default();
    f.policy.maximum_outer_iterations = 2000;
    let coupled = CanonicalFluidProblem::prepare(f.input())
        .unwrap()
        .solve()
        .unwrap();
    f.policy.pressure_velocity_coupling = PressureVelocityCoupling::Segregated;
    let segregated = CanonicalFluidProblem::prepare(f.input())
        .unwrap()
        .solve()
        .unwrap();
    for (u, v) in coupled
        .velocity_m_per_s
        .canonical_values()
        .iter()
        .zip(segregated.velocity_m_per_s.canonical_values())
    {
        assert!(norm(sub(*u, *v)) < 1e-6 * norm(*v).max(1e-3));
    }
    assert!(
        4 * coupled.outer_iterations < segregated.outer_iterations,
        "coupled {} segregated {}",
        coupled.outer_iterations,
        segregated.outer_iterations
    );
}

#[test]
fn policy_caps_are_null_without_a_cap_and_share_the_wire_bound() {
    use super::CanonicalFluidPolicy;
    let policy = CanonicalFluidPolicy::default();
    let json = serde_json::to_value(policy).unwrap();
    assert!(json["maximum_resident_bytes"].is_null());
    assert!(json["maximum_operator_work"].is_null());
    assert_eq!(
        serde_json::from_value::<CanonicalFluidPolicy>(json.clone()).unwrap(),
        policy
    );
    assert_eq!(policy.resident_byte_budget(), u64::MAX);
    assert_eq!(policy.operator_work_budget(), u64::MAX);
    let max = runmat_types::wire_integer::MAX_WIRE_INTEGER;
    for field in ["maximum_resident_bytes", "maximum_operator_work"] {
        let with = |value: serde_json::Value| {
            let mut json = json.clone();
            json[field] = value;
            serde_json::from_value::<CanonicalFluidPolicy>(json)
        };
        let decoded = with(max.into()).unwrap();
        assert_eq!(
            decoded
                .resident_byte_budget()
                .min(decoded.operator_work_budget()),
            max
        );
        assert!(with((max + 1).into()).is_err());
        assert!(with((-1).into()).is_err());
        assert!(with(1.5.into()).is_err());
        let mut omitted = json.clone();
        omitted.as_object_mut().unwrap().remove(field);
        assert!(serde_json::from_value::<CanonicalFluidPolicy>(omitted).is_err());
        let zero = with(0.into()).unwrap();
        assert!(zero.validate().is_err());
    }
    let mut wide = policy;
    wide.maximum_operator_work = Some(max + 1);
    assert!(serde_json::to_value(wide).is_err());
}
