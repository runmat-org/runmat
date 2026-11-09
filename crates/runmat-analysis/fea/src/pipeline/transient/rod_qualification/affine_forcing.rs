//! Spatially exact continuum evidence, separate from the unchanged free-decay spatial gates.
//! rho*u_tt + alpha*rho*u_t = E*(u_xx + beta*u_xxt) + b,
//! u=x*q(t), b=rho*x*(q''+alpha*q'), t(1)=E*(q+beta*q').
use super::*;

mod cold;
mod history;

const OMEGA: f64 = std::f64::consts::TAU;

fn waveform<Q>(amplitude: SignalValue<Q>, cosine: f64, sine: f64) -> SignalRepresentation<Q> {
    SignalRepresentation::Waveform(StandardWaveform {
        kind: StandardWaveformKind::Sine,
        amplitude,
        offset: None,
        // One complete period before initial time avoids a one-sided load-start event.
        start_seconds: runmat_types::physical::Time::new(-std::f64::consts::TAU / OMEGA),
        duration_seconds: (None).map(runmat_types::physical::Time::new),
        frequency_hz: (Some(OMEGA / std::f64::consts::TAU))
            .map(runmat_types::physical::Frequency::new),
        phase_radians: runmat_types::physical::Angle::new(cosine.atan2(sine)),
        duty_cycle: None,
    })
}

fn forced_model(mesh: &SolverMeshArtifact, damped: bool) -> AnalysisModel {
    let mut model = super::model(mesh, damped);
    for (condition, node) in model
        .initial_conditions
        .iter_mut()
        .zip(&mesh.topology.nodes)
    {
        let InitialCondition::Displacement(condition) = condition else {
            panic!("displacement fixture")
        };
        condition.value = InitialValue::UniformVector3(
            [
                AMPLITUDE * node.coordinates_m.canonical_value()[0],
                0.0,
                0.0,
            ]
            .map(Displacement::new),
        );
    }
    replace_fixture_excitations(
        &mut model,
        vec![
            FixtureExcitation {
                id: "affine_tip".into(),
                region_id: "slab_right".into(),
                kind: FixtureExcitationKind::Traction([0.0; 3]),
            },
            FixtureExcitation {
                id: "affine_density".into(),
                region_id: "slab".into(),
                kind: FixtureExcitationKind::BodyForce([0.0; 3]),
            },
        ],
    );
    let alpha = if damped { ALPHA } else { 0.0 };
    let beta = if damped { BETA } else { 0.0 };
    let body_cos = -AMPLITUDE * OMEGA.powi(2); // rho=1
    let body_sin = -alpha * AMPLITUDE * OMEGA;
    let tip_cos = YOUNGS * AMPLITUDE;
    let tip_sin = -YOUNGS * beta * AMPLITUDE * OMEGA;
    for signal in &mut model.signals {
        match signal {
            PhysicalSignal::Traction(signal) => {
                signal.representation = waveform(
                    SignalValue::Vector3([tip_cos.hypot(tip_sin), 0.0, 0.0].map(Traction::new)),
                    tip_cos,
                    tip_sin,
                );
                signal.evidence.sample_count = 0;
            }
            PhysicalSignal::BodyForceDensity(signal) => {
                signal.shape = SignalShape::TopologyField {
                    topology_id: runmat_execution::Digest::from_bytes(
                        *mesh.canonical_digest.bytes(),
                    )
                    .to_string(),
                    location: FieldLocation::Node,
                    component_count: 3,
                };
                signal.representation = waveform(
                    SignalValue::Tensor {
                        shape: vec![mesh.topology.nodes.len(), 3],
                        values: mesh
                            .topology
                            .nodes
                            .iter()
                            .flat_map(|node| {
                                [
                                    body_cos.hypot(body_sin)
                                        * node.coordinates_m.canonical_value()[0],
                                    0.0,
                                    0.0,
                                ]
                                .map(BodyForceDensity::new)
                            })
                            .collect(),
                    },
                    body_cos,
                    body_sin,
                );
                signal.evidence.sample_count = 0;
            }
            _ => panic!("only manufactured physical loads"),
        }
    }
    for action in &mut model.excitations {
        if let ExcitationAction::BodyForce(action) = action {
            action.target = ExcitationTarget::WholeModel;
            action.distribution = ExcitationDistribution::TopologyField;
        }
    }
    model
}

fn reference(time: f64, damped: bool) -> ([f64; 3], [f64; 4]) {
    let q = AMPLITUDE * (OMEGA * time).cos();
    let v = -AMPLITUDE * OMEGA * (OMEGA * time).sin();
    let a = -OMEGA.powi(2) * q;
    let strain = 0.5 * YOUNGS * q * q;
    let kinetic = v * v / 6.0;
    let dissipation = if damped {
        (ALPHA / 3.0 + BETA * YOUNGS)
            * AMPLITUDE.powi(2)
            * OMEGA.powi(2)
            * (time / 2.0 - (2.0 * OMEGA * time).sin() / (4.0 * OMEGA))
    } else {
        0.0
    };
    let work = strain + kinetic + dissipation - 0.5 * YOUNGS * AMPLITUDE.powi(2);
    ([q, v, a], [strain, kinetic, dissipation, work])
}

fn errors(mesh: &SolverMeshArtifact, damped: bool, intervals: usize) -> [f64; 7] {
    let result = run_forced(mesh, damped, intervals);
    errors_from_result(mesh, damped, &result)
}

fn run_forced(mesh: &SolverMeshArtifact, damped: bool, intervals: usize) -> FeaTransientRunResult {
    crate::run_transient_with_options(
        &forced_model(mesh, damped),
        ComputeBackend::Cpu,
        TransientSolveOptions {
            time_step_policy: StructuralTimeStepPolicy::Fixed {
                interval_count: intervals,
            },
            residual_target: 1e-10,
            solver_mesh: Some(mesh.clone()),
            ..TransientSolveOptions::default()
        },
    )
    .unwrap()
}

fn errors_from_result(
    mesh: &SolverMeshArtifact,
    damped: bool,
    result: &FeaTransientRunResult,
) -> [f64; 7] {
    let mut sum = [0.0; 7];
    let energy_scale = 0.5 * YOUNGS * AMPLITUDE.powi(2);
    for (sample, &time) in result.time_points_s.iter().enumerate() {
        let (state, energy) = reference(time, damped);
        let fields = [
            &result.displacement_snapshots[sample],
            &result.velocity_snapshots[sample],
            &result.acceleration_snapshots[sample],
        ];
        for quantity in 0..3 {
            let scale = AMPLITUDE * OMEGA.powi(quantity as i32);
            let values = fields[quantity].as_host_f64().unwrap();
            assert_eq!(values.len(), 3 * mesh.topology.nodes.len());
            for (node, value) in mesh.topology.nodes.iter().zip(values.chunks_exact(3)) {
                assert!(value[1].abs() < 1e-12 && value[2].abs() < 1e-12);
                sum[quantity] += ((value[0]
                    - node.coordinates_m.canonical_value()[0] * state[quantity])
                    / scale)
                    .powi(2)
                    / mesh.topology.nodes.len() as f64;
            }
        }
        let energies = [
            &result.strain_energy_snapshots[sample],
            &result.kinetic_energy_snapshots[sample],
            &result.damping_dissipation_snapshots[sample],
            &result.external_work_snapshots[sample],
        ];
        for (quantity, (field, exact)) in energies.into_iter().zip(energy).enumerate() {
            sum[quantity + 3] += ((field.as_host_f64().unwrap()[0] - exact) / energy_scale).powi(2);
        }
        assert!(result.support_work_snapshots[sample].as_host_f64().unwrap()[0].abs() < 1e-12);
        assert!(
            result.relative_energy_balance_error_snapshots[sample]
                .as_host_f64()
                .unwrap()[0]
                .abs()
                < 1e-6,
            "discrete mechanical work-energy closure at t={time}"
        );
    }
    sum.map(|value| (value / result.time_points_s.len() as f64).sqrt())
}

// witness: FEA-DYNAMICS-RESPONSE-001/time_policy_energy_checkpoints @
//   linear_structural_dynamics/tet4|tet10/numerical_kernel; coverage: partial;
//   check: rod-affine-forcing; boundary: Affine physical forcing qualifies Tet4 and Tet10 temporal fields and work-energy;
//   adaptive time stepping and checkpoints are not covered.
#[test]
fn affine_physical_forcing_qualifies_tet4_tet10_temporal_fields_and_work_energy() {
    for quadratic in [false, true] {
        let mesh = super::mesh(quadratic, 1);
        for damped in [false, true] {
            let errors = [20, 40, 80].map(|intervals| errors(&mesh, damped, intervals));
            eprintln!("affine temporal quadratic={quadratic} damped={damped} errors={errors:?}");
            for quantity in 0..7 {
                // Identically zero undamped dissipation is checked as zero, not a rate.
                if quantity == 5 && !damped {
                    assert!(errors.iter().all(|e| e[quantity] < 1e-12));
                    continue;
                }
                assert!(
                    errors[1][quantity] < errors[0][quantity] / 3.0,
                    "quantity={quantity} errors={errors:?}"
                );
                assert!(
                    errors[2][quantity] < errors[1][quantity] / 3.0,
                    "quantity={quantity} errors={errors:?}"
                );
                assert!(
                    errors[2][quantity] < 0.0002,
                    "quantity={quantity} errors={errors:?}"
                );
            }
        }
    }
}
