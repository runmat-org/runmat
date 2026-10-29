use super::*;
use runmat_analysis_core::*;
mod planning;

fn prescribed(id: &str, face: &str, value: PrescribedValue<Temperature>) -> AnalysisConstraint {
    AnalysisConstraint::Thermal(ThermalConstraint {
        constraint_id: id.into(),
        kind: ThermalConstraintKind::Prescribed {
            target: RegionTarget::boundary(
                runmat_meshing_core::fixtures::fixture_boundary_selection(&(face)),
            ),
            value,
        },
    })
}

fn signal<Q>(id: &str, representation: SignalRepresentation<Q>) -> Signal<Q> {
    let sample_count = match &representation {
        SignalRepresentation::Constant(_) => 1,
        SignalRepresentation::Tabulated(table) => table.knots.len(),
        _ => panic!("qualification fixture requires explicit constants or tables"),
    };
    Signal {
        signal_id: SignalId(id.into()),
        shape: SignalShape::Scalar,
        frame: None,
        representation,
        precision: SignalPrecision::Real64,
        evidence: SignalApproximationEvidence {
            max_absolute_error: Some(0.0),
            max_relative_error: Some(0.0),
            derivative_max_absolute_error: None,
            sample_count,
        },
        provenance: SignalProvenance::Authored,
    }
}

fn temperature_history(id: &str, initial: f64) -> PhysicalSignal {
    PhysicalSignal::Temperature(signal(
        id,
        SignalRepresentation::Tabulated(TabulatedSignal {
            domain: TabulatedSignalDomain::TimeSeconds,
            knots: [0.0, 1.0]
                .into_iter()
                .map(|t| {
                    SignalKnot::continuous(
                        t,
                        SignalOrdinate::Real(SignalValue::Scalar(Temperature::new(
                            initial + 2.0 * t,
                        ))),
                    )
                })
                .collect(),
            interpolation: SignalInterpolation::Linear,
            extrapolation: SignalExtrapolation::Reject,
        }),
    ))
}

#[test]
fn solved_nonuniform_initial_field_obeys_independent_heat_equation_and_power() {
    for quadratic in [false, true] {
        let mesh = if quadratic {
            crate::topology::test_meshes::quadratic_slab_mesh(2)
        } else {
            crate::topology::test_meshes::slab_mesh(2)
        };
        let mut steady = workload();
        steady.model.constraints.clear();
        steady.model.boundaries.clear();
        steady.model.excitations.clear();
        steady.model.excitation_cases.clear();
        steady.model.signals.clear();
        steady.model.initial_conditions.clear();
        steady.model.materials[0].mass_density_kg_per_m3 =
            (Some(2.0)).map(runmat_types::physical::MassDensity::new);
        let material = steady.model.materials[0].thermal.as_mut().unwrap();
        material.conductivity_w_per_mk = runmat_types::physical::ThermalConductivity::new(1.0);
        material.specific_heat_j_per_kgk = runmat_types::physical::SpecificHeat::new(3.0);
        steady.model.constraints = vec![
            prescribed(
                "left",
                "slab_left",
                PrescribedValue::Constant(Temperature::new(300.0)),
            ),
            prescribed(
                "right",
                "slab_right",
                PrescribedValue::Constant(Temperature::new(310.0)),
            ),
        ];
        steady.model.sequence = AnalysisSequence::from_steps(
            "steady-source",
            vec![ProcedureStep::thermal_steady("heat_a")],
        );
        steady.procedure = ThermalProcedureControl::Steady;
        steady.residual_target = 1e-12;
        let producer = steady.execute(&mesh).unwrap();
        for (node, actual) in mesh
            .topology
            .nodes
            .iter()
            .zip(producer.temperature_snapshots[0].as_host_f64().unwrap())
        {
            assert!(
                (actual - (300.0 + 10.0 * node.coordinates_m.canonical_value()[0])).abs() < 1e-8
            );
        }
        let binding = solved_temperature_binding(&producer, &mesh);
        for intervals in [2, 4] {
            let mut consumer = steady.clone();
            let mut step = ProcedureStep::thermal_transient("heat_a");
            step.independent_domain = IndependentDomain::TimeSeconds {
                origin: 0.0,
                start: 0.0,
                end: 1.0,
                interval_count: intervals,
            };
            consumer.model.sequence =
                AnalysisSequence::from_steps("transient-consumer", vec![step]);
            consumer.procedure = ThermalProcedureControl::BackwardEuler {
                snapshot_count: intervals + 1,
            };
            consumer.model.initial_conditions =
                vec![InitialCondition::Temperature(InitialConditionValue {
                    initial_condition_id: "solved-profile".into(),
                    target: InitialConditionTarget::WholeModel,
                    frame: CoordinateFrame::Global,
                    value: InitialValue::FieldArtifact(binding.reference().clone()),
                    applies_before_step_id: "heat_a".into(),
                })];
            consumer.initial_fields = vec![binding.clone()];
            consumer.model.signals = vec![
                temperature_history("left", 300.0),
                temperature_history("right", 310.0),
                PhysicalSignal::HeatGenerationDensity(signal(
                    "heat",
                    SignalRepresentation::Constant(SignalValue::Scalar(
                        HeatGenerationDensity::new(12.0),
                    )),
                )),
            ];
            consumer.model.constraints = vec![
                prescribed(
                    "left",
                    "slab_left",
                    PrescribedValue::Signal(SignalId("left".into())),
                ),
                prescribed(
                    "right",
                    "slab_right",
                    PrescribedValue::Signal(SignalId("right".into())),
                ),
            ];
            consumer.model.excitations =
                vec![ExcitationAction::HeatGeneration(AnalysisExcitation {
                    excitation_id: ExcitationId("heat".into()),
                    target: ExcitationTarget::WholeModel,
                    frame: CoordinateFrame::Global,
                    direction: None,
                    distribution: ExcitationDistribution::IntensityPerVolume,
                    kinematics: ExcitationKinematics::FixedFrame,
                    signal_id: SignalId("heat".into()),
                    active_step_ids: vec!["heat_a".into()],
                    combination_group: None,
                    projection: ProjectionRequirement {
                        preserve_resultant: true,
                        preserve_moment: false,
                        relative_tolerance: 1e-12,
                    },
                })];
            consumer.model.excitation_cases = vec![ExcitationCase {
                case_id: ExcitationCaseId("heat".into()),
                action_ids: vec![ExcitationId("heat".into())],
                activation: ExcitationCaseActivation::Always,
            }];
            let consumer = planning::through_authored_product(consumer, &mesh);
            let cold = ThermalWorkload::from_canonical_bytes(&consumer.canonical_bytes().unwrap())
                .unwrap();
            let result = cold.execute(&mesh).unwrap();
            for (t, field) in result
                .time_points_s
                .iter()
                .zip(&result.temperature_snapshots)
            {
                for (node, actual) in mesh.topology.nodes.iter().zip(field.as_host_f64().unwrap()) {
                    assert!(
                        (actual
                            - (300.0 + 10.0 * node.coordinates_m.canonical_value()[0] + 2.0 * t))
                            .abs()
                            < 1e-8,
                        "quadratic={quadratic}, intervals={intervals}, t={t}, actual={actual}"
                    );
                }
            }
            for field in &result.temperature_gradient_snapshots {
                for gradient in field.as_host_f64().unwrap().chunks_exact(3) {
                    assert!(
                        (gradient[0] - 10.0).abs() < 1e-8
                            && gradient[1].abs() < 1e-8
                            && gradient[2].abs() < 1e-8
                    );
                }
            }
            for field in &result.heat_flux_snapshots {
                for flux in field.as_host_f64().unwrap().chunks_exact(3) {
                    assert!(
                        (flux[0] + 10.0).abs() < 1e-8
                            && flux[1].abs() < 1e-8
                            && flux[2].abs() < 1e-8
                    );
                }
            }
            for field in result.energy_storage_rate_snapshots.iter().skip(1) {
                assert!(field
                    .as_host_f64()
                    .unwrap()
                    .iter()
                    .all(|value| (value - 12.0).abs() < 1e-8));
            }
            for balance in &result.energy_balances {
                assert!((balance.storage_rate_w.canonical_value() - 12.0).abs() < 1e-8);
                assert!((balance.applied_power_w.canonical_value() - 12.0).abs() < 1e-8);
                assert!(balance.prescribed_boundary_power_w.canonical_value().abs() < 1e-8);
                assert!(balance.boundary_loss_w.canonical_value().abs() < 1e-8);
                assert!(balance.relative_residual < 1e-10);
            }
        }
    }
}
