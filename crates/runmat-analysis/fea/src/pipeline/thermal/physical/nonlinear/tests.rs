//! Independent continuum references, not a lumped-body cooling surrogate.
use super::*;
use crate::execution::ThermalWorkload;
use crate::fixtures::{fixture_model, FixtureId};
use crate::topology::test_meshes::{quadratic_slab_mesh, slab_mesh};
use runmat_analysis_core::*;
use runmat_meshing_core::SolverMeshArtifact;

mod initial;
mod retention;
mod temporal;

const TARGET: f64 = 1.0e-9;
const SIGMA: f64 = 5.670_374_419e-8;
const EMISSIVITY: f64 = 0.7;
const AMBIENT: f64 = 300.0;
const K: f64 = 2.0;

fn mesh(quadratic: bool) -> SolverMeshArtifact {
    if quadratic {
        quadratic_slab_mesh(1)
    } else {
        slab_mesh(2)
    }
}

fn workload() -> ThermalWorkload {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.constraints.clear();
    model.boundaries.clear();
    model.excitations.clear();
    model.excitation_cases.clear();
    model.signals.clear();
    model.initial_conditions.clear();
    model.material_assignments[0].target = PropertyRegionTarget::solid("slab");
    model.materials[0].mass_density_kg_per_m3 = Some(1.0);
    let thermal = model.materials[0].thermal.as_mut().unwrap();
    thermal.conductivity_w_per_mk = K;
    thermal.specific_heat_j_per_kgk = 1.0;
    model.thermal = Some(ThermalDomain {
        reference_temperature: Temperature::new(400.0),
    });
    model.sequence =
        AnalysisSequence::from_steps("radiation", vec![ProcedureStep::thermal_steady("heat")]);
    model.boundaries.push(AnalysisBoundary {
        boundary_id: "gray-radiation".into(),
        target: RegionTarget::boundary("slab_right"),
        operator: BoundaryOperator::ThermalRadiation {
            ambient_temperature: PrescribedValue::Constant(Temperature::new(AMBIENT)),
            emissivity: SurfaceEmissivity::new(EMISSIVITY),
        },
    });
    model.boundaries.push(AnalysisBoundary {
        boundary_id: "insulated-sides".into(),
        target: RegionTarget::boundary("slab_insulated"),
        operator: BoundaryOperator::ThermalHeatFlux {
            value: PrescribedValue::Constant(HeatFlux::new(0.0)),
        },
    });
    ThermalWorkload {
        model,
        initial_fields: Vec::new(),
        step_id: ProcedureStepId("heat".into()),
        backend: ComputeBackend::Cpu,
        formulation_requirement: FormulationRequirement::Experimental,
        procedure: ThermalProcedureControl::Steady,
        residual_target: TARGET,
    }
}

fn left_temperature(input: &mut ThermalWorkload, value: PrescribedValue<Temperature>) {
    input
        .model
        .constraints
        .push(AnalysisConstraint::Thermal(ThermalConstraint {
            constraint_id: "fixed-hot-left".into(),
            kind: ThermalConstraintKind::Prescribed {
                target: RegionTarget::boundary("slab_left"),
                value,
            },
        }));
}

fn assert_acceptance(result: &FeaThermalRunResult, solved: usize) {
    assert_eq!(result.nonlinear_evidence.len(), solved);
    assert_eq!(result.residual_norms.len(), solved);
    assert_eq!(result.energy_balances.len(), solved);
    for evidence in &result.nonlinear_evidence {
        assert!((1..=MAX_CORRECTIONS).contains(&evidence.iteration_count));
        assert!(evidence.backtracking_count <= MAX_CORRECTIONS * MAX_TRIALS);
        assert!(evidence.relative_equilibrium_residual <= TARGET);
        assert!(evidence.relative_increment <= TARGET);
        assert!(evidence.power_scale_w.is_finite() && evidence.power_scale_w >= 0.0);
    }
    assert!(result
        .residual_norms
        .iter()
        .all(|value| value.is_finite() && *value <= TARGET));
    assert!(result
        .energy_balances
        .iter()
        .all(|balance| balance.relative_residual <= TARGET));
    result.validate_complete().unwrap();
}

#[test]
fn fixed_hot_slab_matches_independent_gray_surface_scalar_equilibrium() {
    // Exact continuum T is linear. Independent monotone bisection solves the surface law,
    // not the RunMat discrete operator or Newton routine, on a unit-length/unit-area slab.
    let residual = |surface: f64| {
        K * (400.0 - surface) - EMISSIVITY * SIGMA * (surface.powi(4) - AMBIENT.powi(4))
    };
    let (mut low, mut high) = (AMBIENT, 400.0);
    for _ in 0..64 {
        let middle = 0.5 * (low + high);
        if residual(middle) > 0.0 {
            low = middle;
        } else {
            high = middle;
        }
    }
    let surface = 0.5 * (low + high);
    let power = K * (400.0 - surface);
    for quadratic in [false, true] {
        let mesh = mesh(quadratic);
        let mut input = workload();
        left_temperature(
            &mut input,
            PrescribedValue::Constant(Temperature::new(400.0)),
        );
        let result = input.execute(&mesh).unwrap();
        assert_acceptance(&result, 1);
        retention::verify(&input, &mesh, &result);
        for (node, &temperature) in mesh
            .topology
            .nodes
            .iter()
            .zip(result.temperature_snapshots[0].as_host_f64().unwrap())
        {
            let expected = 400.0 + (surface - 400.0) * node.coordinates_m[0];
            assert!(
                (temperature - expected).abs() / 400.0 < 1.0e-8,
                "quadratic={quadratic} x={} actual={temperature} expected={expected}",
                node.coordinates_m[0]
            );
        }
        let balance = result.energy_balances[0];
        assert!((balance.boundary_loss_w / power - 1.0).abs() < 1.0e-8);
        assert!((balance.prescribed_boundary_power_w / power + 1.0).abs() < 1.0e-8);
        assert_eq!(balance.applied_power_w, 0.0);
        assert_eq!(balance.storage_rate_w, 0.0);
        for gradient in result.temperature_gradient_snapshots[0]
            .as_host_f64()
            .unwrap()
            .chunks_exact(3)
        {
            assert!(
                norm(
                    gradient
                        .iter()
                        .zip([surface - 400.0, 0.0, 0.0])
                        .map(|(actual, expected)| actual - expected)
                ) / (400.0 - surface)
                    < 1.0e-8
            );
        }
        for flux in result.heat_flux_snapshots[0]
            .as_host_f64()
            .unwrap()
            .chunks_exact(3)
        {
            assert!(
                norm(
                    flux.iter()
                        .zip([power, 0.0, 0.0])
                        .map(|(actual, expected)| actual - expected)
                ) / power
                    < 1.0e-8
            );
        }
    }
}

#[test]
fn cold_initial_guess_uses_real_backtracking_and_rejected_runs_publish_nothing() {
    let mesh = mesh(false);
    let mut input = workload();
    input.model.thermal.as_mut().unwrap().reference_temperature = Temperature::new(1.0);
    left_temperature(
        &mut input,
        PrescribedValue::Constant(Temperature::new(400.0)),
    );
    let result = input.execute(&mesh).unwrap();
    assert_acceptance(&result, 1);
    assert!(
        result.nonlinear_evidence[0].backtracking_count > 0,
        "the cold start must exercise damping, not only full Newton steps"
    );
    assert!(result.temperature_snapshots[0]
        .as_host_f64()
        .unwrap()
        .iter()
        .all(|temperature| *temperature >= AMBIENT && *temperature <= 400.0));

    let mut invalid = input.clone();
    invalid
        .model
        .thermal
        .as_mut()
        .unwrap()
        .reference_temperature = Temperature::new(-1.0);
    assert!(invalid.execute(&mesh).is_err());
    let mut gpu = input.clone();
    gpu.backend = ComputeBackend::Gpu;
    assert!(
        gpu.execute(&mesh).is_err(),
        "radiation must reject GPU before any CPU fallback"
    );
    let mut inadmissible = input;
    inadmissible.model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = 1;
    assert!(inadmissible.execute(&mesh).is_err());
}

fn table<Q>(values: Vec<Q>) -> TabulatedSignal<Q> {
    let intervals = values.len() - 1;
    TabulatedSignal {
        domain: TabulatedSignalDomain::TimeSeconds,
        knots: values
            .into_iter()
            .enumerate()
            .map(|(index, value)| {
                SignalKnot::continuous(
                    index as f64 / intervals as f64,
                    SignalOrdinate::Real(SignalValue::Scalar(value)),
                )
            })
            .collect(),
        interpolation: SignalInterpolation::Linear,
        extrapolation: SignalExtrapolation::HoldEndpoint,
    }
}

fn manufactured(intervals: usize, mesh: &SolverMeshArtifact) -> ThermalWorkload {
    // Exact PDE T(x,t)=400-10x-20t², rho*c=1, k=2. Thus generation=-40t,
    // left T prescribed, insulated sides and right incoming flux=q_rad(T(1,t))-20.
    // Nonlinear radiation is NOT replaced by a lumped constitutive/cooling approximation.
    let mut input = workload();
    let mut step = ProcedureStep::thermal_transient("heat");
    step.independent_domain = IndependentDomain::TimeSeconds {
        origin: 0.0,
        start: 0.0,
        end: 1.0,
        interval_count: intervals,
    };
    input.model.sequence = AnalysisSequence::from_steps("radiation-cooling", vec![step]);
    input.procedure = ThermalProcedureControl::BackwardEuler {
        snapshot_count: intervals + 1,
    };
    let binding = initial::solved_linear_profile(mesh);
    input.model.initial_conditions = vec![InitialCondition::Temperature(InitialConditionValue {
        initial_condition_id: "original-solved-initial".into(),
        target: InitialConditionTarget::WholeModel,
        frame: CoordinateFrame::Global,
        value: InitialValue::FieldArtifact(binding.reference().clone()),
        applies_before_step_id: "heat".into(),
    })];
    input.initial_fields = vec![binding];
    left_temperature(
        &mut input,
        PrescribedValue::Signal(SignalId("left-temperature".into())),
    );
    input.model.boundaries.push(AnalysisBoundary {
        boundary_id: "manufactured-incoming-flux".into(),
        target: RegionTarget::boundary("slab_right"),
        operator: BoundaryOperator::ThermalHeatFlux {
            value: PrescribedValue::Signal(SignalId("incoming-flux".into())),
        },
    });
    let times = (0..=intervals)
        .map(|index| index as f64 / intervals as f64)
        .collect::<Vec<_>>();
    let evidence = SignalApproximationEvidence {
        // Nonlinear tables are exact only at the solve knots; no false interpolation
        // accuracy claim is attached to the continuous manufactured reference.
        max_absolute_error: None,
        max_relative_error: None,
        derivative_max_absolute_error: None,
        sample_count: times.len(),
    };
    input.model.signals = vec![
        PhysicalSignal::Temperature(Signal {
            signal_id: SignalId("left-temperature".into()),
            shape: SignalShape::Scalar,
            frame: None,
            representation: SignalRepresentation::Tabulated(table(
                times
                    .iter()
                    .map(|t| Temperature::new(400.0 - 20.0 * t * t))
                    .collect(),
            )),
            precision: SignalPrecision::Real64,
            evidence: evidence.clone(),
            provenance: SignalProvenance::Authored,
        }),
        PhysicalSignal::HeatFlux(Signal {
            signal_id: SignalId("incoming-flux".into()),
            shape: SignalShape::Scalar,
            frame: None,
            representation: SignalRepresentation::Tabulated(table(
                times
                    .iter()
                    .map(|t| {
                        let temperature = 390.0 - 20.0 * t * t;
                        HeatFlux::new(
                            EMISSIVITY * SIGMA * (temperature.powi(4) - AMBIENT.powi(4)) - 20.0,
                        )
                    })
                    .collect(),
            )),
            precision: SignalPrecision::Real64,
            evidence: evidence.clone(),
            provenance: SignalProvenance::Authored,
        }),
        PhysicalSignal::HeatGenerationDensity(Signal {
            signal_id: SignalId("generation".into()),
            shape: SignalShape::Scalar,
            frame: None,
            representation: SignalRepresentation::Tabulated(table(
                times
                    .iter()
                    .map(|t| HeatGenerationDensity::new(-40.0 * t))
                    .collect(),
            )),
            precision: SignalPrecision::Real64,
            evidence,
            provenance: SignalProvenance::Authored,
        }),
    ];
    input.model.excitations = vec![ExcitationAction::HeatGeneration(AnalysisExcitation {
        excitation_id: ExcitationId("generation".into()),
        target: ExcitationTarget::WholeModel,
        frame: CoordinateFrame::Global,
        direction: None,
        distribution: ExcitationDistribution::IntensityPerVolume,
        kinematics: ExcitationKinematics::FixedFrame,
        signal_id: SignalId("generation".into()),
        active_step_ids: vec!["heat".into()],
        combination_group: None,
        projection: ProjectionRequirement {
            preserve_resultant: true,
            preserve_moment: false,
            relative_tolerance: 1.0e-12,
        },
    })];
    input.model.excitation_cases = vec![ExcitationCase {
        case_id: ExcitationCaseId("generation".into()),
        action_ids: vec![ExcitationId("generation".into())],
        activation: ExcitationCaseActivation::Always,
    }];
    // Authenticate the actual retained producer field through the original workload's codec.
    ThermalWorkload::from_canonical_bytes(&input.canonical_bytes().unwrap()).unwrap()
}
