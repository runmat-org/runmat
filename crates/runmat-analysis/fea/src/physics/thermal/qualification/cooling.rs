//! Original continuum cooling workload, bound to a closed original temperature field.
use super::*;

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

pub fn manufactured(
    intervals: usize,
    binding: crate::execution::InitialFieldBinding,
) -> ThermalWorkload {
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
        target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("slab_right")),
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
