use super::*;

fn force_signal(id: &str, representation: SignalRepresentation<Force>) -> PhysicalSignal {
    PhysicalSignal::Force(Signal {
        signal_id: SignalId(id.to_string()),
        shape: SignalShape::Scalar,
        frame: None,
        representation,
        precision: SignalPrecision::Real64,
        evidence: SignalApproximationEvidence {
            max_absolute_error: None,
            max_relative_error: None,
            derivative_max_absolute_error: None,
            sample_count: 1,
        },
        provenance: SignalProvenance::Authored,
    })
}

#[test]
fn malformed_signal_domain_and_generator_are_rejected() {
    let mut model = valid_model();
    model.signals = vec![force_signal(
        "force_history",
        SignalRepresentation::Tabulated(TabulatedSignal {
            domain: TabulatedSignalDomain::TimeSeconds,
            knots: vec![
                SignalKnot::continuous(
                    0.0,
                    SignalOrdinate::Real(SignalValue::Scalar(Force::new(1.0))),
                ),
                SignalKnot::continuous(
                    1.0,
                    SignalOrdinate::Real(SignalValue::Scalar(Force::new(2.0))),
                ),
                SignalKnot::continuous(
                    0.5,
                    SignalOrdinate::Real(SignalValue::Scalar(Force::new(3.0))),
                ),
            ],
            interpolation: SignalInterpolation::Linear,
            extrapolation: SignalExtrapolation::Reject,
        }),
    )];
    assert!(matches!(
        validate_model(&model),
        Err(AnalysisValidationError::InvalidSignal { .. })
    ));

    model.signals = vec![force_signal(
        "generated_force",
        SignalRepresentation::Generator(SignalGenerator {
            invocation: GeneratorInvocation {
                callable: GeneratorCallable::ProjectFunction {
                    qualified_name: "anonymousScript".to_string(),
                },
                arguments: Vec::new(),
                effects: GeneratorEffectPolicy::Pure,
            },
            domain: GeneratorDomain::TimeSeconds,
            sampling: SignalSamplingPolicy::Uniform {
                start: 0.0,
                end: 1.0,
                sample_count: 101,
            },
            interpolation: SignalInterpolation::Linear,
            extrapolation: SignalExtrapolation::Reject,
        }),
    )];
    assert!(matches!(
        validate_model(&model),
        Err(AnalysisValidationError::InvalidSignal { .. })
    ));
}

#[test]
fn excitation_signal_quantity_and_distribution_are_checked() {
    let mut model = valid_model();
    model.signals = vec![force_signal(
        "tip_force",
        SignalRepresentation::Constant(SignalValue::Scalar(Force::new(1000.0))),
    )];
    let excitation = AnalysisExcitation {
        excitation_id: ExcitationId("tip_pressure".to_string()),
        target: ExcitationTarget::Region(RegionTarget::boundary("tip")),
        frame: CoordinateFrame::Global,
        direction: None,
        distribution: ExcitationDistribution::IntensityPerArea,
        kinematics: ExcitationKinematics::FixedFrame,
        signal_id: SignalId("tip_force".to_string()),
        active_step_ids: vec!["step_static".to_string()],
        combination_group: None,
        projection: ProjectionRequirement {
            preserve_resultant: true,
            preserve_moment: false,
            relative_tolerance: 1.0e-10,
        },
    };
    model.excitations = vec![ExcitationAction::Pressure(excitation)];
    assert!(matches!(
        validate_model(&model),
        Err(AnalysisValidationError::InvalidExcitation { .. })
    ));
}

#[test]
fn typed_excitation_case_and_initial_condition_are_accepted() {
    let mut model = valid_model();
    model.signals = vec![force_signal(
        "tip_force",
        SignalRepresentation::Constant(SignalValue::Scalar(Force::new(1000.0))),
    )];
    model.excitations = vec![ExcitationAction::Force(AnalysisExcitation {
        excitation_id: ExcitationId("tip_force_action".to_string()),
        target: ExcitationTarget::Region(RegionTarget::boundary("tip")),
        frame: CoordinateFrame::Global,
        direction: Some(Direction {
            vector: [0.0, -1.0, 0.0],
        }),
        distribution: ExcitationDistribution::PreserveResultant,
        kinematics: ExcitationKinematics::FixedFrame,
        signal_id: SignalId("tip_force".to_string()),
        active_step_ids: vec!["step_static".to_string()],
        combination_group: Some("service".to_string()),
        projection: ProjectionRequirement {
            preserve_resultant: true,
            preserve_moment: true,
            relative_tolerance: 1.0e-10,
        },
    })];
    model.excitation_cases = vec![ExcitationCase {
        case_id: ExcitationCaseId("service_case".to_string()),
        action_ids: vec![ExcitationId("tip_force_action".to_string())],
        activation: ExcitationCaseActivation::Always,
    }];
    model.initial_conditions = vec![InitialCondition::Velocity(InitialConditionValue {
        initial_condition_id: "initial_velocity".to_string(),
        target: InitialConditionTarget::Region(RegionTarget::volume("body")),
        frame: CoordinateFrame::Global,
        value: InitialValue::UniformVector3([
            Velocity::new(0.0),
            Velocity::new(0.0),
            Velocity::new(0.0),
        ]),
        applies_before_step_id: "step_static".to_string(),
    })];

    validate_model(&model).expect("typed excitation model should be valid");
    let encoded = serde_json::to_value(&model).expect("model should serialize");
    let decoded: AnalysisModel = serde_json::from_value(encoded).expect("model should deserialize");
    assert_eq!(decoded, model);
}

#[test]
fn excitation_case_membership_is_owned_by_cases_and_exhaustive() {
    let mut model = valid_model();
    model.signals = vec![force_signal(
        "tip_force",
        SignalRepresentation::Constant(SignalValue::Scalar(Force::new(1000.0))),
    )];
    model.excitations = vec![ExcitationAction::Force(AnalysisExcitation {
        excitation_id: ExcitationId("tip_force_action".to_string()),
        target: ExcitationTarget::Region(RegionTarget::boundary("tip")),
        frame: CoordinateFrame::Global,
        direction: Some(Direction {
            vector: [0.0, -1.0, 0.0],
        }),
        distribution: ExcitationDistribution::PreserveResultant,
        kinematics: ExcitationKinematics::FixedFrame,
        signal_id: SignalId("tip_force".to_string()),
        active_step_ids: vec!["step_static".to_string()],
        combination_group: None,
        projection: ProjectionRequirement {
            preserve_resultant: true,
            preserve_moment: false,
            relative_tolerance: 1.0e-10,
        },
    })];
    model.excitation_cases = vec![ExcitationCase {
        case_id: ExcitationCaseId("different_case".to_string()),
        action_ids: vec![ExcitationId("tip_force_action".to_string())],
        activation: ExcitationCaseActivation::Always,
    }];

    validate_model(&model).expect("the case action inventory is authoritative");

    model.excitation_cases.push(ExcitationCase {
        case_id: ExcitationCaseId("second_case".to_string()),
        action_ids: vec![ExcitationId("tip_force_action".to_string())],
        activation: ExcitationCaseActivation::Always,
    });
    validate_model(&model).expect("one physical action may participate in several cases");
}

#[test]
fn scalar_directional_excitation_requires_direction() {
    let mut model = valid_model();
    model.signals = vec![force_signal(
        "tip_force",
        SignalRepresentation::Constant(SignalValue::Scalar(Force::new(1000.0))),
    )];
    model.excitations = vec![ExcitationAction::Force(AnalysisExcitation {
        excitation_id: ExcitationId("tip_force_action".to_string()),
        target: ExcitationTarget::Region(RegionTarget::boundary("tip")),
        frame: CoordinateFrame::Global,
        direction: None,
        distribution: ExcitationDistribution::PreserveResultant,
        kinematics: ExcitationKinematics::FixedFrame,
        signal_id: SignalId("tip_force".to_string()),
        active_step_ids: vec!["step_static".to_string()],
        combination_group: None,
        projection: ProjectionRequirement {
            preserve_resultant: true,
            preserve_moment: false,
            relative_tolerance: 1.0e-10,
        },
    })];

    assert!(matches!(
        validate_model(&model),
        Err(AnalysisValidationError::InvalidExcitation { .. })
    ));
}

#[test]
fn initial_condition_shape_is_physics_typed() {
    let mut model = valid_model();
    model.initial_conditions = vec![InitialCondition::Temperature(InitialConditionValue {
        initial_condition_id: "initial_temperature".to_string(),
        target: InitialConditionTarget::WholeModel,
        frame: CoordinateFrame::Global,
        value: InitialValue::UniformVector3([
            Temperature::new(293.15),
            Temperature::new(293.15),
            Temperature::new(293.15),
        ]),
        applies_before_step_id: "step_static".to_string(),
    })];

    assert!(matches!(
        validate_model(&model),
        Err(AnalysisValidationError::InvalidInitialCondition { .. })
    ));
}

#[test]
fn initial_temperature_rejects_values_below_absolute_zero() {
    let mut model = valid_model();
    model.initial_conditions = vec![InitialCondition::Temperature(InitialConditionValue {
        initial_condition_id: "invalid_initial_temperature".to_string(),
        target: InitialConditionTarget::WholeModel,
        frame: CoordinateFrame::Global,
        value: InitialValue::Uniform(Temperature::new(-0.01)),
        applies_before_step_id: "step_static".to_string(),
    })];

    assert!(matches!(
        validate_model(&model),
        Err(AnalysisValidationError::InvalidInitialCondition { .. })
    ));
}

#[test]
fn invalid_response_spectrum_is_rejected() {
    let mut model = valid_model();
    model.response_spectra = vec![ResponseSpectrum {
        spectrum_id: ResponseSpectrumId("launch_srs".to_string()),
        abscissa: SpectrumAbscissa::FrequencyHertz(runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Frequency,
            _,
        >::new(vec![10.0, 100.0])),
        ordinates: NumericalDataSource::authored(vec![5.0, 20.0], "sha256:spectrum"),
        damping: SpectrumDamping::Ratio(1.2),
        response: SpectrumResponse::AbsoluteAcceleration,
        envelope: SpectrumEnvelope::Maximax,
        interpolation: SpectrumInterpolation::LogLog,
        extrapolation: SpectrumExtrapolation::Reject,
        direction: SpectrumDirection {
            frame: CoordinateFrame::Global,
            vector: [0.0, 0.0, 1.0],
        },
        modal_combination: ModalCombination::CompleteQuadraticCombination,
        directional_combination: DirectionalCombination::Srss,
        truncation: ModalTruncationPolicy {
            minimum_cumulative_mass_ratio: 0.9,
            correction: ModalTruncationCorrection::MissingMass {
                zero_period_acceleration: 20.0,
            },
        },
        rigid_response: RigidResponsePolicy::LindleyYow {
            zero_period_acceleration: 20.0,
        },
        construction: SpectrumConstruction::AuthoredDesign,
        evidence: SpectrumEvidence {
            sampling_adequacy_ratio: 1.0,
            max_interpolation_error: Some(0.01),
        },
    }];
    assert!(matches!(
        validate_model(&model),
        Err(AnalysisValidationError::InvalidResponseSpectrum { .. })
    ));

    model.response_spectra[0].damping = SpectrumDamping::Ratio(0.05);
    model.response_spectra[0].construction = SpectrumConstruction::DerivedFromAccelerationHistory {
        source_signal_id: SignalId("missing_acceleration".to_string()),
        method: ShockResponseSpectrumMethod::ExactPiecewiseLinear,
    };
    assert!(matches!(
        validate_model(&model),
        Err(AnalysisValidationError::InvalidResponseSpectrum { .. })
    ));
}
