use runmat_analysis_core::{
    AnalysisExcitation, AnalysisModel, CoordinateFrame, ElectricCurrent, ElectricPotential,
    ExcitationAction, ExcitationCase, ExcitationCaseActivation, ExcitationCaseId,
    ExcitationDistribution, ExcitationId, ExcitationKinematics, ExcitationTarget, Force,
    HeatGenerationDensity, PhysicalSignal, Pressure, ProjectionRequirement, RegionTarget, Signal,
    SignalApproximationEvidence, SignalAxis, SignalBreakpoint, SignalExtrapolation, SignalId,
    SignalInterpolation, SignalOrdinate, SignalPrecision, SignalProvenance, SignalRepresentation,
    SignalShape, SignalValue, TabulatedSignal, Temperature, Velocity,
};

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ConstantExcitation {
    Force([f64; 3]),
    AcousticPressure(f64),
    HeatGeneration(f64),
    ElectricCurrent {
        real_amperes: f64,
        imaginary_amperes: f64,
        frequency_hz: Option<f64>,
    },
}

pub(crate) fn append_constant_excitation(
    model: &mut AnalysisModel,
    id: impl Into<String>,
    region_id: impl Into<String>,
    value: ConstantExcitation,
) {
    let id = id.into();
    let region_id = region_id.into();
    let case_id = ExcitationCaseId(format!("{id}.case"));
    let mut action_ids = Vec::new();
    match value {
        ConstantExcitation::Force(value) => append_vector(
            model,
            &id,
            case_id.clone(),
            ExcitationTarget::Region(RegionTarget::boundary(region_id)),
            value.map(Force::new),
            ExcitationDistribution::PreserveResultant,
            PhysicalSignal::Force,
            ExcitationAction::Force,
        ),
        ConstantExcitation::AcousticPressure(value) => append_scalar(
            model,
            &id,
            case_id.clone(),
            ExcitationTarget::Region(RegionTarget::boundary(region_id)),
            Pressure::new(value),
            ExcitationDistribution::IntensityPerArea,
            PhysicalSignal::Pressure,
            ExcitationAction::AcousticPressure,
        ),
        ConstantExcitation::HeatGeneration(value) => append_scalar(
            model,
            &id,
            case_id.clone(),
            ExcitationTarget::Region(RegionTarget::volume(region_id)),
            HeatGenerationDensity::new(value),
            ExcitationDistribution::IntensityPerVolume,
            PhysicalSignal::HeatGenerationDensity,
            ExcitationAction::HeatGeneration,
        ),
        ConstantExcitation::ElectricCurrent {
            real_amperes,
            imaginary_amperes,
            frequency_hz,
        } => append_complex_scalar(
            model,
            &id,
            case_id.clone(),
            ExcitationTarget::Region(RegionTarget::volume(region_id)),
            ElectricCurrent::new(real_amperes),
            ElectricCurrent::new(imaginary_amperes),
            frequency_hz,
            ExcitationDistribution::PreserveResultant,
            PhysicalSignal::ElectricCurrent,
            ExcitationAction::ElectricCurrent,
        ),
    }
    action_ids.push(ExcitationId(id));
    model.excitation_cases.push(ExcitationCase {
        case_id,
        action_ids,
        activation: ExcitationCaseActivation::Always,
    });
}

pub(crate) fn append_temperature_history(
    model: &mut AnalysisModel,
    id: impl Into<String>,
    time_seconds: Vec<f64>,
    kelvin: Vec<f64>,
) -> SignalId {
    append_scalar_history(
        model,
        id,
        time_seconds,
        kelvin.into_iter().map(Temperature::new).collect(),
        PhysicalSignal::Temperature,
    )
}

pub(crate) fn append_electric_potential_history(
    model: &mut AnalysisModel,
    id: impl Into<String>,
    time_seconds: Vec<f64>,
    volts: Vec<f64>,
) -> SignalId {
    append_scalar_history(
        model,
        id,
        time_seconds,
        volts.into_iter().map(ElectricPotential::new).collect(),
        PhysicalSignal::ElectricPotential,
    )
}

pub(crate) fn append_velocity_history(
    model: &mut AnalysisModel,
    id: impl Into<String>,
    time_seconds: Vec<f64>,
    meters_per_second: Vec<f64>,
) -> SignalId {
    append_scalar_history(
        model,
        id,
        time_seconds,
        meters_per_second.into_iter().map(Velocity::new).collect(),
        PhysicalSignal::Velocity,
    )
}

fn append_scalar_history<Q>(
    model: &mut AnalysisModel,
    id: impl Into<String>,
    time_seconds: Vec<f64>,
    values: Vec<Q>,
    signal: impl FnOnce(Signal<Q>) -> PhysicalSignal,
) -> SignalId {
    let signal_id = SignalId(id.into());
    let sample_count = values.len();
    model.signals.push(signal(Signal {
        signal_id: signal_id.clone(),
        shape: SignalShape::Scalar,
        frame: None,
        representation: SignalRepresentation::Tabulated(TabulatedSignal {
            axis: SignalAxis::TimeSeconds(time_seconds),
            values: values
                .into_iter()
                .map(|value| SignalOrdinate::Real(SignalValue::Scalar(value)))
                .collect(),
            interpolation: SignalInterpolation::Linear,
            extrapolation: SignalExtrapolation::HoldEndpoint,
            breakpoints: Vec::new(),
        }),
        precision: SignalPrecision::Real64,
        evidence: SignalApproximationEvidence {
            max_absolute_error: None,
            max_relative_error: None,
            derivative_max_absolute_error: None,
            sample_count,
        },
        provenance: SignalProvenance::Authored,
    }));
    signal_id
}

fn append_scalar<Q>(
    model: &mut AnalysisModel,
    id: &str,
    case_id: ExcitationCaseId,
    target: ExcitationTarget,
    value: Q,
    distribution: ExcitationDistribution,
    signal: impl FnOnce(Signal<Q>) -> PhysicalSignal,
    action: impl FnOnce(AnalysisExcitation) -> ExcitationAction,
) {
    append(
        model,
        id,
        case_id,
        target,
        SignalShape::Scalar,
        SignalRepresentation::Constant(SignalValue::Scalar(value)),
        distribution,
        signal,
        action,
        SignalPrecision::Real64,
    );
}

fn append_vector<Q>(
    model: &mut AnalysisModel,
    id: &str,
    case_id: ExcitationCaseId,
    target: ExcitationTarget,
    value: [Q; 3],
    distribution: ExcitationDistribution,
    signal: impl FnOnce(Signal<Q>) -> PhysicalSignal,
    action: impl FnOnce(AnalysisExcitation) -> ExcitationAction,
) {
    append(
        model,
        id,
        case_id,
        target,
        SignalShape::Vector3,
        SignalRepresentation::Constant(SignalValue::Vector3(value)),
        distribution,
        signal,
        action,
        SignalPrecision::Real64,
    );
}

fn append_complex_scalar<Q>(
    model: &mut AnalysisModel,
    id: &str,
    case_id: ExcitationCaseId,
    target: ExcitationTarget,
    real: Q,
    imaginary: Q,
    frequency_hz: Option<f64>,
    distribution: ExcitationDistribution,
    signal: impl FnOnce(Signal<Q>) -> PhysicalSignal,
    action: impl FnOnce(AnalysisExcitation) -> ExcitationAction,
) {
    let representation = match frequency_hz {
        Some(frequency_hz) => SignalRepresentation::Tabulated(TabulatedSignal {
            axis: SignalAxis::FrequencyHertz(vec![frequency_hz]),
            values: vec![SignalOrdinate::Complex {
                real: SignalValue::Scalar(real),
                imaginary: SignalValue::Scalar(imaginary),
            }],
            interpolation: SignalInterpolation::Linear,
            extrapolation: SignalExtrapolation::Reject,
            breakpoints: Vec::<SignalBreakpoint>::new(),
        }),
        None => SignalRepresentation::Constant(SignalValue::Scalar(real)),
    };
    append(
        model,
        id,
        case_id,
        target,
        SignalShape::Scalar,
        representation,
        distribution,
        signal,
        action,
        if frequency_hz.is_some() {
            SignalPrecision::Complex128
        } else {
            SignalPrecision::Real64
        },
    );
}

fn append<Q>(
    model: &mut AnalysisModel,
    id: &str,
    case_id: ExcitationCaseId,
    target: ExcitationTarget,
    shape: SignalShape,
    representation: SignalRepresentation<Q>,
    distribution: ExcitationDistribution,
    signal: impl FnOnce(Signal<Q>) -> PhysicalSignal,
    action: impl FnOnce(AnalysisExcitation) -> ExcitationAction,
    precision: SignalPrecision,
) {
    let signal_id = SignalId(format!("{id}.signal"));
    let frame = matches!(&shape, SignalShape::Vector3).then_some(CoordinateFrame::Global);
    model.signals.push(signal(Signal {
        signal_id: signal_id.clone(),
        shape,
        frame,
        representation,
        precision,
        evidence: SignalApproximationEvidence {
            max_absolute_error: None,
            max_relative_error: None,
            derivative_max_absolute_error: None,
            sample_count: 1,
        },
        provenance: SignalProvenance::Authored,
    }));
    model.excitations.push(action(AnalysisExcitation {
        excitation_id: ExcitationId(id.to_string()),
        case_id,
        target,
        frame: CoordinateFrame::Global,
        direction: None,
        distribution,
        kinematics: ExcitationKinematics::FixedFrame,
        signal_id,
        active_step_ids: Vec::new(),
        combination_group: None,
        projection: ProjectionRequirement {
            preserve_resultant: matches!(distribution, ExcitationDistribution::PreserveResultant),
            preserve_moment: false,
            relative_tolerance: 1.0e-10,
        },
    }));
}
