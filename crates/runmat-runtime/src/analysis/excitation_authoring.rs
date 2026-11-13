use runmat_analysis_core::{
    AnalysisExcitation, AnalysisModel, CoordinateFrame, ElectricCurrent, ExcitationAction,
    ExcitationCase, ExcitationCaseActivation, ExcitationCaseId, ExcitationDistribution,
    ExcitationId, ExcitationKinematics, ExcitationTarget, Force, HeatGenerationDensity,
    PhysicalSignal, Pressure, ProjectionRequirement, RegionTarget, Signal,
    SignalApproximationEvidence, SignalExtrapolation, SignalId, SignalInterpolation, SignalKnot,
    SignalOrdinate, SignalPrecision, SignalProvenance, SignalRepresentation, SignalShape,
    SignalValue, TabulatedSignal, TabulatedSignalDomain,
};

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ConstantExcitation {
    Force([f64; 3]),
    AcousticPressure(f64),
    HeatGeneration(f64),
    ElectricCurrent {
        real_amperes: f64,
        imaginary_amperes: f64,
        frequency_hz: Option<runmat_types::physical::Frequency>,
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
            ExcitationTarget::Region(RegionTarget::boundary(region_id)),
            value.map(Force::new),
            ExcitationDistribution::PreserveResultant,
            PhysicalSignal::Force,
            ExcitationAction::Force,
        ),
        ConstantExcitation::AcousticPressure(value) => append_scalar(
            model,
            &id,
            ExcitationTarget::Region(RegionTarget::boundary(region_id)),
            Pressure::new(value),
            ExcitationDistribution::IntensityPerArea,
            PhysicalSignal::Pressure,
            ExcitationAction::AcousticPressure,
        ),
        ConstantExcitation::HeatGeneration(value) => append_scalar(
            model,
            &id,
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

fn append_scalar<Q>(
    model: &mut AnalysisModel,
    id: &str,
    target: ExcitationTarget,
    value: Q,
    distribution: ExcitationDistribution,
    signal: impl FnOnce(Signal<Q>) -> PhysicalSignal,
    action: impl FnOnce(AnalysisExcitation) -> ExcitationAction,
) {
    append(
        model,
        id,
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
    target: ExcitationTarget,
    value: [Q; 3],
    distribution: ExcitationDistribution,
    signal: impl FnOnce(Signal<Q>) -> PhysicalSignal,
    action: impl FnOnce(AnalysisExcitation) -> ExcitationAction,
) {
    append(
        model,
        id,
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
    target: ExcitationTarget,
    real: Q,
    imaginary: Q,
    frequency_hz: Option<runmat_types::physical::Frequency>,
    distribution: ExcitationDistribution,
    signal: impl FnOnce(Signal<Q>) -> PhysicalSignal,
    action: impl FnOnce(AnalysisExcitation) -> ExcitationAction,
) {
    let representation = match frequency_hz {
        Some(frequency_hz) => SignalRepresentation::Tabulated(TabulatedSignal {
            domain: TabulatedSignalDomain::FrequencyHertz,
            knots: vec![SignalKnot::continuous(
                frequency_hz.canonical_value(),
                SignalOrdinate::Complex {
                    real: SignalValue::Scalar(real),
                    imaginary: SignalValue::Scalar(imaginary),
                },
            )],
            interpolation: SignalInterpolation::Linear,
            extrapolation: SignalExtrapolation::Reject,
        }),
        None => SignalRepresentation::Constant(SignalValue::Scalar(real)),
    };
    append(
        model,
        id,
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
