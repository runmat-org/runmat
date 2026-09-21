use std::collections::BTreeSet;

use runmat_analysis_core::{
    AuthoredNumericExpression, AuthoredNumericExpressionKind, AuthoredParameterizedSignal,
    AuthoredSignalExpression, GeneratorCallable, PhysicalSignal, SignalDomainKind,
    SignalRepresentation,
};

use super::{FeaSignalPresentation, FeaSignalRepresentation};

pub(super) fn present_signal(signal: &PhysicalSignal) -> FeaSignalPresentation {
    FeaSignalPresentation {
        signal_id: signal.signal_id().0.clone(),
        quantity: signal.quantity(),
        domain: signal.domain(),
        shape: signal.shape().clone(),
        representation: representation(signal),
    }
}

pub(super) fn present_parameterized_signal(
    signal: &AuthoredParameterizedSignal,
) -> FeaSignalPresentation {
    let mut input_ids = BTreeSet::new();
    match &signal.value {
        AuthoredSignalExpression::Scalar(expression) => collect_inputs(expression, &mut input_ids),
        AuthoredSignalExpression::Vector3(expressions) => {
            for expression in expressions {
                collect_inputs(expression, &mut input_ids);
            }
        }
    }
    FeaSignalPresentation {
        signal_id: signal.signal_id.0.clone(),
        quantity: signal.quantity,
        domain: SignalDomainKind::StudyParameter,
        shape: signal.shape(),
        representation: FeaSignalRepresentation::Parameterized {
            input_ids: input_ids.into_iter().collect(),
        },
    }
}

fn collect_inputs(expression: &AuthoredNumericExpression, input_ids: &mut BTreeSet<String>) {
    match &expression.kind {
        AuthoredNumericExpressionKind::Literal { .. } => {}
        AuthoredNumericExpressionKind::ModelInput { input } => {
            input_ids.insert(input.input_id.clone());
        }
        AuthoredNumericExpressionKind::Negate { value } => collect_inputs(value, input_ids),
        AuthoredNumericExpressionKind::Binary { left, right, .. } => {
            collect_inputs(left, input_ids);
            collect_inputs(right, input_ids);
        }
    }
}

fn representation(signal: &PhysicalSignal) -> FeaSignalRepresentation {
    if let Some(generator) = signal.generator() {
        let (module_path, source_digest) = match &generator.invocation.callable {
            GeneratorCallable::ImportedFunction(identity) => (
                Some(identity.module_path.clone()),
                Some(identity.source_digest.clone()),
            ),
            GeneratorCallable::ProjectFunction { .. } | GeneratorCallable::BoundFunction { .. } => {
                (None, None)
            }
        };
        return FeaSignalRepresentation::Function {
            name: generator.invocation.callable.display_name().to_string(),
            module_path,
            source_digest,
        };
    }
    macro_rules! classify {
        ($value:expr) => {
            match &$value.representation {
                SignalRepresentation::Constant(_) => FeaSignalRepresentation::Constant,
                SignalRepresentation::Tabulated(_) => FeaSignalRepresentation::Tabulated,
                SignalRepresentation::Waveform(_) => FeaSignalRepresentation::Waveform,
                SignalRepresentation::Artifact(_) => FeaSignalRepresentation::Artifact,
                SignalRepresentation::Generator(_) => unreachable!("generator handled above"),
            }
        };
    }
    match signal {
        PhysicalSignal::Displacement(value) => classify!(value),
        PhysicalSignal::Rotation(value) => classify!(value),
        PhysicalSignal::Velocity(value) => classify!(value),
        PhysicalSignal::Acceleration(value) => classify!(value),
        PhysicalSignal::AngularVelocity(value) => classify!(value),
        PhysicalSignal::AngularAcceleration(value) => classify!(value),
        PhysicalSignal::Force(value) => classify!(value),
        PhysicalSignal::Moment(value) => classify!(value),
        PhysicalSignal::Traction(value) => classify!(value),
        PhysicalSignal::Pressure(value) => classify!(value),
        PhysicalSignal::BodyForceDensity(value) => classify!(value),
        PhysicalSignal::Temperature(value) => classify!(value),
        PhysicalSignal::TemperatureDifference(value) => classify!(value),
        PhysicalSignal::HeatFlux(value) => classify!(value),
        PhysicalSignal::HeatGenerationDensity(value) => classify!(value),
        PhysicalSignal::ElectricPotential(value) => classify!(value),
        PhysicalSignal::ElectricCurrent(value) => classify!(value),
        PhysicalSignal::ElectricCharge(value) => classify!(value),
        PhysicalSignal::ElectricCurrentDensity(value) => classify!(value),
        PhysicalSignal::ElectricChargeDensity(value) => classify!(value),
        PhysicalSignal::MagneticVectorPotential(value) => classify!(value),
        PhysicalSignal::MassFlowRate(value) => classify!(value),
    }
}
