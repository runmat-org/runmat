use std::collections::BTreeSet;

use runmat_analysis_core::{
    AuthoredNumericExpression, AuthoredNumericExpressionKind, AuthoredParameterizedSignal,
    AuthoredSignalExpression, CanonicalQuantity, GeneratorCallable, PhysicalQuantityKind,
    PhysicalSignal, Signal, SignalDomainKind, SignalRepresentation, SignalValue,
};

use super::{FeaSignalConstantPresentation, FeaSignalPresentation, FeaSignalRepresentation};

pub(super) fn present_signal(signal: &PhysicalSignal) -> FeaSignalPresentation {
    FeaSignalPresentation {
        signal_id: signal.signal_id().0.clone(),
        quantity: signal.quantity(),
        domain: signal.domain(),
        shape: signal.shape().clone(),
        representation: representation(signal),
        constant: constant(signal),
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
        constant: None,
    }
}

fn constant(signal: &PhysicalSignal) -> Option<FeaSignalConstantPresentation> {
    macro_rules! values {
        ($signal:expr) => {
            constant_values($signal, canonical_unit(signal.quantity()))
        };
    }
    match signal {
        PhysicalSignal::Displacement(value) => values!(value),
        PhysicalSignal::Rotation(value) => values!(value),
        PhysicalSignal::Velocity(value) => values!(value),
        PhysicalSignal::Acceleration(value) => values!(value),
        PhysicalSignal::AngularVelocity(value) => values!(value),
        PhysicalSignal::AngularAcceleration(value) => values!(value),
        PhysicalSignal::Force(value) => values!(value),
        PhysicalSignal::Moment(value) => values!(value),
        PhysicalSignal::Traction(value) => values!(value),
        PhysicalSignal::Pressure(value) => values!(value),
        PhysicalSignal::BodyForceDensity(value) => values!(value),
        PhysicalSignal::Temperature(value) => values!(value),
        PhysicalSignal::TemperatureDifference(value) => values!(value),
        PhysicalSignal::HeatFlux(value) => values!(value),
        PhysicalSignal::HeatGenerationDensity(value) => values!(value),
        PhysicalSignal::ElectricPotential(value) => values!(value),
        PhysicalSignal::ElectricCurrent(value) => values!(value),
        PhysicalSignal::ElectricCharge(value) => values!(value),
        PhysicalSignal::ElectricCurrentDensity(value) => values!(value),
        PhysicalSignal::ElectricChargeDensity(value) => values!(value),
        PhysicalSignal::MagneticVectorPotential(value) => values!(value),
        PhysicalSignal::MassFlowRate(value) => values!(value),
    }
}

fn constant_values<Q: CanonicalQuantity>(
    signal: &Signal<Q>,
    canonical_unit: &str,
) -> Option<FeaSignalConstantPresentation> {
    let SignalRepresentation::Constant(value) = &signal.representation else {
        return None;
    };
    let values = match value {
        SignalValue::Scalar(value) => vec![value.canonical_value()],
        SignalValue::Vector3(values) => {
            values.iter().map(|value| value.canonical_value()).collect()
        }
        SignalValue::Tensor { values, .. } => {
            values.iter().map(|value| value.canonical_value()).collect()
        }
    };
    Some(FeaSignalConstantPresentation {
        values,
        canonical_unit: canonical_unit.to_string(),
    })
}

const fn canonical_unit(quantity: PhysicalQuantityKind) -> &'static str {
    match quantity {
        PhysicalQuantityKind::Displacement => "m",
        PhysicalQuantityKind::Rotation => "rad",
        PhysicalQuantityKind::Velocity => "m/s",
        PhysicalQuantityKind::Acceleration => "m/s^2",
        PhysicalQuantityKind::AngularVelocity => "rad/s",
        PhysicalQuantityKind::AngularAcceleration => "rad/s^2",
        PhysicalQuantityKind::Force => "N",
        PhysicalQuantityKind::Moment => "N*m",
        PhysicalQuantityKind::Traction | PhysicalQuantityKind::Pressure => "Pa",
        PhysicalQuantityKind::BodyForceDensity => "N/m^3",
        PhysicalQuantityKind::Temperature | PhysicalQuantityKind::TemperatureDifference => "K",
        PhysicalQuantityKind::HeatFlux => "W/m^2",
        PhysicalQuantityKind::HeatGenerationDensity => "W/m^3",
        PhysicalQuantityKind::ElectricPotential => "V",
        PhysicalQuantityKind::ElectricCurrent => "A",
        PhysicalQuantityKind::ElectricCharge => "C",
        PhysicalQuantityKind::ElectricCurrentDensity => "A/m^2",
        PhysicalQuantityKind::ElectricChargeDensity => "C/m^3",
        PhysicalQuantityKind::MagneticVectorPotential => "Wb/m",
        PhysicalQuantityKind::MassFlowRate => "kg/s",
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
