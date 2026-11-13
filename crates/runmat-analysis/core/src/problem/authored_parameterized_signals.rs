use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::{
    authored_expressions::{AuthoredExpressionError, AuthoredNumericExpression},
    authored_inputs::{AuthoredInputDefinition, AuthoredInputType, AuthoredInputValue},
    coordinate_frame::CoordinateFrame,
    quantity::CanonicalQuantity,
    signal::{
        PhysicalQuantityKind, PhysicalSignal, Signal, SignalApproximationEvidence, SignalId,
        SignalPrecision, SignalProvenance, SignalRepresentation, SignalShape, SignalValue,
    },
};

/// A constant physical signal whose value depends on reusable-model inputs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredParameterizedSignal {
    pub signal_id: SignalId,
    pub quantity: PhysicalQuantityKind,
    pub frame: Option<CoordinateFrame>,
    pub value: AuthoredSignalExpression,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "shape", content = "value")]
pub enum AuthoredSignalExpression {
    Scalar(AuthoredNumericExpression),
    Vector3([AuthoredNumericExpression; 3]),
}

impl AuthoredParameterizedSignal {
    pub fn shape(&self) -> SignalShape {
        match self.value {
            AuthoredSignalExpression::Scalar(_) => SignalShape::Scalar,
            AuthoredSignalExpression::Vector3(_) => SignalShape::Vector3,
        }
    }

    pub fn validate(
        &self,
        inputs: &[AuthoredInputDefinition],
    ) -> Result<(), AuthoredParameterizedSignalError> {
        if self.signal_id.0.trim().is_empty() {
            return Err(AuthoredParameterizedSignalError::EmptyId);
        }
        let expected = quantity_input_type(self.quantity)
            .ok_or(AuthoredParameterizedSignalError::UnsupportedQuantity)?;
        let expressions = match &self.value {
            AuthoredSignalExpression::Scalar(value) => std::slice::from_ref(value),
            AuthoredSignalExpression::Vector3(values) => values.as_slice(),
        };
        for expression in expressions {
            expression
                .validate(inputs)
                .map_err(AuthoredParameterizedSignalError::InvalidExpression)?;
            if !expected.accepts_type(expression.value_type) {
                return Err(AuthoredParameterizedSignalError::WrongExpressionType);
            }
        }
        match (&self.value, &self.frame) {
            (AuthoredSignalExpression::Scalar(_), None)
            | (AuthoredSignalExpression::Vector3(_), Some(_)) => Ok(()),
            _ => Err(AuthoredParameterizedSignalError::InvalidFrame),
        }
    }

    pub fn realize(
        &self,
        inputs: &BTreeMap<String, AuthoredInputValue>,
    ) -> Result<PhysicalSignal, AuthoredParameterizedSignalError> {
        let value = match &self.value {
            AuthoredSignalExpression::Scalar(value) => SignalValue::Scalar(
                value
                    .evaluate_for_quantity(inputs, self.quantity)
                    .map_err(AuthoredParameterizedSignalError::InvalidExpression)?,
            ),
            AuthoredSignalExpression::Vector3(values) => SignalValue::Vector3([
                values[0]
                    .evaluate_for_quantity(inputs, self.quantity)
                    .map_err(AuthoredParameterizedSignalError::InvalidExpression)?,
                values[1]
                    .evaluate_for_quantity(inputs, self.quantity)
                    .map_err(AuthoredParameterizedSignalError::InvalidExpression)?,
                values[2]
                    .evaluate_for_quantity(inputs, self.quantity)
                    .map_err(AuthoredParameterizedSignalError::InvalidExpression)?,
            ]),
        };
        let signal = Signal {
            signal_id: self.signal_id.clone(),
            shape: self.shape(),
            frame: self.frame.clone(),
            representation: SignalRepresentation::Constant(value),
            precision: SignalPrecision::Real64,
            evidence: SignalApproximationEvidence {
                max_absolute_error: Some(0.0),
                max_relative_error: Some(0.0),
                derivative_max_absolute_error: None,
                sample_count: 1,
            },
            provenance: SignalProvenance::Authored,
        };
        Ok(match self.quantity {
            PhysicalQuantityKind::Displacement => physical(signal, PhysicalSignal::Displacement),
            PhysicalQuantityKind::Rotation => physical(signal, PhysicalSignal::Rotation),
            PhysicalQuantityKind::Velocity => physical(signal, PhysicalSignal::Velocity),
            PhysicalQuantityKind::Acceleration => physical(signal, PhysicalSignal::Acceleration),
            PhysicalQuantityKind::AngularVelocity => {
                physical(signal, PhysicalSignal::AngularVelocity)
            }
            PhysicalQuantityKind::AngularAcceleration => {
                physical(signal, PhysicalSignal::AngularAcceleration)
            }
            PhysicalQuantityKind::Force => physical(signal, PhysicalSignal::Force),
            PhysicalQuantityKind::Moment => physical(signal, PhysicalSignal::Moment),
            PhysicalQuantityKind::Traction => physical(signal, PhysicalSignal::Traction),
            PhysicalQuantityKind::Pressure => physical(signal, PhysicalSignal::Pressure),
            PhysicalQuantityKind::BodyForceDensity => {
                physical(signal, PhysicalSignal::BodyForceDensity)
            }
            PhysicalQuantityKind::Temperature => physical(signal, PhysicalSignal::Temperature),
            PhysicalQuantityKind::TemperatureDifference => {
                physical(signal, PhysicalSignal::TemperatureDifference)
            }
            PhysicalQuantityKind::HeatFlux => physical(signal, PhysicalSignal::HeatFlux),
            PhysicalQuantityKind::HeatGenerationDensity => {
                physical(signal, PhysicalSignal::HeatGenerationDensity)
            }
            PhysicalQuantityKind::Power => physical(signal, PhysicalSignal::HeatRate),
            PhysicalQuantityKind::ElectricPotential => {
                physical(signal, PhysicalSignal::ElectricPotential)
            }
            PhysicalQuantityKind::ElectricCurrent => {
                physical(signal, PhysicalSignal::ElectricCurrent)
            }
            PhysicalQuantityKind::ElectricCharge => {
                physical(signal, PhysicalSignal::ElectricCharge)
            }
            PhysicalQuantityKind::ElectricCurrentDensity => {
                physical(signal, PhysicalSignal::ElectricCurrentDensity)
            }
            PhysicalQuantityKind::ElectricChargeDensity => {
                physical(signal, PhysicalSignal::ElectricChargeDensity)
            }
            PhysicalQuantityKind::MagneticVectorPotential => {
                physical(signal, PhysicalSignal::MagneticVectorPotential)
            }
            PhysicalQuantityKind::MassFlowRate => physical(signal, PhysicalSignal::MassFlowRate),
            PhysicalQuantityKind::VolumeFlowRate => {
                physical(signal, PhysicalSignal::VolumeFlowRate)
            }
            _ => return Err(AuthoredParameterizedSignalError::UnsupportedQuantity),
        })
    }
}

fn physical<Q: CanonicalQuantity>(
    signal: Signal<f64>,
    wrap: impl FnOnce(Signal<Q>) -> PhysicalSignal,
) -> PhysicalSignal {
    let value = match signal.representation {
        SignalRepresentation::Constant(SignalValue::Scalar(value)) => {
            SignalValue::Scalar(Q::from_canonical_value(value))
        }
        SignalRepresentation::Constant(SignalValue::Vector3(values)) => {
            SignalValue::Vector3(values.map(Q::from_canonical_value))
        }
        _ => unreachable!("parameterized signals realize only constant values"),
    };
    wrap(Signal {
        signal_id: signal.signal_id,
        shape: signal.shape,
        frame: signal.frame,
        representation: SignalRepresentation::Constant(value),
        precision: signal.precision,
        evidence: signal.evidence,
        provenance: signal.provenance,
    })
}

fn quantity_input_type(quantity: PhysicalQuantityKind) -> Option<AuthoredInputType> {
    if !matches!(
        quantity,
        PhysicalQuantityKind::Displacement
            | PhysicalQuantityKind::Rotation
            | PhysicalQuantityKind::Velocity
            | PhysicalQuantityKind::Acceleration
            | PhysicalQuantityKind::AngularVelocity
            | PhysicalQuantityKind::AngularAcceleration
            | PhysicalQuantityKind::Force
            | PhysicalQuantityKind::Moment
            | PhysicalQuantityKind::Traction
            | PhysicalQuantityKind::Pressure
            | PhysicalQuantityKind::BodyForceDensity
            | PhysicalQuantityKind::Temperature
            | PhysicalQuantityKind::TemperatureDifference
            | PhysicalQuantityKind::HeatFlux
            | PhysicalQuantityKind::HeatGenerationDensity
            | PhysicalQuantityKind::Power
            | PhysicalQuantityKind::ElectricPotential
            | PhysicalQuantityKind::ElectricCurrent
            | PhysicalQuantityKind::ElectricCharge
            | PhysicalQuantityKind::ElectricCurrentDensity
            | PhysicalQuantityKind::ElectricChargeDensity
            | PhysicalQuantityKind::MagneticVectorPotential
            | PhysicalQuantityKind::MassFlowRate
            | PhysicalQuantityKind::VolumeFlowRate
    ) {
        return None;
    }
    Some(AuthoredInputType::Quantity(
        runmat_types::physical::DynamicDimension::try_new(quantity.dimension(), Some(quantity))
            .expect("catalog kind"),
    ))
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AuthoredParameterizedSignalError {
    #[error("parameterized signal id must not be empty")]
    EmptyId,
    #[error("parameterized signal quantity is not available as a declaration-input type")]
    UnsupportedQuantity,
    #[error("parameterized signal expression is invalid: {0}")]
    InvalidExpression(AuthoredExpressionError),
    #[error("parameterized signal expression has the wrong physical quantity")]
    WrongExpressionType,
    #[error("scalar parameterized signals must omit a frame and vectors must declare one")]
    InvalidFrame,
}
