use serde::{Deserialize, Serialize};

use super::{
    coordinate_frame::CoordinateFrame,
    initial_condition::{InitialFieldArtifactRef, InitialValue},
    model_authoring::AuthoredRegionSource,
    quantity::{
        AngularVelocity, Displacement, ElectricPotential, FiniteQuantity, MagneticVectorPotential,
        Pressure, Temperature, Velocity,
    },
    signal::{PhysicalQuantityKind, PhysicalSignal, SignalDomainKind, SignalShape},
};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "target")]
pub enum AuthoredInitialConditionTarget {
    Region(AuthoredRegionSource),
    WholeModel,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum AuthoredInitialConditionKind {
    Displacement(InitialValue<Displacement>),
    Velocity(InitialValue<Velocity>),
    AngularVelocity(InitialValue<AngularVelocity>),
    Temperature(InitialValue<Temperature>),
    ElectricPotential(InitialValue<ElectricPotential>),
    MagneticVectorPotential(InitialValue<MagneticVectorPotential>),
    AcousticPressure(InitialValue<Pressure>),
    FluidPressure(InitialValue<Pressure>),
    FluidVelocity(InitialValue<Velocity>),
}

impl AuthoredInitialConditionKind {
    pub fn field_artifact_ref(&self) -> Option<&InitialFieldArtifactRef> {
        match self {
            Self::Displacement(InitialValue::FieldArtifact(reference))
            | Self::Velocity(InitialValue::FieldArtifact(reference))
            | Self::AngularVelocity(InitialValue::FieldArtifact(reference))
            | Self::Temperature(InitialValue::FieldArtifact(reference))
            | Self::ElectricPotential(InitialValue::FieldArtifact(reference))
            | Self::MagneticVectorPotential(InitialValue::FieldArtifact(reference))
            | Self::AcousticPressure(InitialValue::FieldArtifact(reference))
            | Self::FluidPressure(InitialValue::FieldArtifact(reference))
            | Self::FluidVelocity(InitialValue::FieldArtifact(reference)) => Some(reference),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredInitialCondition {
    pub initial_condition_id: String,
    pub target: AuthoredInitialConditionTarget,
    pub frame: CoordinateFrame,
    pub kind: AuthoredInitialConditionKind,
}

impl AuthoredInitialCondition {
    pub fn validate(
        &self,
        signals: &[PhysicalSignal],
    ) -> Result<(), AuthoredInitialConditionError> {
        if self.initial_condition_id.trim().is_empty() {
            return Err(AuthoredInitialConditionError::EmptyId);
        }
        crate::validate_coordinate_frame(&self.frame)
            .map_err(|_| AuthoredInitialConditionError::InvalidFrame)?;
        match &self.kind {
            AuthoredInitialConditionKind::Displacement(value) => validate_value(
                value,
                InitialShape::Vector3,
                PhysicalQuantityKind::Displacement,
                signals,
            ),
            AuthoredInitialConditionKind::Velocity(value)
            | AuthoredInitialConditionKind::FluidVelocity(value) => validate_value(
                value,
                InitialShape::Vector3,
                PhysicalQuantityKind::Velocity,
                signals,
            ),
            AuthoredInitialConditionKind::AngularVelocity(value) => validate_value(
                value,
                InitialShape::Vector3,
                PhysicalQuantityKind::AngularVelocity,
                signals,
            ),
            AuthoredInitialConditionKind::Temperature(value) => {
                validate_value(
                    value,
                    InitialShape::Scalar,
                    PhysicalQuantityKind::Temperature,
                    signals,
                )?;
                if matches!(value, InitialValue::Uniform(value) if value.canonical_value() < 0.0) {
                    return Err(AuthoredInitialConditionError::BelowAbsoluteZero);
                }
                Ok(())
            }
            AuthoredInitialConditionKind::ElectricPotential(value) => validate_value(
                value,
                InitialShape::Scalar,
                PhysicalQuantityKind::ElectricPotential,
                signals,
            ),
            AuthoredInitialConditionKind::MagneticVectorPotential(value) => validate_value(
                value,
                InitialShape::Vector3,
                PhysicalQuantityKind::MagneticVectorPotential,
                signals,
            ),
            AuthoredInitialConditionKind::AcousticPressure(value)
            | AuthoredInitialConditionKind::FluidPressure(value) => validate_value(
                value,
                InitialShape::Scalar,
                PhysicalQuantityKind::Pressure,
                signals,
            ),
        }
    }

    pub fn region_source(&self) -> Option<&AuthoredRegionSource> {
        match &self.target {
            AuthoredInitialConditionTarget::Region(source) => Some(source),
            AuthoredInitialConditionTarget::WholeModel => None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum InitialShape {
    Scalar,
    Vector3,
}

fn validate_value<Q: FiniteQuantity>(
    value: &InitialValue<Q>,
    shape: InitialShape,
    quantity: PhysicalQuantityKind,
    signals: &[PhysicalSignal],
) -> Result<(), AuthoredInitialConditionError> {
    let valid = match value {
        InitialValue::Uniform(value) => matches!(shape, InitialShape::Scalar) && value.is_finite(),
        InitialValue::UniformVector3(values) => {
            matches!(shape, InitialShape::Vector3) && values.iter().all(FiniteQuantity::is_finite)
        }
        InitialValue::SignalSample {
            signal_id,
            abscissa,
            ..
        } => {
            abscissa.is_finite()
                && abscissa.to_bits() != (-0.0_f64).to_bits()
                && signals.iter().any(|signal| {
                    signal.signal_id() == signal_id
                        && signal.quantity() == quantity
                        && signal.domain() != SignalDomainKind::Frequency
                        && matches!(
                            (shape, signal.shape()),
                            (InitialShape::Scalar, SignalShape::Scalar)
                                | (InitialShape::Vector3, SignalShape::Vector3)
                        )
                })
        }
        InitialValue::FieldArtifact(artifact) => valid_artifact(artifact),
    };
    valid
        .then_some(())
        .ok_or(AuthoredInitialConditionError::InvalidValue)
}

fn valid_artifact(artifact: &InitialFieldArtifactRef) -> bool {
    !artifact.artifact_id.trim().is_empty()
        && !artifact.content_digest.trim().is_empty()
        && !artifact.topology_digest.trim().is_empty()
        && !artifact.field_id.trim().is_empty()
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AuthoredInitialConditionError {
    #[error("initial-condition id must not be empty")]
    EmptyId,
    #[error("initial-condition frame is invalid")]
    InvalidFrame,
    #[error("initial-condition value is incomplete, non-finite, or incompatible")]
    InvalidValue,
    #[error("initial temperature must be at or above zero kelvin")]
    BelowAbsoluteZero,
}
