//! A declared independent axis retains physical type separately from its generic
//! interpolation stencil. Missing named-axis metadata must be resolved, not guessed.
use super::*;
use crate::{AuthoredInputDefinition, AuthoredInputType};
use runmat_types::{
    physical::{DynamicDimension, Quantity, QuantityKind, QuantityStorage, ScalarQuantityStorage},
    PhysicalDimension,
};

fn parameter_axis(
    parameter_id: &str,
    explicit: Option<DynamicDimension>,
    inputs: Option<&[AuthoredInputDefinition]>,
) -> Result<DynamicDimension, String> {
    if parameter_id.trim().is_empty() {
        return Err("study-parameter axis requires an id".into());
    }
    let declared = inputs
        .and_then(|definitions| definitions.iter().find(|d| d.input_id == parameter_id))
        .map(|definition| match definition.value_type {
            AuthoredInputType::Quantity(descriptor) => Ok(descriptor),
            AuthoredInputType::Real | AuthoredInputType::Integer => {
                Ok(PhysicalDimension::DIMENSIONLESS.into())
            }
            _ => Err("study-parameter axis must reference a numeric input".to_string()),
        })
        .transpose()?;
    match (explicit, declared) {
        (Some(a), Some(b)) if !a.is_compatible_with(b) => Err("study-parameter axis units disagree with its input declaration".into()),
        (Some(a), Some(b)) => a.with_semantic_kind(a.kind().or(b.kind())).map_err(|e| e.to_string()),
        (Some(a), None) | (None, Some(a)) => Ok(a),
        (None, None) => Err(format!("study-parameter axis `{parameter_id}` has unresolved units; bind its numeric input declaration or supply explicit canonical coordinate units")),
    }
}
impl TabulatedSignalDomain {
    pub fn axis_descriptor(
        &self,
        inputs: Option<&[AuthoredInputDefinition]>,
    ) -> Result<DynamicDimension, String> {
        match self {
            Self::TimeSeconds => Ok(DynamicDimension::try_new(
                PhysicalDimension::TIME,
                Some(QuantityKind::Time),
            )
            .expect("static axis")),
            Self::FrequencyHertz => Ok(DynamicDimension::try_new(
                PhysicalDimension::FREQUENCY,
                Some(QuantityKind::Frequency),
            )
            .expect("static axis")),
            Self::StudyParameter {
                parameter_id,
                quantity,
            } => parameter_axis(parameter_id, *quantity, inputs),
        }
    }
    fn bind(&mut self, inputs: &[AuthoredInputDefinition]) -> Result<(), String> {
        let descriptor = self.axis_descriptor(Some(inputs))?;
        if let Self::StudyParameter { quantity, .. } = self {
            *quantity = Some(descriptor);
        }
        Ok(())
    }
}
impl GeneratorDomain {
    pub fn axis_descriptor(
        &self,
        inputs: Option<&[AuthoredInputDefinition]>,
    ) -> Result<DynamicDimension, String> {
        match self {
            Self::TimeSeconds => TabulatedSignalDomain::TimeSeconds.axis_descriptor(inputs),
            Self::FrequencyHertz => TabulatedSignalDomain::FrequencyHertz.axis_descriptor(inputs),
            Self::StudyParameter {
                parameter_id,
                quantity,
            } => parameter_axis(parameter_id, *quantity, inputs),
        }
    }
    fn bind(&mut self, inputs: &[AuthoredInputDefinition]) -> Result<(), String> {
        let descriptor = self.axis_descriptor(Some(inputs))?;
        if let Self::StudyParameter { quantity, .. } = self {
            *quantity = Some(descriptor);
        }
        Ok(())
    }
}
impl SignalArtifactDomain {
    pub fn axis_descriptor(
        &self,
        inputs: Option<&[AuthoredInputDefinition]>,
    ) -> Result<Option<DynamicDimension>, String> {
        Ok(match self {
            Self::Constant => None,
            Self::TimeSeconds { .. } => {
                Some(TabulatedSignalDomain::TimeSeconds.axis_descriptor(inputs)?)
            }
            Self::FrequencyHertz { .. } => {
                Some(TabulatedSignalDomain::FrequencyHertz.axis_descriptor(inputs)?)
            }
            Self::StudyParameter {
                parameter_id,
                quantity,
                ..
            } => Some(parameter_axis(parameter_id, *quantity, inputs)?),
        })
    }
    fn bind(&mut self, inputs: &[AuthoredInputDefinition]) -> Result<(), String> {
        let descriptor = self.axis_descriptor(Some(inputs))?;
        if let Self::StudyParameter { quantity, .. } = self {
            *quantity = descriptor;
        }
        Ok(())
    }
}

/// Mathematical knot storage borrowed under the axis's shared quantity header.
/// Ordinates remain separately typed; no per-knot quantity object or copy is needed.
pub struct SignalAxisCoordinates<'a, Q>(&'a [SignalKnot<Q>]);
impl<Q> QuantityStorage for SignalAxisCoordinates<'_, Q> {
    fn all_finite(&self) -> bool {
        self.0.iter().all(|knot| knot.abscissa.is_finite())
    }
}
impl<Q> ScalarQuantityStorage for SignalAxisCoordinates<'_, Q> {
    fn scalar_count(&self) -> usize {
        self.0.len()
    }
    fn scalar_value(&self, index: usize) -> Option<f64> {
        self.0.get(index).map(|knot| knot.abscissa)
    }
}
impl<Q> TabulatedSignal<Q> {
    pub fn coordinate_field(
        &self,
        inputs: Option<&[AuthoredInputDefinition]>,
    ) -> Result<Quantity<DynamicDimension, SignalAxisCoordinates<'_, Q>>, String> {
        Quantity::try_from_descriptor(
            self.domain.axis_descriptor(inputs)?,
            SignalAxisCoordinates(&self.knots),
        )
        .map_err(|e| e.to_string())
    }
}
impl<Q> Signal<Q> {
    fn bind_parameter_axis(&mut self, inputs: &[AuthoredInputDefinition]) -> Result<(), String> {
        match &mut self.representation {
            SignalRepresentation::Tabulated(table) => table.domain.bind(inputs),
            SignalRepresentation::Generator(generator) => generator.domain.bind(inputs),
            SignalRepresentation::Artifact(artifact) => artifact.domain.bind(inputs),
            _ => Ok(()),
        }
    }
}

impl PhysicalSignal {
    pub fn bind_parameter_axis(
        &mut self,
        inputs: &[AuthoredInputDefinition],
    ) -> Result<(), String> {
        match self {
            Self::Displacement(signal) => signal.bind_parameter_axis(inputs),
            Self::Rotation(signal) => signal.bind_parameter_axis(inputs),
            Self::Velocity(signal) => signal.bind_parameter_axis(inputs),
            Self::Acceleration(signal) => signal.bind_parameter_axis(inputs),
            Self::AngularVelocity(signal) => signal.bind_parameter_axis(inputs),
            Self::AngularAcceleration(signal) => signal.bind_parameter_axis(inputs),
            Self::Force(signal) => signal.bind_parameter_axis(inputs),
            Self::Moment(signal) => signal.bind_parameter_axis(inputs),
            Self::Traction(signal) => signal.bind_parameter_axis(inputs),
            Self::Pressure(signal) => signal.bind_parameter_axis(inputs),
            Self::BodyForceDensity(signal) => signal.bind_parameter_axis(inputs),
            Self::Temperature(signal) => signal.bind_parameter_axis(inputs),
            Self::TemperatureDifference(signal) => signal.bind_parameter_axis(inputs),
            Self::HeatFlux(signal) => signal.bind_parameter_axis(inputs),
            Self::HeatGenerationDensity(signal) => signal.bind_parameter_axis(inputs),
            Self::HeatRate(signal) => signal.bind_parameter_axis(inputs),
            Self::ElectricPotential(signal) => signal.bind_parameter_axis(inputs),
            Self::ElectricCurrent(signal) => signal.bind_parameter_axis(inputs),
            Self::ElectricCharge(signal) => signal.bind_parameter_axis(inputs),
            Self::ElectricCurrentDensity(signal) => signal.bind_parameter_axis(inputs),
            Self::ElectricChargeDensity(signal) => signal.bind_parameter_axis(inputs),
            Self::MagneticVectorPotential(signal) => signal.bind_parameter_axis(inputs),
            Self::MassFlowRate(signal) => signal.bind_parameter_axis(inputs),
            Self::VolumeFlowRate(signal) => signal.bind_parameter_axis(inputs),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unresolved_named_axis_binds_to_the_declared_quantity_without_ratio_default() {
        let mut domain = GeneratorDomain::StudyParameter {
            parameter_id: "travel".into(),
            quantity: None,
        };
        assert!(domain.axis_descriptor(None).is_err());
        let inputs = [AuthoredInputDefinition {
            input_id: "travel".into(),
            value_type: AuthoredInputType::Quantity(PhysicalDimension::LENGTH.into()),
            default: None,
            minimum: None,
            maximum: None,
        }];
        domain.bind(&inputs).unwrap();
        assert_eq!(
            domain.axis_descriptor(None).unwrap().dimension(),
            PhysicalDimension::LENGTH
        );
        let incompatible = GeneratorDomain::StudyParameter {
            parameter_id: "travel".into(),
            quantity: Some(PhysicalDimension::DIMENSIONLESS.into()),
        };
        assert!(incompatible.axis_descriptor(Some(&inputs)).is_err());
    }
    #[test]
    fn coordinate_view_keeps_axis_kind_without_copying_generic_knots() {
        let table = TabulatedSignal {
            domain: TabulatedSignalDomain::TimeSeconds,
            knots: vec![
                SignalKnot::continuous(0.0, SignalOrdinate::Real(SignalValue::Scalar(1.0))),
                SignalKnot::continuous(0.5, SignalOrdinate::Real(SignalValue::Scalar(2.0))),
            ],
            interpolation: SignalInterpolation::Linear,
            extrapolation: SignalExtrapolation::Reject,
        };
        let coordinates = table.coordinate_field(None).unwrap();
        assert_eq!(
            coordinates.canonical_values().0.as_ptr(),
            table.knots.as_ptr()
        );
        assert_eq!(coordinates.scalar_count(), 2);
        let point = coordinates.scalar_at(1).unwrap();
        assert_eq!(point.canonical_value(), 0.5);
        assert_eq!(point.semantic_kind(), Some(QuantityKind::Time));
        assert!(point
            .resolve_kind::<runmat_types::physical::kind::Frequency>()
            .is_err());
    }
}
