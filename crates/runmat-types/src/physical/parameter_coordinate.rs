//! Formal evaluator coordinates retain axis identity even when no physical unit
//! is known. A native coordinate is never silently interpreted as a ratio.
use super::{DerivativeAxis, DerivativeError, DynamicDimension, ParameterAxis, Quantity};
use serde::{ser::SerializeStruct, Deserialize, Deserializer, Serialize, Serializer};

#[derive(Debug, Clone, PartialEq)]
pub struct ParameterCoordinate<A> {
    value: f64,
    axis: A,
}
impl<A: ParameterAxis> ParameterCoordinate<A> {
    pub fn try_new(value: f64, axis: A) -> Result<Self, DerivativeError> {
        axis.validate_parameter_axis()
            .map_err(DerivativeError::InvalidAxis)?;
        if !value.is_finite() {
            return Err(DerivativeError::NonFiniteNumerator);
        }
        // Formal parameter coordinates have an origin of their own, not an
        // absolute-temperature origin. Affine physical fields use SourceMeasure.
        if axis
            .parameter_unit()
            .is_some_and(|unit| unit.offset() != 0.0)
        {
            return Err(DerivativeError::InvalidQuantity(
                super::UnitError::InvalidAffineUnit,
            ));
        }
        Ok(Self { value, axis })
    }
    pub fn axis(&self) -> &A {
        &self.axis
    }
    pub fn with_native_coordinate<T>(&self, evaluate: impl FnOnce(f64, &A) -> T) -> T {
        evaluate(self.value, &self.axis)
    }
    pub fn resolved_quantity(&self) -> Result<Option<Quantity<DynamicDimension>>, DerivativeError> {
        let Some(unit) = self.axis.parameter_unit() else {
            return Ok(None);
        };
        let value = unit
            .to_canonical(self.value)
            .map_err(DerivativeError::InvalidQuantity)?;
        Quantity::try_from_descriptor(DynamicDimension::from_unit(unit), value)
            .map(Some)
            .map_err(DerivativeError::InvalidQuantity)
    }
}
impl<A: ParameterAxis + Serialize> Serialize for ParameterCoordinate<A> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        ParameterCoordinate::try_new(self.value, &self.axis).map_err(serde::ser::Error::custom)?;
        let mut fields = serializer.serialize_struct("ParameterCoordinate", 2)?;
        fields.serialize_field("value", &self.value)?;
        fields.serialize_field("axis", &self.axis)?;
        fields.end()
    }
}
impl<'de, A: ParameterAxis + Deserialize<'de>> Deserialize<'de> for ParameterCoordinate<A> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Wire<A> {
            value: f64,
            axis: A,
        }
        let wire = Wire::<A>::deserialize(deserializer)?;
        Self::try_new(wire.value, wire.axis).map_err(serde::de::Error::custom)
    }
}

/// A derivative whose numerator is itself a possibly native parameter axis,
/// e.g. one component of a pcurve's mixed (angle, length) UV coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct ParameterCoordinateDerivative<A, const D: usize = 1> {
    coefficient: f64,
    numerator_axis: A,
    axes: [DerivativeAxis<A>; D],
}
impl<A: ParameterAxis, const D: usize> ParameterCoordinateDerivative<A, D> {
    pub fn try_new(
        coefficient: f64,
        numerator_axis: A,
        axes: [DerivativeAxis<A>; D],
    ) -> Result<Self, DerivativeError> {
        let value = Self {
            coefficient,
            numerator_axis,
            axes,
        };
        value.validate()?;
        Ok(value)
    }
    fn validate(&self) -> Result<(), DerivativeError> {
        if D == 0 {
            return Err(DerivativeError::EmptyAxes);
        }
        if !self.coefficient.is_finite() {
            return Err(DerivativeError::NonFiniteNumerator);
        }
        self.numerator_axis
            .validate_parameter_axis()
            .map_err(DerivativeError::InvalidAxis)?;
        for term in &self.axes {
            if term.order == 0 || term.order > 127 {
                return Err(DerivativeError::InvalidOrder);
            }
            term.axis
                .validate_parameter_axis()
                .map_err(DerivativeError::InvalidAxis)?;
        }
        Ok(())
    }
    pub fn numerator_axis(&self) -> &A {
        &self.numerator_axis
    }
    pub fn axes(&self) -> &[DerivativeAxis<A>; D] {
        &self.axes
    }
    pub fn with_native_coefficient<T>(
        &self,
        evaluate: impl FnOnce(f64, &A, &[DerivativeAxis<A>; D]) -> T,
    ) -> T {
        evaluate(self.coefficient, &self.numerator_axis, &self.axes)
    }
    pub fn resolved_quantity(&self) -> Result<Option<Quantity<DynamicDimension>>, DerivativeError> {
        self.validate()?;
        let Some(numerator) = self.numerator_axis.parameter_unit() else {
            return Ok(None);
        };
        let mut descriptor = DynamicDimension::from_unit(numerator);
        let mut scale = numerator.scale();
        for term in &self.axes {
            let Some(unit) = term.axis.parameter_unit() else {
                return Ok(None);
            };
            for _ in 0..term.order {
                descriptor = descriptor
                    .divide(DynamicDimension::from_unit(unit))
                    .ok_or(DerivativeError::DimensionOverflow)?;
            }
            // Offsets cancel in differentials; only the declared axis scale acts.
            scale /= unit.scale().powi(i32::from(term.order));
            if !scale.is_finite() || scale <= 0.0 {
                return Err(DerivativeError::ScaleOverflow);
            }
        }
        Quantity::try_from_descriptor(descriptor, self.coefficient * scale)
            .map(Some)
            .map_err(DerivativeError::InvalidQuantity)
    }
}
impl<A: ParameterAxis + Serialize, const D: usize> Serialize for ParameterCoordinateDerivative<A, D>
where
    [DerivativeAxis<A>; D]: Serialize,
{
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.validate().map_err(serde::ser::Error::custom)?;
        let mut fields = serializer.serialize_struct("ParameterCoordinateDerivative", 3)?;
        fields.serialize_field("coefficient", &self.coefficient)?;
        fields.serialize_field("numerator_axis", &self.numerator_axis)?;
        fields.serialize_field("axes", &self.axes)?;
        fields.end()
    }
}
impl<'de, A: ParameterAxis + Deserialize<'de>, const D: usize> Deserialize<'de>
    for ParameterCoordinateDerivative<A, D>
where
    [DerivativeAxis<A>; D]: Deserialize<'de>,
{
    fn deserialize<S: Deserializer<'de>>(deserializer: S) -> Result<Self, S::Error> {
        #[derive(Deserialize)]
        #[serde(
            deny_unknown_fields,
            bound(deserialize = "A:Deserialize<'de>, [DerivativeAxis<A>;D]:Deserialize<'de>")
        )]
        struct Wire<A, const D: usize> {
            coefficient: f64,
            numerator_axis: A,
            axes: [DerivativeAxis<A>; D],
        }
        let wire = Wire::<A, D>::deserialize(deserializer)?;
        Self::try_new(wire.coefficient, wire.numerator_axis, wire.axes)
            .map_err(serde::de::Error::custom)
    }
}
