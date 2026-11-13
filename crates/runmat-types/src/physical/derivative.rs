//! Checked derivatives of a physical numerator with respect to declared
//! parameter coordinates. A native evaluator parameter is not fabricated as SI.
use super::{
    DynamicDimension, FiniteQuantity, Quantity, QuantityMarker, QuantityStorage, UnitDefinition,
    UnitError,
};
use serde::{ser::SerializeStruct, Deserialize, Deserializer, Serialize, Serializer};

/// Immutable coordinate metadata. A None unit keeps an explicit native/formal
/// axis; it must never be treated as a dimensionless or length coordinate.
pub trait ParameterAxis {
    fn validate_parameter_axis(&self) -> Result<(), String>;
    fn parameter_unit(&self) -> Option<UnitDefinition>;
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DerivativeAxis<A> {
    pub axis: A,
    pub order: u8,
}
impl<A> DerivativeAxis<A> {
    pub fn new(axis: A, order: u8) -> Self {
        Self { axis, order }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParametricDerivative<N: QuantityMarker, S, A, const D: usize = 1> {
    numerator: Quantity<N, S>,
    axes: [DerivativeAxis<A>; D],
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DerivativeError {
    #[error("a derivative requires at least one parameter axis")]
    EmptyAxes,
    #[error("derivative orders must be between 1 and 127")]
    InvalidOrder,
    #[error("invalid derivative parameter axis: {0}")]
    InvalidAxis(String),
    #[error("derivative dimensions exceed the supported exponent range")]
    DimensionOverflow,
    #[error("native parameter dimensions are unresolved")]
    UnresolvedAxis,
    #[error("derivative numerator must contain finite values")]
    NonFiniteNumerator,
    #[error("derivative parameter normalization is not finite and representable")]
    ScaleOverflow,
    #[error("invalid resolved derivative quantity: {0}")]
    InvalidQuantity(UnitError),
}

impl<N: QuantityMarker, S: QuantityStorage, A: ParameterAxis, const D: usize>
    ParametricDerivative<N, S, A, D>
{
    pub fn try_new(
        numerator: Quantity<N, S>,
        axes: [DerivativeAxis<A>; D],
    ) -> Result<Self, DerivativeError> {
        let value = Self { numerator, axes };
        value.validate()?;
        Ok(value)
    }
    fn validate(&self) -> Result<(), DerivativeError> {
        if D == 0 {
            return Err(DerivativeError::EmptyAxes);
        }
        if !self.numerator.canonical_values().all_finite() {
            return Err(DerivativeError::NonFiniteNumerator);
        }
        for term in &self.axes {
            if term.order == 0 || term.order > i8::MAX as u8 {
                return Err(DerivativeError::InvalidOrder);
            }
            term.axis
                .validate_parameter_axis()
                .map_err(DerivativeError::InvalidAxis)?;
        }
        self.resolved_descriptor()?;
        Ok(())
    }
    pub fn is_finite(&self) -> bool {
        self.validate().is_ok()
    }
    pub fn axes(&self) -> &[DerivativeAxis<A>; D] {
        &self.axes
    }

    /// No scalar/array is exposed without keeping the differentiation frames in
    /// the same numerical scope. There is intentionally no numerator getter.
    pub fn with_canonical_storage<R>(
        &self,
        numerical: impl FnOnce(&S, &[DerivativeAxis<A>; D]) -> R,
    ) -> R {
        numerical(self.numerator.canonical_values(), &self.axes)
    }

    /// Transform coefficient storage without changing its physical numerator or
    /// denominator frames (for example a world-space affine vector transform).
    pub fn try_map_canonical_storage<T: QuantityStorage>(
        self,
        map: impl FnOnce(S) -> T,
    ) -> Result<ParametricDerivative<N, T, A, D>, DerivativeError> {
        ParametricDerivative::try_new(
            Quantity::new(map(self.numerator.into_canonical_values())),
            self.axes,
        )
    }

    /// SI exponents and nominal angle powers are complete only when every
    /// denominator is declared. Native axes remain None rather than Length.
    pub fn resolved_descriptor(&self) -> Result<Option<DynamicDimension>, DerivativeError> {
        let mut descriptor = DynamicDimension::try_with_angle(
            N::KIND.dimension(),
            N::KIND.angle_power(),
            Some(N::KIND),
        )
        .map_err(DerivativeError::InvalidQuantity)?;
        let mut unresolved = false;
        for term in &self.axes {
            let Some(unit) = term.axis.parameter_unit() else {
                unresolved = true;
                continue;
            };
            let denominator = DynamicDimension::from_unit(unit)
                .power(term.order as i8)
                .ok_or(DerivativeError::DimensionOverflow)?;
            descriptor = descriptor
                .divide(denominator)
                .ok_or(DerivativeError::DimensionOverflow)?;
        }
        Ok((!unresolved).then_some(descriptor))
    }
}

/// Owned numerical storage that can be normalized without reallocating a
/// duplicate buffer. The enclosing checked derivative validates the result.
pub trait QuantityScaleStorage: QuantityStorage {
    fn scale_canonical_storage(&mut self, factor: f64);
}
impl QuantityScaleStorage for f64 {
    fn scale_canonical_storage(&mut self, factor: f64) {
        *self *= factor;
    }
}
impl<T: QuantityScaleStorage, const N: usize> QuantityScaleStorage for [T; N] {
    fn scale_canonical_storage(&mut self, factor: f64) {
        for value in self {
            value.scale_canonical_storage(factor);
        }
    }
}
impl<T: QuantityScaleStorage> QuantityScaleStorage for Vec<T> {
    fn scale_canonical_storage(&mut self, factor: f64) {
        for value in self {
            value.scale_canonical_storage(factor);
        }
    }
}
impl<T: QuantityScaleStorage> QuantityScaleStorage for Option<T> {
    fn scale_canonical_storage(&mut self, factor: f64) {
        if let Some(value) = self {
            value.scale_canonical_storage(factor);
        }
    }
}

impl<N: QuantityMarker, S: QuantityScaleStorage, A: ParameterAxis, const D: usize>
    ParametricDerivative<N, S, A, D>
{
    /// Convert numerator-per-original-parameter to a canonical derived quantity.
    /// Denominator scale is divided once per derivative order. Affine coordinate
    /// offsets do not enter differentials. Failure never returns a partial value.
    pub fn into_canonical_quantity(self) -> Result<Quantity<DynamicDimension, S>, DerivativeError> {
        self.validate()?;
        let descriptor = self
            .resolved_descriptor()?
            .ok_or(DerivativeError::UnresolvedAxis)?;
        let mut denominator_scale = 1.0;
        for term in &self.axes {
            let unit = term
                .axis
                .parameter_unit()
                .ok_or(DerivativeError::UnresolvedAxis)?;
            denominator_scale *= unit.scale().powi(i32::from(term.order));
            if !denominator_scale.is_finite() || denominator_scale <= 0.0 {
                return Err(DerivativeError::ScaleOverflow);
            }
        }
        let factor = 1.0 / denominator_scale;
        if !factor.is_finite() || factor <= 0.0 {
            return Err(DerivativeError::ScaleOverflow);
        }
        let mut storage = self.numerator.into_canonical_values();
        storage.scale_canonical_storage(factor);
        Quantity::try_from_descriptor(descriptor, storage).map_err(DerivativeError::InvalidQuantity)
    }
}

impl<N: QuantityMarker, S: QuantityStorage, A: ParameterAxis, const D: usize> FiniteQuantity
    for ParametricDerivative<N, S, A, D>
{
    fn is_finite(&self) -> bool {
        self.validate().is_ok()
    }
}
impl<N: QuantityMarker, S: QuantityStorage + Serialize, A: ParameterAxis, const D: usize> Serialize
    for ParametricDerivative<N, S, A, D>
where
    [DerivativeAxis<A>; D]: Serialize,
{
    fn serialize<T: Serializer>(&self, serializer: T) -> Result<T::Ok, T::Error> {
        self.validate().map_err(serde::ser::Error::custom)?;
        let mut fields = serializer.serialize_struct("ParametricDerivative", 2)?;
        fields.serialize_field("numerator", &self.numerator)?;
        fields.serialize_field("axes", &self.axes)?;
        fields.end()
    }
}
impl<
        'de,
        N: QuantityMarker,
        S: QuantityStorage + Deserialize<'de>,
        A: ParameterAxis,
        const D: usize,
    > Deserialize<'de> for ParametricDerivative<N, S, A, D>
where
    [DerivativeAxis<A>; D]: Deserialize<'de>,
{
    fn deserialize<T: Deserializer<'de>>(deserializer: T) -> Result<Self, T::Error> {
        #[derive(Deserialize)]
        #[serde(
            deny_unknown_fields,
            bound(
                deserialize = "Quantity<N,S>: Deserialize<'de>, [DerivativeAxis<A>;D]: Deserialize<'de>"
            )
        )]
        struct Wire<N: QuantityMarker, S, A, const D: usize> {
            numerator: Quantity<N, S>,
            axes: [DerivativeAxis<A>; D],
        }
        let wire = Wire::<N, S, A, D>::deserialize(deserializer)?;
        Self::try_new(wire.numerator, wire.axes).map_err(serde::de::Error::custom)
    }
}

impl<A: ParameterAxis + ?Sized> ParameterAxis for &A {
    fn validate_parameter_axis(&self) -> Result<(), String> {
        (**self).validate_parameter_axis()
    }
    fn parameter_unit(&self) -> Option<UnitDefinition> {
        (**self).parameter_unit()
    }
}
