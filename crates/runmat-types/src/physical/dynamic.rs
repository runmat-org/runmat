//! Dynamic descriptors preserve SI dimensions, nominal angle powers, and only
//! semantic kinds actually known. Storage is the same Quantity<K, S> family.
use super::{
    catalog::sealed, Quantity, QuantityKind, QuantityMarker, QuantitySpec, QuantityStorage,
    UnitDefinition, UnitError,
};
use crate::PhysicalDimension;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
fn angle_is_zero(value: &i8) -> bool {
    *value == 0
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[cfg_attr(feature = "tool-schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct DynamicDimension {
    dimension: PhysicalDimension,
    #[serde(default, skip_serializing_if = "angle_is_zero")]
    angle_power: i8,
    #[serde(skip_serializing_if = "Option::is_none")]
    semantic_kind: Option<QuantityKind>,
}
impl DynamicDimension {
    pub fn try_new(
        dimension: PhysicalDimension,
        semantic_kind: Option<QuantityKind>,
    ) -> Result<Self, UnitError> {
        Self::try_with_angle(
            dimension,
            semantic_kind.map_or(0, QuantityKind::angle_power),
            semantic_kind,
        )
    }
    pub fn try_with_angle(
        dimension: PhysicalDimension,
        angle_power: i8,
        semantic_kind: Option<QuantityKind>,
    ) -> Result<Self, UnitError> {
        if semantic_kind.is_some_and(|kind| kind.dimension() != dimension) {
            return Err(UnitError::DimensionMismatch);
        }
        if semantic_kind.is_some_and(|kind| kind.angle_power() != angle_power) {
            return Err(UnitError::AngleMismatch);
        }
        Ok(Self {
            dimension,
            angle_power,
            semantic_kind,
        })
    }
    /// Compatibility is explicit and does not change descriptor equality.
    /// An absent kind can be resolved by a declared field or operation context.
    pub fn is_compatible_with(self, other: Self) -> bool {
        self.dimension == other.dimension
            && self.angle_power == other.angle_power
            && !matches!((self.semantic_kind, other.semantic_kind), (Some(a),Some(b)) if a != b)
    }
    pub const fn from_unit(unit: UnitDefinition) -> Self {
        Self {
            dimension: unit.dimension(),
            angle_power: unit.angle_power(),
            semantic_kind: None,
        }
    }
    /// A source type declares dimensions/nominal units; an expected value role is
    /// resolved at its field/operation boundary rather than guessed by dimension.
    pub const fn quantity_type(kind: QuantityKind) -> Self {
        Self {
            dimension: kind.dimension(),
            angle_power: kind.angle_power(),
            semantic_kind: None,
        }
    }
    pub fn with_semantic_kind(self, kind: Option<QuantityKind>) -> Result<Self, UnitError> {
        Self::try_with_angle(self.dimension, self.angle_power, kind)
    }
    pub const fn without_semantic_kind(self) -> Self {
        Self {
            semantic_kind: None,
            ..self
        }
    }
    pub const fn dimension(self) -> PhysicalDimension {
        self.dimension
    }
    pub const fn angle_power(self) -> i8 {
        self.angle_power
    }
    pub const fn kind(self) -> Option<QuantityKind> {
        self.semantic_kind
    }
    pub fn multiply(self, other: Self) -> Option<Self> {
        Some(Self {
            dimension: self.dimension.multiply(other.dimension)?,
            angle_power: self.angle_power.checked_add(other.angle_power)?,
            semantic_kind: None,
        })
    }
    pub fn divide(self, other: Self) -> Option<Self> {
        Some(Self {
            dimension: self.dimension.divide(other.dimension)?,
            angle_power: self.angle_power.checked_sub(other.angle_power)?,
            semantic_kind: None,
        })
    }
    pub fn power(self, power: i8) -> Option<Self> {
        Some(Self {
            dimension: self.dimension.power(power)?,
            angle_power: self.angle_power.checked_mul(power)?,
            semantic_kind: None,
        })
    }
}
impl From<PhysicalDimension> for DynamicDimension {
    fn from(dimension: PhysicalDimension) -> Self {
        Self {
            dimension,
            angle_power: 0,
            semantic_kind: None,
        }
    }
}
impl sealed::Spec for DynamicDimension {}
impl QuantitySpec for DynamicDimension {
    fn physical_dimension(self) -> PhysicalDimension {
        self.dimension
    }
    fn semantic_kind(self) -> Option<QuantityKind> {
        self.semantic_kind
    }
    fn nominal_angle_power(self) -> i8 {
        self.angle_power
    }
}
impl<'de> Deserialize<'de> for DynamicDimension {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Wire {
            dimension: PhysicalDimension,
            #[serde(default)]
            angle_power: i8,
            #[serde(default)]
            semantic_kind: Option<QuantityKind>,
        }
        let w = Wire::deserialize(d)?;
        Self::try_with_angle(w.dimension, w.angle_power, w.semantic_kind)
            .map_err(serde::de::Error::custom)
    }
}

pub type DimensionedScalar = Quantity<DynamicDimension>;
impl<S: QuantityStorage> Quantity<DynamicDimension, S> {
    pub fn try_from_dimension(
        dimension: PhysicalDimension,
        semantic_kind: Option<QuantityKind>,
        value: S,
    ) -> Result<Self, UnitError> {
        Self::try_from_descriptor(DynamicDimension::try_new(dimension, semantic_kind)?, value)
    }
    pub fn try_from_descriptor(spec: DynamicDimension, value: S) -> Result<Self, UnitError> {
        if !value.all_finite() {
            return Err(UnitError::NonFiniteValue);
        }
        Ok(Self { spec, value })
    }
    pub fn require_kind(&self, kind: QuantityKind) -> Result<(), UnitError> {
        if self.dimension() != kind.dimension() {
            return Err(UnitError::DimensionMismatch);
        }
        if self.angle_power() != kind.angle_power() {
            return Err(UnitError::AngleMismatch);
        }
        if self.semantic_kind().is_some_and(|actual| actual != kind) {
            return Err(UnitError::KindMismatch);
        }
        Ok(())
    }
    pub fn resolve_kind<K: QuantityMarker>(self) -> Result<Quantity<K, S>, UnitError> {
        self.require_kind(K::KIND)?;
        Ok(Quantity::new(self.value))
    }
}

#[doc(hidden)]
#[derive(Serialize, Deserialize)]
#[cfg_attr(feature = "tool-schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct DimensionedQuantityWire<S> {
    pub dimension: PhysicalDimension,
    #[serde(default, skip_serializing_if = "angle_is_zero")]
    pub angle_power: i8,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub semantic_kind: Option<QuantityKind>,
    pub value_si: S,
}
impl<S: Serialize + QuantityStorage> Serialize for Quantity<DynamicDimension, S> {
    fn serialize<T: Serializer>(&self, s: T) -> Result<T::Ok, T::Error> {
        if !self.canonical_values().all_finite() {
            return Err(serde::ser::Error::custom(UnitError::NonFiniteValue));
        }
        DimensionedQuantityWire {
            dimension: self.dimension(),
            angle_power: self.angle_power(),
            semantic_kind: self.semantic_kind(),
            value_si: self.canonical_values(),
        }
        .serialize(s)
    }
}
impl<'de, S: Deserialize<'de> + QuantityStorage> Deserialize<'de>
    for Quantity<DynamicDimension, S>
{
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let w = DimensionedQuantityWire::<S>::deserialize(d)?;
        let spec = DynamicDimension::try_with_angle(w.dimension, w.angle_power, w.semantic_kind)
            .map_err(serde::de::Error::custom)?;
        Self::try_from_descriptor(spec, w.value_si).map_err(serde::de::Error::custom)
    }
}
#[cfg(feature = "tool-schema")]
impl<S: schemars::JsonSchema> schemars::JsonSchema for Quantity<DynamicDimension, S> {
    fn schema_name() -> String {
        format!("DimensionedQuantity{}", S::schema_name())
    }
    fn json_schema(g: &mut schemars::gen::SchemaGenerator) -> schemars::schema::Schema {
        <DimensionedQuantityWire<S> as schemars::JsonSchema>::json_schema(g)
    }
}

impl<K: super::QuantitySpec, S: super::QuantityStorage> Quantity<K, S> {
    /// Erasure retains every known kind and nominal angle; it never guesses a
    /// semantic role from the seven SI exponents.
    pub fn try_into_dynamic(self) -> Result<Quantity<DynamicDimension, S>, UnitError> {
        let descriptor = DynamicDimension::try_with_angle(
            self.dimension(),
            self.angle_power(),
            self.semantic_kind(),
        )?;
        Quantity::try_from_descriptor(descriptor, self.into_canonical_values())
    }
}
