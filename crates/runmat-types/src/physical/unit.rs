use super::{Quantity, QuantityMarker, QuantityValueError};
use crate::PhysicalDimension;
use serde::{Deserialize, Deserializer, Serialize};
use std::marker::PhantomData;

/// Validated conversion from an authored/exchange unit into canonical SI.
/// Original unit symbols live in source evidence, never replace conversion arithmetic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct UnitDefinition {
    pub(super) dimension: PhysicalDimension,
    pub(super) scale_bits: u64,
    pub(super) offset_bits: u64,
    pub(super) angle_power: i8,
}
impl UnitDefinition {
    pub fn try_new(
        dimension: PhysicalDimension,
        scale: f64,
        offset: f64,
    ) -> Result<Self, UnitError> {
        Self::try_with_angle(dimension, scale, offset, 0)
    }
    pub fn try_for_kind(
        kind: super::QuantityKind,
        scale: f64,
        offset: f64,
    ) -> Result<Self, UnitError> {
        if offset != 0.0 && !kind.permits_affine_unit() {
            return Err(UnitError::InvalidAffineUnit);
        }
        Self::try_with_angle(kind.dimension(), scale, offset, kind.angle_power())
    }
    pub fn try_with_angle(
        dimension: PhysicalDimension,
        scale: f64,
        offset: f64,
        angle_power: i8,
    ) -> Result<Self, UnitError> {
        if !scale.is_finite() || scale <= 0.0 || !offset.is_finite() {
            return Err(UnitError::InvalidConversion);
        }
        if offset != 0.0 && (dimension != PhysicalDimension::TEMPERATURE || angle_power != 0) {
            return Err(UnitError::InvalidAffineUnit);
        }
        Ok(Self {
            dimension,
            angle_power,
            scale_bits: scale.to_bits(),
            offset_bits: if offset == 0.0 {
                0.0_f64.to_bits()
            } else {
                offset.to_bits()
            },
        })
    }
    pub const fn dimension(&self) -> PhysicalDimension {
        self.dimension
    }
    pub const fn angle_power(&self) -> i8 {
        self.angle_power
    }
    pub fn scale(&self) -> f64 {
        f64::from_bits(self.scale_bits)
    }
    pub fn offset(&self) -> f64 {
        f64::from_bits(self.offset_bits)
    }
    pub const fn scale_bits(&self) -> u64 {
        self.scale_bits
    }
    pub const fn offset_bits(&self) -> u64 {
        self.offset_bits
    }
    pub fn to_canonical(&self, value: f64) -> Result<f64, UnitError> {
        let value = value * self.scale() + self.offset();
        if !value.is_finite() {
            return Err(UnitError::NonFiniteValue);
        }
        Ok(value)
    }
    pub fn from_canonical(&self, value: f64) -> Result<f64, UnitError> {
        let value = (value - self.offset()) / self.scale();
        if !value.is_finite() {
            return Err(UnitError::NonFiniteValue);
        }
        Ok(value)
    }
}
#[doc(hidden)]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[cfg_attr(feature = "tool-schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct UnitDefinitionWire {
    pub dimension: PhysicalDimension,
    pub scale: f64,
    pub offset: f64,
    #[serde(default, skip_serializing_if = "angle_is_zero")]
    pub angle_power: i8,
}
fn angle_is_zero(value: &i8) -> bool {
    *value == 0
}

impl<'de> Deserialize<'de> for UnitDefinition {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let wire = UnitDefinitionWire::deserialize(d)?;
        Self::try_with_angle(wire.dimension, wire.scale, wire.offset, wire.angle_power)
            .map_err(serde::de::Error::custom)
    }
}
/// A unit admitted for one physical quantity kind. For example a Celsius offset
/// is admissible for absolute Temperature and inadmissible for its difference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent, bound = "")]
pub struct Unit<K: QuantityMarker> {
    definition: UnitDefinition,
    #[serde(skip)]
    kind: PhantomData<K>,
}
impl<K: QuantityMarker> Unit<K> {
    pub fn try_from_definition(definition: UnitDefinition) -> Result<Self, UnitError> {
        if definition.dimension() != K::KIND.dimension() {
            return Err(UnitError::DimensionMismatch);
        }
        if definition.angle_power() != K::KIND.angle_power() {
            return Err(UnitError::AngleMismatch);
        }
        if definition.offset() != 0.0 && !K::AFFINE {
            return Err(UnitError::InvalidAffineUnit);
        }
        Ok(Self {
            definition,
            kind: PhantomData,
        })
    }
    pub fn definition(&self) -> &UnitDefinition {
        &self.definition
    }
    pub fn quantity(&self, value: f64) -> Result<Quantity<K>, UnitError> {
        Quantity::try_new(self.definition.to_canonical(value)?)
            .map_err(|_| UnitError::NonFiniteValue)
    }
    pub fn display_value(&self, value: Quantity<K>) -> Result<f64, UnitError> {
        self.definition.from_canonical(value.canonical_value())
    }
}
impl<'de, K: QuantityMarker> Deserialize<'de> for Unit<K> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        Self::try_from_definition(UnitDefinition::deserialize(d)?).map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum UnitError {
    #[error("unit symbol must be nonempty, bounded and contain no control characters")]
    InvalidSymbol,
    #[error("unit scale must be positive and finite and its offset finite")]
    InvalidConversion,
    #[error("unit dimension does not match the physical quantity")]
    DimensionMismatch,
    #[error("nominal angle powers do not match the required quantity")]
    AngleMismatch,
    #[error("semantic physical quantity kinds do not match")]
    KindMismatch,
    #[error("affine conversion requires an absolute temperature quantity")]
    InvalidAffineUnit,
    #[error("physical conversion must produce a finite value")]
    NonFiniteValue,
}

/// Original value/unit and attribution survive normalization. The canonical
/// value is computed from source evidence, never accepted independently.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(deny_unknown_fields, bound = "")]
pub struct SourceMeasure<K: QuantityMarker> {
    source_value: f64,
    source_unit_symbol: String,
    source_unit: Unit<K>,
    attribution: UnitAttribution,
}
impl<K: QuantityMarker> SourceMeasure<K> {
    pub fn try_new(
        value: f64,
        symbol: impl Into<String>,
        unit: Unit<K>,
        attribution: UnitAttribution,
    ) -> Result<Self, UnitError> {
        let symbol = symbol.into();
        if symbol.is_empty() || symbol.len() > 256 || symbol.chars().any(char::is_control) {
            return Err(UnitError::InvalidSymbol);
        }
        unit.quantity(value)?;
        attribution.validate()?;
        Ok(Self {
            source_value: value,
            source_unit_symbol: symbol,
            source_unit: unit,
            attribution,
        })
    }
    pub fn canonical(&self) -> Result<Quantity<K>, UnitError> {
        self.source_unit.quantity(self.source_value)
    }
    pub fn source_value(&self) -> f64 {
        self.source_value
    }
    pub fn source_unit(&self) -> &Unit<K> {
        &self.source_unit
    }
    pub fn source_unit_symbol(&self) -> &str {
        &self.source_unit_symbol
    }
    pub fn attribution(&self) -> &UnitAttribution {
        &self.attribution
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "tool-schema", derive(schemars::JsonSchema))]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum UnitAttribution {
    FileDeclaration { context: String },
    Authored { source: String },
    ExplicitOverride { source: String, reason: String },
    CanonicalComputation,
}

// Keep errors independently named at numeric and conversion admission boundaries.
impl From<QuantityValueError> for UnitError {
    fn from(_: QuantityValueError) -> Self {
        Self::NonFiniteValue
    }
}

impl Serialize for UnitDefinition {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        UnitDefinitionWire {
            dimension: self.dimension,
            scale: self.scale(),
            offset: self.offset(),
            angle_power: self.angle_power(),
        }
        .serialize(s)
    }
}
impl<'de, K: QuantityMarker> Deserialize<'de> for SourceMeasure<K> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields, bound(deserialize = "Unit<K>: Deserialize<'de>"))]
        struct Wire<K: QuantityMarker> {
            source_value: f64,
            source_unit_symbol: String,
            source_unit: Unit<K>,
            attribution: UnitAttribution,
        }
        let w = Wire::<K>::deserialize(d)?;
        Self::try_new(
            w.source_value,
            w.source_unit_symbol,
            w.source_unit,
            w.attribution,
        )
        .map_err(serde::de::Error::custom)
    }
}

pub type LengthUnit = Unit<super::kind::Length>;
#[allow(non_upper_case_globals)]
impl LengthUnit {
    pub const Meter: Self = Self::catalog(1.0);
    pub const Millimeter: Self = Self::catalog(0.001);
    pub const Centimeter: Self = Self::catalog(0.01);
    pub const Micrometer: Self = Self::catalog(0.000001);
    pub const Kilometer: Self = Self::catalog(1000.0);
    pub const Inch: Self = Self::catalog(0.0254);
    pub const Foot: Self = Self::catalog(0.3048);
    const fn catalog(scale: f64) -> Self {
        Self {
            definition: UnitDefinition {
                dimension: PhysicalDimension::LENGTH,
                scale_bits: scale.to_bits(),
                offset_bits: 0.0f64.to_bits(),
                angle_power: 0,
            },
            kind: PhantomData,
        }
    }
    /// The length unit of size `meters_per_unit`; `None` unless finite and positive. Every SI
    /// prefix on the metre and every conversion-based unit (inch, foot, mil, ...) is a length
    /// unit of its exact size.
    pub fn from_meters_per_unit(meters_per_unit: f64) -> Option<Self> {
        let definition =
            UnitDefinition::try_new(PhysicalDimension::LENGTH, meters_per_unit, 0.0).ok()?;
        Self::try_from_definition(definition).ok()
    }
    pub fn meters_per_unit(self) -> f64 {
        self.definition.scale()
    }
    /// The length unit a conversion-based unit's standard name defines (inch, foot, mil,
    /// microinch), case-insensitive. A declared conversion factor always governs; this is
    /// the definition for a named unit whose factor is absent.
    pub fn from_conversion_name(name: &str) -> Option<Self> {
        let meters_per_unit = match name.trim().to_ascii_lowercase().as_str() {
            "inch" | "in" => 0.0254,
            "foot" | "ft" => 0.3048,
            "mil" | "thou" => 2.54e-5,
            "microinch" => 2.54e-8,
            _ => return None,
        };
        Self::from_meters_per_unit(meters_per_unit)
    }
    pub fn symbol(self) -> &'static str {
        match self.definition.scale_bits {
            bits if bits == 1.0f64.to_bits() => "m",
            bits if bits == 0.001f64.to_bits() => "mm",
            bits if bits == 0.01f64.to_bits() => "cm",
            bits if bits == 0.000001f64.to_bits() => "um",
            bits if bits == 1000.0f64.to_bits() => "km",
            bits if bits == 0.0254f64.to_bits() => "in",
            bits if bits == 0.3048f64.to_bits() => "ft",
            bits if bits == 1e-9f64.to_bits() => "nm",
            bits if bits == 1e-10f64.to_bits() => "Angstrom",
            bits if bits == 1e-12f64.to_bits() => "pm",
            bits if bits == 0.1f64.to_bits() => "dm",
            bits if bits == 2.54e-5f64.to_bits() => "mil",
            bits if bits == 2.54e-8f64.to_bits() => "uin",
            bits if bits == 1609.344f64.to_bits() => "mi",
            _ => "custom length unit",
        }
    }
}

#[cfg(feature = "tool-schema")]
impl schemars::JsonSchema for UnitDefinition {
    fn schema_name() -> String {
        "PhysicalUnitDefinition".into()
    }
    fn json_schema(g: &mut schemars::gen::SchemaGenerator) -> schemars::schema::Schema {
        UnitDefinitionWire::json_schema(g)
    }
}
#[cfg(feature = "tool-schema")]
impl<K: QuantityMarker> schemars::JsonSchema for Unit<K> {
    fn schema_name() -> String {
        format!("{:?}Unit", K::KIND)
    }
    fn json_schema(g: &mut schemars::gen::SchemaGenerator) -> schemars::schema::Schema {
        UnitDefinition::json_schema(g)
    }
}

impl UnitAttribution {
    pub fn owned_buffer_bytes(&self) -> Option<usize> {
        match self {
            Self::FileDeclaration { context } => Some(context.capacity()),
            Self::Authored { source } => Some(source.capacity()),
            Self::ExplicitOverride { source, reason } => {
                source.capacity().checked_add(reason.capacity())
            }
            Self::CanonicalComputation => Some(0),
        }
    }
    pub fn validate(&self) -> Result<(), UnitError> {
        fn valid(s: &str) -> bool {
            !s.trim().is_empty() && s.len() <= 1024 && !s.chars().any(char::is_control)
        }
        match self {
            Self::FileDeclaration { context } if valid(context) => Ok(()),
            Self::Authored { source } if valid(source) => Ok(()),
            Self::ExplicitOverride { source, reason } if valid(source) && valid(reason) => Ok(()),
            Self::CanonicalComputation => Ok(()),
            _ => Err(UnitError::InvalidSymbol),
        }
    }
}
