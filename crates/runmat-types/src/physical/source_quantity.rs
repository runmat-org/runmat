//! Type-erased source measures for checked catalogs and external exchanges.
//! The quantity kind is validated on admission and projects back to the same
//! Quantity<K> representation used by statically known model fields.
use super::{AnyQuantity, QuantityKind, UnitAttribution, UnitDefinition, UnitError};
use serde::{Deserialize, Deserializer, Serialize};
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "tool-schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct SourceQuantity {
    kind: QuantityKind,
    source_value: f64,
    source_unit: UnitDefinition,
    source_symbol: String,
    attribution: UnitAttribution,
}
impl SourceQuantity {
    pub fn try_new(
        kind: QuantityKind,
        value: f64,
        unit: UnitDefinition,
        symbol: impl Into<String>,
        attribution: UnitAttribution,
    ) -> Result<Self, UnitError> {
        let value = Self {
            kind,
            source_value: value,
            source_unit: unit,
            source_symbol: symbol.into(),
            attribution,
        };
        value.validate()?;
        Ok(value)
    }
    pub fn from_canonical(
        kind: QuantityKind,
        value: f64,
        attribution: UnitAttribution,
    ) -> Result<Self, UnitError> {
        Self::try_new(
            kind,
            value,
            UnitDefinition::try_for_kind(kind, 1.0, 0.0)?,
            kind.canonical_field(),
            attribution,
        )
    }
    pub fn kind(&self) -> QuantityKind {
        self.kind
    }
    pub fn dimension(&self) -> crate::PhysicalDimension {
        self.kind.dimension()
    }
    pub fn source_value(&self) -> f64 {
        self.source_value
    }
    pub fn source_unit(&self) -> &UnitDefinition {
        &self.source_unit
    }
    pub fn source_symbol(&self) -> &str {
        &self.source_symbol
    }
    pub fn attribution(&self) -> &UnitAttribution {
        &self.attribution
    }
    pub fn canonical_value(&self) -> f64 {
        self.source_unit
            .to_canonical(self.source_value)
            .expect("admitted source measure")
    }
    pub fn canonical(&self) -> AnyQuantity {
        AnyQuantity::try_new(self.kind, self.canonical_value()).expect("admitted source measure")
    }
    pub fn with_source_value(&self, value: f64) -> Result<Self, UnitError> {
        Self::try_new(
            self.kind,
            value,
            self.source_unit,
            self.source_symbol.clone(),
            self.attribution.clone(),
        )
    }
    pub fn owned_buffer_bytes(&self) -> Option<usize> {
        self.source_symbol
            .capacity()
            .checked_add(self.attribution.owned_buffer_bytes()?)
    }
    pub fn validate(&self) -> Result<(), UnitError> {
        if self.kind.dimension() != self.source_unit.dimension() {
            return Err(UnitError::DimensionMismatch);
        }
        if self.kind.angle_power() != self.source_unit.angle_power() {
            return Err(UnitError::AngleMismatch);
        }
        if !self.kind.permits_affine_unit() && self.source_unit.offset() != 0.0 {
            return Err(UnitError::InvalidAffineUnit);
        }
        if self.source_symbol.trim().is_empty()
            || self.source_symbol.len() > 256
            || self.source_symbol.chars().any(char::is_control)
        {
            return Err(UnitError::InvalidSymbol);
        }
        self.attribution.validate()?;
        self.source_unit.to_canonical(self.source_value)?;
        Ok(())
    }
}
impl<'de> Deserialize<'de> for SourceQuantity {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Wire {
            kind: QuantityKind,
            source_value: f64,
            source_unit: UnitDefinition,
            source_symbol: String,
            attribution: UnitAttribution,
        }
        let w = Wire::deserialize(d)?;
        Self::try_new(
            w.kind,
            w.source_value,
            w.source_unit,
            w.source_symbol,
            w.attribution,
        )
        .map_err(serde::de::Error::custom)
    }
}

impl PartialEq for SourceQuantity {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind
            && self.source_value.to_bits() == other.source_value.to_bits()
            && self.source_unit == other.source_unit
            && self.source_symbol == other.source_symbol
            && self.attribution == other.attribution
    }
}
impl Eq for SourceQuantity {}
