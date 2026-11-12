//! Units in `.part`, `.assembly` and `.drawing` files.
//!
//! RunMat has one unit authority, `runmat-types::physical`: its registry parses every unit
//! expression (`mm`, `kg/m^3`, `50 mil`), its dimensions check every value, and its
//! [`unit_mismatch_message`](runmat_types::physical::unit_mismatch_message) writes every
//! wrong-unit message. [`Units`] is this crate's handle on it, and [`Dimension`] is its
//! dimension descriptor. A value's dimension travels to the Desktop as the design catalog's wire
//! form (`{ base, angle }`, see [`wire`]).

pub use runmat_types::physical::DynamicDimension as Dimension;
use runmat_types::physical::{
    parse_unit_expecting, parse_unit_expression, QuantityKind, UnitDefinition,
};
use runmat_types::PhysicalDimension;
use serde::{Deserialize, Serialize};
use std::fmt;

/// A quantity kind's name as a design file writes it (`length`, `angle`, `mass_density`).
/// Equal dimension does not mean equal kind; the authority decides compatibility.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct QuantityKindId(pub String);

impl QuantityKindId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
}

impl fmt::Display for QuantityKindId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// The unit authority as the design layer reads it: `runmat-types::physical`'s registry and
/// quantity catalog.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Units;

impl Units {
    /// The dimension a quantity kind name declares: any catalog kind by its snake-case name
    /// (`length`, `force`, `mass_density`), plus the design vocabulary's `distance` (a length)
    /// and the plain-number kinds `real`, `count` and `integer`.
    pub fn kind(&self, name: &QuantityKindId) -> Option<Dimension> {
        let kind = match name.0.as_str() {
            "distance" => QuantityKind::Length,
            "dimensionless" | "real" | "count" | "integer" => QuantityKind::Dimensionless,
            other => serde_json::from_value(serde_json::Value::String(other.to_owned())).ok()?,
        };
        Some(Dimension::quantity_type(kind))
    }

    /// A unit expression as written after a number (`mm`, `kg/m^3`, `deg`). `1` and an empty
    /// unit are a plain number.
    pub fn unit(&self, expression: &str) -> Option<UnitDefinition> {
        match expression.trim() {
            "" | "1" => UnitDefinition::try_new(PhysicalDimension::DIMENSIONLESS, 1.0, 0.0).ok(),
            written => parse_unit_expression(written).ok(),
        }
    }

    /// [`Units::unit`] where the value must measure `expected`: where a mass is expected, `g`
    /// is the gram (`0.35 g`), and elsewhere standard gravity.
    pub fn unit_expecting(&self, expression: &str, expected: Dimension) -> Option<UnitDefinition> {
        match expression.trim() {
            "" | "1" => self.unit(expression),
            written => parse_unit_expecting(written, expected.dimension()).ok(),
        }
    }

    /// A fingerprint of the registry; it joins regeneration cache keys.
    pub fn fingerprint(&self) -> String {
        "runmat-types/physical/1".to_owned()
    }
}

/// The dimension of a unit definition, without a semantic kind.
pub fn dimension_of(unit: &UnitDefinition) -> Dimension {
    Dimension::from_unit(*unit)
}

/// A dimension on the wire: the design catalog's `{ base: [8 exponents], angle }`
/// ([`runmat_design_commands::Dimension`]), which the Desktop reads. Use with
/// `#[serde(with = "crate::units::wire")]`.
pub mod wire {
    use super::Dimension;
    use runmat_design_commands::Dimension as Wire;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn to_wire(dimension: Dimension) -> Wire {
        Wire::from(dimension)
    }

    pub fn from_wire(wire: Wire) -> Dimension {
        wire.to_physical()
    }

    pub fn serialize<S: Serializer>(dimension: &Dimension, s: S) -> Result<S::Ok, S::Error> {
        to_wire(*dimension).serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Dimension, D::Error> {
        Wire::deserialize(d).map(from_wire)
    }

    /// The same, for an optional dimension.
    pub mod option {
        use super::{from_wire, to_wire, Dimension, Wire};
        use serde::{Deserialize, Deserializer, Serialize, Serializer};

        pub fn serialize<S: Serializer>(
            dimension: &Option<Dimension>,
            s: S,
        ) -> Result<S::Ok, S::Error> {
            dimension.map(to_wire).serialize(s)
        }

        pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Dimension>, D::Error> {
            Option::<Wire>::deserialize(d).map(|w| w.map(from_wire))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_kinds_and_units_from_the_one_authority() {
        let units = Units;
        assert_eq!(
            units.kind(&QuantityKindId::new("distance")),
            Some(Dimension::LENGTH)
        );
        assert_eq!(
            units.kind(&QuantityKindId::new("angle")),
            Some(Dimension::ANGLE)
        );
        assert_eq!(
            units.kind(&QuantityKindId::new("count")),
            Some(Dimension::DIMENSIONLESS)
        );
        assert!(units.kind(&QuantityKindId::new("force")).is_some());
        assert!(units.kind(&QuantityKindId::new("no_such_kind")).is_none());
        let mil = units.unit("mil").unwrap();
        assert_eq!(dimension_of(&mil), Dimension::LENGTH);
        assert!((mil.scale() - 2.54e-5).abs() < 1e-18);
        assert_eq!(dimension_of(&units.unit("deg").unwrap()), Dimension::ANGLE);
        assert!(units.unit("furlong").is_none());
    }

    #[test]
    fn dimensions_keep_the_catalog_wire_form() {
        let json = serde_json::to_value(wire::to_wire(Dimension::ANGLE)).unwrap();
        assert_eq!(
            json,
            serde_json::json!({ "base": [0, 0, 0, 0, 0, 0, 0, 0], "angle": 1 })
        );
        assert_eq!(
            wire::from_wire(wire::to_wire(Dimension::LENGTH)),
            Dimension::LENGTH
        );
    }
}

/// A written length in millimetres: a quantity in any length unit the catalog knows (`mm`, `in`,
/// `ft`, …) or a bare number, which is millimetres. None for anything else.
pub fn length_mm(v: &crate::value::ValueExpr) -> Option<f64> {
    match v {
        crate::value::ValueExpr::Quantity { value, unit } => {
            let info = Units.unit(unit)?;
            (dimension_of(&info) == Dimension::LENGTH).then_some(value * info.scale() * 1000.0)
        }
        crate::value::ValueExpr::Number { value } => Some(*value),
        _ => None,
    }
}
