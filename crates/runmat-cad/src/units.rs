//! The unit seam.
//!
//! RunMat has one unit authority (`runmat-types::physical`, reached through the `.fea` registry).
//! This crate does not build a second one. It keeps a minimal dimension type for checking and a
//! [`UnitAuthority`] trait; the real registry is adapted to the trait later. [`StubUnits`] covers
//! the handful of kinds and symbols the design-layer tests need, and nothing more.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Exponents over the eight SI-style bases the authority uses (length, mass, time, current,
/// temperature, amount, luminous intensity, information), plus a separate angle power, matching
/// `runmat-types::physical::PhysicalDimension`'s layout so the adapter is a field copy.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct Dimension {
    pub base: [i8; 8],
    pub angle: i8,
}

impl Dimension {
    pub const DIMENSIONLESS: Dimension = Dimension {
        base: [0; 8],
        angle: 0,
    };
    pub const LENGTH: Dimension = Dimension {
        base: [1, 0, 0, 0, 0, 0, 0, 0],
        angle: 0,
    };
    pub const MASS: Dimension = Dimension {
        base: [0, 1, 0, 0, 0, 0, 0, 0],
        angle: 0,
    };
    pub const ANGLE: Dimension = Dimension {
        base: [0; 8],
        angle: 1,
    };
    pub const AREA: Dimension = Dimension {
        base: [2, 0, 0, 0, 0, 0, 0, 0],
        angle: 0,
    };

    pub fn multiply(self, other: Dimension) -> Dimension {
        let mut base = [0i8; 8];
        for (i, b) in base.iter_mut().enumerate() {
            *b = self.base[i] + other.base[i];
        }
        Dimension {
            base,
            angle: self.angle + other.angle,
        }
    }

    pub fn divide(self, other: Dimension) -> Dimension {
        let mut base = [0i8; 8];
        for (i, b) in base.iter_mut().enumerate() {
            *b = self.base[i] - other.base[i];
        }
        Dimension {
            base,
            angle: self.angle - other.angle,
        }
    }

    pub fn power(self, exponent: i8) -> Dimension {
        let mut base = [0i8; 8];
        for (i, b) in base.iter_mut().enumerate() {
            *b = self.base[i] * exponent;
        }
        Dimension {
            base,
            angle: self.angle * exponent,
        }
    }

    pub fn is_dimensionless(self) -> bool {
        self == Dimension::DIMENSIONLESS
    }
}

impl fmt::Display for Dimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const NAMES: [&str; 8] = ["L", "M", "T", "I", "Θ", "N", "J", "B"];
        let mut wrote = false;
        for (i, e) in self.base.iter().enumerate() {
            if *e != 0 {
                if wrote {
                    f.write_str("·")?;
                }
                if *e == 1 {
                    write!(f, "{}", NAMES[i])?;
                } else {
                    write!(f, "{}^{}", NAMES[i], e)?;
                }
                wrote = true;
            }
        }
        if self.angle != 0 {
            if wrote {
                f.write_str("·")?;
            }
            write!(f, "angle^{}", self.angle)?;
            wrote = true;
        }
        if !wrote {
            f.write_str("1")?;
        }
        Ok(())
    }
}

/// A quantity kind's catalog name (`length`, `angle`, `mass_density`). Equal dimension does not
/// mean equal kind; the authority decides compatibility.
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KindInfo {
    pub id: QuantityKindId,
    pub dimension: Dimension,
    /// The canonical SI unit symbol used in messages.
    pub si_symbol: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UnitInfo {
    pub symbol: String,
    pub dimension: Dimension,
    /// Multiply a value in this unit by `scale` to get SI.
    pub scale: f64,
}

/// What the design layer needs from the unit authority.
pub trait UnitAuthority {
    /// Look a quantity kind up by its catalog name.
    fn kind(&self, name: &QuantityKindId) -> Option<KindInfo>;
    /// Parse a unit expression as written after a number (`mm`, `kg/m^3`).
    fn unit(&self, expression: &str) -> Option<UnitInfo>;
    /// A fingerprint of the catalog; it joins regeneration cache keys.
    fn fingerprint(&self) -> String;
}

/// A deliberately small stand-in for the `.fea` registry: length, area, angle, mass,
/// dimensionless and count. It parses single symbols only.
#[derive(Clone, Debug, Default)]
pub struct StubUnits;

impl UnitAuthority for StubUnits {
    fn kind(&self, name: &QuantityKindId) -> Option<KindInfo> {
        let (dimension, si) = match name.0.as_str() {
            "length" | "distance" => (Dimension::LENGTH, "m"),
            "area" => (Dimension::AREA, "m^2"),
            "angle" => (Dimension::ANGLE, "rad"),
            "mass" => (Dimension::MASS, "kg"),
            "dimensionless" | "real" | "count" | "integer" => (Dimension::DIMENSIONLESS, "1"),
            _ => return None,
        };
        Some(KindInfo {
            id: name.clone(),
            dimension,
            si_symbol: si.to_owned(),
        })
    }

    fn unit(&self, expression: &str) -> Option<UnitInfo> {
        let (dimension, scale) = match expression {
            "m" => (Dimension::LENGTH, 1.0),
            "mm" => (Dimension::LENGTH, 1e-3),
            // Surface roughness (Ra) and fine tolerances read in micrometres.
            "um" | "µm" => (Dimension::LENGTH, 1e-6),
            "cm" => (Dimension::LENGTH, 1e-2),
            "in" => (Dimension::LENGTH, 0.0254),
            "ft" => (Dimension::LENGTH, 0.3048),
            "m^2" => (Dimension::AREA, 1.0),
            "mm^2" => (Dimension::AREA, 1e-6),
            "rad" => (Dimension::ANGLE, 1.0),
            "deg" => (Dimension::ANGLE, std::f64::consts::PI / 180.0),
            "kg" => (Dimension::MASS, 1.0),
            "gram" => (Dimension::MASS, 1e-3),
            "1" | "" => (Dimension::DIMENSIONLESS, 1.0),
            _ => return None,
        };
        Some(UnitInfo {
            symbol: expression.to_owned(),
            dimension,
            scale,
        })
    }

    fn fingerprint(&self) -> String {
        "runmat-cad/stub-units/3".to_owned()
    }
}
