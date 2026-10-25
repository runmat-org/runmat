use serde::{Deserialize, Serialize};

/// Length unit of geometry coordinates. Metres, millimetres and inches have names an author can
/// write; any other unit a CAD file declares (an SI prefix on the metre, feet, mils, microinches,
/// or a conversion-based unit with its own factor) is carried exactly as `Length`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnitSystem {
    Unspecified,
    Meter,
    Millimeter,
    Inch,
    /// A declared length unit of the given size in metres. Construct with
    /// [`UnitSystem::from_meters_per_unit`], which maps the named units to their variants.
    Length {
        meters_per_unit: LengthScale,
    },
}

impl UnitSystem {
    /// The unit whose size is `meters_per_unit`; `None` unless finite and positive.
    pub fn from_meters_per_unit(meters_per_unit: f64) -> Option<Self> {
        if !meters_per_unit.is_finite() || meters_per_unit <= 0.0 {
            return None;
        }
        let close = |reference: f64| (meters_per_unit / reference - 1.0).abs() <= 1.0e-12;
        Some(if close(1.0) {
            Self::Meter
        } else if close(0.001) {
            Self::Millimeter
        } else if close(0.0254) {
            Self::Inch
        } else {
            Self::Length {
                meters_per_unit: LengthScale(meters_per_unit.to_bits()),
            }
        })
    }

    /// Size of one unit in metres; `None` for `Unspecified`.
    pub fn meters_per_unit(self) -> Option<f64> {
        match self {
            Self::Unspecified => None,
            Self::Meter => Some(1.0),
            Self::Millimeter => Some(0.001),
            Self::Inch => Some(0.0254),
            Self::Length { meters_per_unit } => Some(meters_per_unit.meters()),
        }
    }
}

/// Exact positive length in metres, compared and hashed by value bits so units stay `Eq`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LengthScale(u64);

impl LengthScale {
    pub fn meters(self) -> f64 {
        f64::from_bits(self.0)
    }
}

impl Serialize for LengthScale {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_f64(self.meters())
    }
}

impl<'de> Deserialize<'de> for LengthScale {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let meters = f64::deserialize(deserializer)?;
        if !meters.is_finite() || meters <= 0.0 {
            return Err(serde::de::Error::custom(
                "length unit scale must be finite and positive",
            ));
        }
        Ok(Self(meters.to_bits()))
    }
}
