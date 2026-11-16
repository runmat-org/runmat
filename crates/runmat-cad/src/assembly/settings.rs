//! A part's or an assembly's document settings: the unit system and decimals its readouts
//! use, and on a part its drafting standard and general tolerance.
//!
//! Written as `settings = { units = ips }` (an assembly) or
//! `settings = { units = mmgs, standard = asme, tolerance = medium, decimals = 2 }` (a part).
//! Mass properties, the BOM's mass column and Measure's readouts show their values in these
//! units: `mmgs` (millimeter, gram), `mks` (meter, kilogram) or `ips` (inch, pound). A file
//! without settings reads in millimeters with masses in grams and kilograms, as before settings
//! existed. The model itself is always in SI; the settings change only how values read.
//!
//! - `decimals`: readouts round to this many decimal places; without it they keep six
//!   significant digits.
//! - `tolerance`: the ISO 2768-1 general tolerance class for linear dimensions written without
//!   a tolerance (`fine`, `medium`, `coarse`, `very_coarse`). Measure lists the permissible
//!   deviation beside a distance ([`GeneralTolerance::linear`]).
//! - `standard`: the drafting standard the part's 3D annotations follow (`iso` or `asme`,
//!   [`crate::drawing::standard::DrawingStandard`]). On an ASME part a GD&T frame with a
//!   characteristic ASME Y14.5-2018 withdrew is reported.

use serde::{Deserialize, Serialize};

/// A unit system for lengths, areas, volumes, masses, densities and inertias.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnitSystem {
    /// Millimeter, gram, second.
    #[default]
    Mmgs,
    /// Meter, kilogram, second.
    Mks,
    /// Inch, pound, second.
    Ips,
}

/// `settings = { units = … }` on a part or an assembly.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DocumentSettings {
    #[serde(default)]
    pub units: UnitSystem,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub standard: Option<crate::drawing::standard::DrawingStandard>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tolerance: Option<GeneralTolerance>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decimals: Option<u32>,
}

impl DocumentSettings {
    /// How this file's readouts show values.
    pub fn readout(&self) -> Readout {
        Readout {
            units: self.units,
            decimals: self.decimals,
            tolerance: self.tolerance,
        }
    }
}

/// An ISO 2768-1 general tolerance class for linear dimensions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeneralTolerance {
    /// Class f.
    Fine,
    /// Class m.
    Medium,
    /// Class c.
    Coarse,
    /// Class v.
    VeryCoarse,
}

/// ISO 2768-1:1989 Table 1, permissible deviations for linear dimensions (mm): the upper end of
/// each nominal size range, then the deviation in classes f, m, c and v. `None`: the standard
/// gives no value for that class and range.
const LINEAR: [(f64, [Option<f64>; 4]); 8] = [
    (3.0, [Some(0.05), Some(0.1), Some(0.2), None]),
    (6.0, [Some(0.05), Some(0.1), Some(0.3), Some(0.5)]),
    (30.0, [Some(0.1), Some(0.2), Some(0.5), Some(1.0)]),
    (120.0, [Some(0.15), Some(0.3), Some(0.8), Some(1.5)]),
    (400.0, [Some(0.2), Some(0.5), Some(1.2), Some(2.5)]),
    (1000.0, [Some(0.3), Some(0.8), Some(2.0), Some(4.0)]),
    (2000.0, [Some(0.5), Some(1.2), Some(3.0), Some(6.0)]),
    (4000.0, [None, Some(2.0), Some(4.0), Some(8.0)]),
];

impl GeneralTolerance {
    /// The class letter ISO 2768-1 names it by.
    pub fn letter(self) -> char {
        match self {
            GeneralTolerance::Fine => 'f',
            GeneralTolerance::Medium => 'm',
            GeneralTolerance::Coarse => 'c',
            GeneralTolerance::VeryCoarse => 'v',
        }
    }

    /// The permissible deviation (±, meters) for a linear dimension of `nominal` meters. `None`
    /// below 0.5 mm, above 4000 mm, and where the table has no value for the class.
    pub fn linear(self, nominal: f64) -> Option<f64> {
        let mm = nominal.abs() * 1000.0;
        if mm < 0.5 {
            return None;
        }
        let column = match self {
            GeneralTolerance::Fine => 0,
            GeneralTolerance::Medium => 1,
            GeneralTolerance::Coarse => 2,
            GeneralTolerance::VeryCoarse => 3,
        };
        LINEAR
            .iter()
            .find(|(upper, _)| mm <= *upper)
            .and_then(|(_, d)| d[column])
            .map(|d| d / 1000.0)
    }
}

/// How a file's readouts show values: its units, rounded to its decimals, with its general
/// tolerance for distances.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Readout {
    pub units: UnitSystem,
    pub decimals: Option<u32>,
    pub tolerance: Option<GeneralTolerance>,
}

impl From<UnitSystem> for Readout {
    fn from(units: UnitSystem) -> Self {
        Readout {
            units,
            decimals: None,
            tolerance: None,
        }
    }
}

const INCH: f64 = 0.0254;
const POUND: f64 = 0.453_592_37;

/// `value` with up to `digits` significant digits, without trailing zeros; scientific when it
/// is very small or very large.
pub fn significant(value: f64, digits: i32) -> String {
    if value == 0.0 || !value.is_finite() {
        return if value.is_finite() {
            "0".into()
        } else {
            value.to_string()
        };
    }
    let magnitude = value.abs().log10().floor() as i32;
    if !(-4..9).contains(&magnitude) {
        return format!("{:.*e}", (digits - 1).max(0) as usize, value);
    }
    let places = (digits - 1 - magnitude).max(0) as usize;
    let text = format!("{value:.places$}");
    if text.contains('.') {
        text.trim_end_matches('0').trim_end_matches('.').to_owned()
    } else {
        text
    }
}

impl UnitSystem {
    /// The `units` key as written (`mmgs`, `mks`, `ips`).
    pub fn key(self) -> &'static str {
        match self {
            UnitSystem::Mmgs => "mmgs",
            UnitSystem::Mks => "mks",
            UnitSystem::Ips => "ips",
        }
    }

    pub fn length_unit(self) -> &'static str {
        match self {
            UnitSystem::Mmgs => "mm",
            UnitSystem::Mks => "m",
            UnitSystem::Ips => "in",
        }
    }

    pub fn mass_unit(self) -> &'static str {
        match self {
            UnitSystem::Mmgs => "g",
            UnitSystem::Mks => "kg",
            UnitSystem::Ips => "lb",
        }
    }

    /// Meters in this system's length unit.
    pub fn length_value(self, m: f64) -> f64 {
        match self {
            UnitSystem::Mmgs => m * 1000.0,
            UnitSystem::Mks => m,
            UnitSystem::Ips => m / INCH,
        }
    }

    /// Kilograms in this system's mass unit.
    pub fn mass_value(self, kg: f64) -> f64 {
        match self {
            UnitSystem::Mmgs => kg * 1000.0,
            UnitSystem::Mks => kg,
            UnitSystem::Ips => kg / POUND,
        }
    }

    /// A length in meters, as it reads: `12.5 mm`, `0.0125 m`, `0.4921 in`.
    pub fn length(self, m: f64) -> String {
        Readout::from(self).length(m)
    }

    pub fn area(self, m2: f64) -> String {
        Readout::from(self).area(m2)
    }

    pub fn volume(self, m3: f64) -> String {
        Readout::from(self).volume(m3)
    }

    pub fn mass(self, kg: f64) -> String {
        Readout::from(self).mass(kg)
    }

    pub fn density(self, kg_m3: f64) -> String {
        Readout::from(self).density(kg_m3)
    }

    pub fn inertia(self, kg_m2: f64) -> String {
        Readout::from(self).inertia(kg_m2)
    }
}

impl Readout {
    /// `value` to the decimals, else to `digits` significant digits.
    pub fn number(self, value: f64, digits: i32) -> String {
        match self.decimals {
            Some(d) => format!("{value:.*}", d as usize),
            None => significant(value, digits),
        }
    }

    pub fn length_value(self, m: f64) -> f64 {
        self.units.length_value(m)
    }

    pub fn length_unit(self) -> &'static str {
        self.units.length_unit()
    }

    pub fn mass_unit(self) -> &'static str {
        self.units.mass_unit()
    }

    /// A length in meters.
    pub fn length(self, m: f64) -> String {
        format!(
            "{} {}",
            self.number(self.units.length_value(m), 6),
            self.length_unit()
        )
    }

    /// An area in square meters.
    pub fn area(self, m2: f64) -> String {
        let l = self.units.length_value(1.0);
        format!("{} {}²", self.number(m2 * l * l, 6), self.length_unit())
    }

    /// A volume in cubic meters.
    pub fn volume(self, m3: f64) -> String {
        let l = self.units.length_value(1.0);
        format!("{} {}³", self.number(m3 * l * l * l, 6), self.length_unit())
    }

    /// A mass in kilograms.
    pub fn mass(self, kg: f64) -> String {
        format!(
            "{} {}",
            self.number(self.units.mass_value(kg), 6),
            self.mass_unit()
        )
    }

    /// A density in kilograms per cubic meter (significant digits: densities are small numbers
    /// in millimeter units).
    pub fn density(self, kg_m3: f64) -> String {
        let l = self.units.length_value(1.0);
        format!(
            "{} {}/{}³",
            significant(self.units.mass_value(kg_m3) / (l * l * l), 5),
            self.mass_unit(),
            self.length_unit()
        )
    }

    /// A moment of inertia in kilogram square meters.
    pub fn inertia(self, kg_m2: f64) -> String {
        let l = self.units.length_value(1.0);
        format!(
            "{} {}·{}²",
            self.number(self.units.mass_value(kg_m2) * l * l, 6),
            self.mass_unit(),
            self.length_unit()
        )
    }

    /// The general tolerance on a distance of `m` meters, as it reads:
    /// `± 0.3 mm (ISO 2768-m)`. `None` without a class, or outside the table.
    pub fn general_tolerance(self, m: f64) -> Option<String> {
        let class = self.tolerance?;
        let deviation = class.linear(m)?;
        Some(format!(
            "± {} {} (ISO 2768-{})",
            significant(self.units.length_value(deviation), 4),
            self.length_unit(),
            class.letter()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_system_reads_its_own_units() {
        // 100 × 10 × 10 mm of steel: 1e-5 m³, 0.0785 kg.
        assert_eq!(UnitSystem::Mmgs.mass(0.0785), "78.5 g");
        assert_eq!(UnitSystem::Mmgs.volume(1e-5), "10000 mm³");
        assert_eq!(UnitSystem::Mks.mass(0.0785), "0.0785 kg");
        assert_eq!(UnitSystem::Mks.length(0.0125), "0.0125 m");
        assert_eq!(UnitSystem::Ips.length(0.0254), "1 in");
        assert_eq!(UnitSystem::Ips.mass(0.45359237), "1 lb");
        assert_eq!(UnitSystem::Ips.volume(1e-5), "0.610237 in³");
        assert_eq!(UnitSystem::Mmgs.density(7850.0), "0.00785 g/mm³");
        assert_eq!(UnitSystem::Mks.inertia(2.0), "2 kg·m²");
        assert_eq!(UnitSystem::Mks.volume(1e-5), "1.00000e-5 m³");
    }

    #[test]
    fn decimals_round_and_the_general_tolerance_follows_iso_2768() {
        let r = Readout {
            units: UnitSystem::Mmgs,
            decimals: Some(2),
            tolerance: Some(GeneralTolerance::Medium),
        };
        assert_eq!(r.mass(0.0785), "78.50 g");
        assert_eq!(r.length(0.0125), "12.50 mm");
        // 100 mm sits in 30–120 mm: ±0.3 in class m, ±0.15 in f, ±0.8 in c.
        assert_eq!(
            r.general_tolerance(0.1).as_deref(),
            Some("± 0.3 mm (ISO 2768-m)")
        );
        assert_eq!(GeneralTolerance::Fine.linear(0.1), Some(0.15e-3));
        assert_eq!(GeneralTolerance::Coarse.linear(0.1), Some(0.8e-3));
        // Range ends belong to the lower range: 6 mm is "over 3 up to 6".
        assert_eq!(GeneralTolerance::Medium.linear(0.006), Some(0.1e-3));
        // Class v has no value up to 3 mm, class f none over 2000 mm; nothing under 0.5 mm.
        assert_eq!(GeneralTolerance::VeryCoarse.linear(0.002), None);
        assert_eq!(GeneralTolerance::Fine.linear(3.0), None);
        assert_eq!(GeneralTolerance::Medium.linear(0.0004), None);
        let inches = Readout {
            units: UnitSystem::Ips,
            ..r
        };
        assert_eq!(
            inches.general_tolerance(0.1).as_deref(),
            Some("± 0.01181 in (ISO 2768-m)")
        );
    }
}
