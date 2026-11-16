//! An assembly's document settings: the unit system its readouts use.
//!
//! Written in the assembly as `settings = { units = ips }`. Mass properties, the BOM's mass
//! column and Measure's readouts show their values in these units: `mmgs` (millimeter, gram),
//! `mks` (meter, kilogram) or `ips` (inch, pound). An assembly without settings reads in
//! millimeters with masses in grams and kilograms, as before settings existed. The model itself
//! is always in SI; the settings change only how values read.

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

/// `settings = { units = … }` on an assembly.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DocumentSettings {
    #[serde(default)]
    pub units: UnitSystem,
}

const INCH: f64 = 0.0254;
const POUND: f64 = 0.453_592_37;

/// `value` with up to `digits` significant digits, without trailing zeros; scientific when it
/// is very small or very large.
pub fn significant(value: f64, digits: i32) -> String {
    if value == 0.0 || !value.is_finite() {
        return if value.is_finite() { "0".into() } else { value.to_string() };
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
        format!("{} {}", significant(self.length_value(m), 6), self.length_unit())
    }

    /// An area in square meters.
    pub fn area(self, m2: f64) -> String {
        let l = self.length_value(1.0);
        format!("{} {}²", significant(m2 * l * l, 6), self.length_unit())
    }

    /// A volume in cubic meters.
    pub fn volume(self, m3: f64) -> String {
        let l = self.length_value(1.0);
        format!("{} {}³", significant(m3 * l * l * l, 6), self.length_unit())
    }

    /// A mass in kilograms.
    pub fn mass(self, kg: f64) -> String {
        format!("{} {}", significant(self.mass_value(kg), 6), self.mass_unit())
    }

    /// A density in kilograms per cubic meter.
    pub fn density(self, kg_m3: f64) -> String {
        let l = self.length_value(1.0);
        format!(
            "{} {}/{}³",
            significant(self.mass_value(kg_m3) / (l * l * l), 5),
            self.mass_unit(),
            self.length_unit()
        )
    }

    /// A moment of inertia in kilogram square meters.
    pub fn inertia(self, kg_m2: f64) -> String {
        let l = self.length_value(1.0);
        format!(
            "{} {}·{}²",
            significant(self.mass_value(kg_m2) * l * l, 6),
            self.mass_unit(),
            self.length_unit()
        )
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
}
