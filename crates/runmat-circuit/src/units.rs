//! Units for circuit values: component values (`100 nF`, `4.7 kohm`, `10 uH`), rule values,
//! simulation settings and library parameters all read their units from the one unit registry,
//! `runmat_types::physical`, the same one `.part`, `.assembly` and `.fea` use.
//!
//! The circuit's rule engine (`runmat_rules`) is dependency-free and takes a [`UnitResolver`];
//! [`RegistryUnits`] is that resolver over the registry. It adds only the notations circuit
//! text uses that aren't units of a physical quantity:
//! - `%` and `ppm`, ratios (`± 10 %`, `50 ppm`);
//! - `oz`, copper weight as a thickness (1 oz = 34.79 µm of copper);
//! - the symbol spellings `Ω`, `Ohm`, `°` and `°C`, read as `ohm`, `deg` and `degC`;
//! - `W/mK` and `mm2`, read as `W/m/K` and `mm^2`.
//!
//! A param whose name says what it measures reads its units expecting that ([`param_quantity`]):
//! `mass = 0.35 g` is 0.35 grams, where elsewhere `g` is standard gravity.

use runmat_rules::{Dim, UnitDef, UnitResolver};
use runmat_types::physical::{parse_unit_expecting, parse_unit_expression};
use runmat_types::PhysicalDimension;

/// The unit registry as the circuit's rule engine reads units.
#[derive(Clone, Copy, Debug, Default)]
pub struct RegistryUnits;

impl UnitResolver for RegistryUnits {
    fn resolve(&self, symbol: &str) -> Option<UnitDef> {
        resolve(symbol, None)
    }
}

/// The registry where the value must measure one dimension: a symbol with two readings takes
/// the one that fits (`g` is the gram where a mass is expected;
/// [`runmat_types::physical::parse_unit_expecting`]).
#[derive(Clone, Copy, Debug)]
pub struct ExpectingUnits(pub PhysicalDimension);

impl UnitResolver for ExpectingUnits {
    fn resolve(&self, symbol: &str) -> Option<UnitDef> {
        resolve(symbol, Some(self.0))
    }
}

/// What a param measures when its name says so: `mass` is a mass.
pub fn param_dimension(name: &str) -> Option<PhysicalDimension> {
    (name == "mass").then_some(PhysicalDimension::MASS)
}

/// A param's value text as a quantity, read with the units its name expects.
pub fn param_quantity(name: &str, text: &str) -> Result<runmat_rules::Quantity, String> {
    match param_dimension(name) {
        Some(expected) => runmat_rules::parse_quantity(text, &ExpectingUnits(expected)),
        None => runmat_rules::parse_quantity(text, &RegistryUnits),
    }
}

/// The rule engine's dimension for a registry dimension.
pub fn dim_of(d: PhysicalDimension) -> Dim {
    Dim([
        d.length,
        d.mass,
        d.time,
        d.current,
        d.temperature,
        d.amount,
        d.luminous_intensity,
    ])
}

fn resolve(symbol: &str, expected: Option<PhysicalDimension>) -> Option<UnitDef> {
    {
        let ratio = |scale: f64| {
            Some(UnitDef {
                scale,
                offset: 0.0,
                dim: Dim::NONE,
            })
        };
        let symbol = symbol.trim();
        match symbol {
            "" => return ratio(1.0),
            "%" => return ratio(0.01),
            "ppm" => return ratio(1e-6),
            "oz" => {
                return Some(UnitDef {
                    scale: 34.79e-6,
                    offset: 0.0,
                    dim: Dim::LENGTH,
                })
            }
            _ => {}
        }
        let spelled = match symbol {
            "°" => "deg".to_owned(),
            "°C" => "degC".to_owned(),
            "W/mK" | "W/(m*K)" => "W/m/K".to_owned(),
            "mm2" => "mm^2".to_owned(),
            other => other.replace('Ω', "ohm").replace("Ohm", "ohm"),
        };
        let unit = match expected {
            Some(expected) => parse_unit_expecting(&spelled, expected).ok()?,
            None => parse_unit_expression(&spelled).ok()?,
        };
        let d = unit.dimension();
        if d.information != 0 {
            return None;
        }
        Some(UnitDef {
            scale: unit.scale(),
            offset: unit.offset(),
            dim: dim_of(d),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_rules::parse_quantity;

    fn si(text: &str) -> (f64, Dim) {
        let q = parse_quantity(text, &RegistryUnits).unwrap_or_else(|e| panic!("{text}: {e}"));
        (q.value, q.dim)
    }

    #[test]
    fn component_values_read_through_the_registry() {
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-12 * b.abs().max(1e-30);
        let (v, d) = si("100 nF");
        assert!(close(v, 100e-9) && d == Dim::CAPACITANCE);
        let (v, d) = si("4.7uF");
        assert!(close(v, 4.7e-6) && d == Dim::CAPACITANCE);
        let (v, d) = si("4.7 µF");
        assert!(close(v, 4.7e-6) && d == Dim::CAPACITANCE);
        let (v, d) = si("10 uH");
        assert!(close(v, 10e-6) && d == Dim::INDUCTANCE);
        let (v, d) = si("2 mS");
        assert!(close(v, 2e-3) && d == Dim::CONDUCTANCE);
        assert_eq!(si("10 kohm"), (10_000.0, Dim::RESISTANCE));
        assert_eq!(si("10 kΩ"), (10_000.0, Dim::RESISTANCE));
        assert_eq!(si("1 Mohm"), (1e6, Dim::RESISTANCE));
        assert_eq!(si("3.3 V").1, Dim::VOLTAGE);
        assert_eq!(si("1 mV").0, 1e-3);
        assert_eq!(si("16 MHz"), (16e6, Dim::FREQUENCY));
        assert_eq!(si("1e-3"), (1e-3, Dim::NONE));
    }

    #[test]
    fn lengths_ratios_and_temperatures() {
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-15;
        let (v, d) = si("8mil");
        assert!(close(v, 8.0 * 25.4e-6) && d == Dim::LENGTH);
        assert!(close(si("0.2mm").0, 0.0002));
        assert!(close(si("1 oz").0, 34.79e-6));
        assert_eq!(si("10 %"), (0.1, Dim::NONE));
        assert!(close(si("50 ppm").0, 50e-6));
        assert!((si("25degC").0 - 298.15).abs() < 1e-9);
        assert!((si("90 °").0 - std::f64::consts::FRAC_PI_2).abs() < 1e-15);
        assert_eq!(si("1 K/W").1, Dim::THERMAL_RESISTANCE);
        assert_eq!(si("1 W/mK").1, Dim::THERMAL_CONDUCTIVITY);
        let (v, d) = si("1 mm2");
        assert!((v - 1e-6).abs() < 1e-18 && d == Dim::AREA);
        assert!(parse_quantity("3 furlongs", &RegistryUnits).is_err());
    }
}
