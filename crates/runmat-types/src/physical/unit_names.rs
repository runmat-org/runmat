//! Unit long names (`mm` → millimeter) and a unit's facts for hovers: its name, the kind of
//! quantity it measures and its SI conversion. Symbols are the ones [`super::parse_unit_expression`]
//! reads, SI-prefixed forms included.

use super::{parse_unit_expression, unit_kind};

/// Long names of the units the reader knows by symbol, before SI prefixes.
const NAMES: &[(&str, &str)] = &[
    ("m", "meter"),
    ("in", "inch"),
    ("inch", "inch"),
    ("ft", "foot"),
    ("mil", "mil"),
    ("thou", "thou"),
    ("s", "second"),
    ("Hz", "hertz"),
    ("kg", "kilogram"),
    ("gram", "gram"),
    ("g", "standard gravity, or gram where a mass is expected"),
    ("N", "newton"),
    ("Pa", "pascal"),
    ("J", "joule"),
    ("W", "watt"),
    ("K", "kelvin"),
    ("degC", "degree Celsius"),
    ("delta_degC", "degree Celsius difference"),
    ("A", "ampere"),
    ("C", "coulomb"),
    ("V", "volt"),
    ("ohm", "ohm"),
    ("F", "farad"),
    ("H", "henry"),
    ("S", "siemens"),
    ("T", "tesla"),
    ("rad", "radian"),
    ("deg", "degree"),
    ("rpm", "revolution per minute"),
    ("B", "byte"),
    ("KiB", "kibibyte"),
    ("MiB", "mebibyte"),
    ("GiB", "gibibyte"),
];

const PREFIX_NAMES: &[(&str, &str)] = &[
    ("T", "tera"),
    ("G", "giga"),
    ("M", "mega"),
    ("k", "kilo"),
    ("m", "milli"),
    ("c", "centi"),
    ("u", "micro"),
    ("n", "nano"),
    ("p", "pico"),
    ("f", "femto"),
];

/// The long name of a unit symbol: `mm` → `millimeter`, `kohm` → `kiloohm`, `MPa` → `megapascal`.
pub fn unit_name(symbol: &str) -> Option<String> {
    if let Some((_, name)) = NAMES.iter().find(|(s, _)| *s == symbol) {
        return Some((*name).to_string());
    }
    if parse_unit_expression(symbol).is_err() {
        return None;
    }
    PREFIX_NAMES.iter().find_map(|(prefix, prefix_name)| {
        let base = symbol.strip_prefix(prefix)?;
        let base_name = match base {
            "g" => "gram",
            _ => NAMES.iter().find(|(s, _)| *s == base)?.1,
        };
        Some(format!("{prefix_name}{base_name}"))
    })
}

/// What a hover shows for a unit.
#[derive(Clone, Debug, PartialEq)]
pub struct UnitFacts {
    /// The long name, for a single symbol (`millimeter`).
    pub name: Option<String>,
    /// The kind of quantity (`length`), when the dimension has a name.
    pub kind: Option<&'static str>,
    /// The coherent SI unit (`m`).
    pub si: Option<&'static str>,
    /// One of the unit in SI units.
    pub scale: f64,
    /// The SI value of the unit's zero (degree Celsius: 273.15 K).
    pub offset: f64,
}

/// Facts about a unit written as an expression (`mm`, `kg/m^3`).
pub fn unit_facts(text: &str) -> Option<UnitFacts> {
    let unit = parse_unit_expression(text).ok()?;
    let kind = unit_kind(unit.dimension(), unit.angle_power());
    Some(UnitFacts {
        name: unit_name(text),
        kind: kind.map(|k| k.kind),
        si: kind.map(|k| k.si_example),
        scale: unit.scale(),
        offset: unit.offset(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_reader_symbol_has_a_name() {
        let symbols = [
            "m", "mm", "um", "nm", "cm", "km", "in", "inch", "ft", "mil", "thou", "s", "ms", "us",
            "ns", "Hz", "kHz", "MHz", "GHz", "kg", "gram", "g", "N", "kN", "Pa", "kPa", "MPa", "GPa",
            "J", "W", "mW", "kW", "MW", "K", "degC", "delta_degC", "A", "mA", "kA", "C", "V", "mV",
            "kV", "MV", "ohm", "kohm", "Mohm", "F", "H", "S", "T", "mT", "rad", "deg", "rpm", "B",
            "KiB", "MiB", "GiB", "nF", "uH",
        ];
        for symbol in symbols {
            assert!(parse_unit_expression(symbol).is_ok(), "{symbol} does not parse");
            assert!(unit_name(symbol).is_some(), "{symbol} has no name");
        }
        assert_eq!(unit_name("mm").as_deref(), Some("millimeter"));
        assert_eq!(unit_name("kohm").as_deref(), Some("kiloohm"));
        assert_eq!(unit_name("MPa").as_deref(), Some("megapascal"));
        assert_eq!(unit_name("xyz"), None);
    }

    #[test]
    fn facts_give_kind_and_si_scale() {
        let mm = unit_facts("mm").unwrap();
        assert_eq!(mm.kind, Some("length"));
        assert_eq!(mm.si, Some("m"));
        assert!((mm.scale - 1e-3).abs() < 1e-15);
        let density = unit_facts("kg/m^3").unwrap();
        assert_eq!(density.kind, Some("mass per volume"));
        assert_eq!(density.name, None);
    }
}
