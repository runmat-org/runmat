//! Attribute values, physical dimensions and the unit table used by numeric comparisons.
//!
//! Numbers are held in SI base units with a dimension vector, so `[width>=0.2mm]` compares an
//! item whose width is stored as `0.00025 m` correctly, and `[width>=2V]` is a dimension
//! mismatch rather than a silent false.
//!
//! The unit table here is a seam: at M1 the `.fea` unit registry becomes the one authority and
//! is supplied through [`UnitResolver`]. [`BuiltinUnits`] covers the electrical and mechanical
//! units the circuit rules need until then.

use std::cmp::Ordering;
use std::fmt;

/// Exponents of the SI base dimensions: metre, kilogram, second, ampere, kelvin, mole, candela.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Dim(pub [i8; 7]);

impl Dim {
    pub const NONE: Dim = Dim([0; 7]);
    pub const LENGTH: Dim = Dim([1, 0, 0, 0, 0, 0, 0]);
    pub const MASS: Dim = Dim([0, 1, 0, 0, 0, 0, 0]);
    pub const TIME: Dim = Dim([0, 0, 1, 0, 0, 0, 0]);
    pub const CURRENT: Dim = Dim([0, 0, 0, 1, 0, 0, 0]);
    pub const TEMPERATURE: Dim = Dim([0, 0, 0, 0, 1, 0, 0]);
    pub const FREQUENCY: Dim = Dim([0, 0, -1, 0, 0, 0, 0]);
    pub const VOLTAGE: Dim = Dim([2, 1, -3, -1, 0, 0, 0]);
    pub const RESISTANCE: Dim = Dim([2, 1, -3, -2, 0, 0, 0]);
    pub const CONDUCTANCE: Dim = Dim([-2, -1, 3, 2, 0, 0, 0]);
    pub const CAPACITANCE: Dim = Dim([-2, -1, 4, 2, 0, 0, 0]);
    pub const INDUCTANCE: Dim = Dim([2, 1, -2, -2, 0, 0, 0]);
    pub const POWER: Dim = Dim([2, 1, -3, 0, 0, 0, 0]);
    pub const ENERGY: Dim = Dim([2, 1, -2, 0, 0, 0, 0]);
    pub const CHARGE: Dim = Dim([0, 0, 1, 1, 0, 0, 0]);
    pub const FORCE: Dim = Dim([1, 1, -2, 0, 0, 0, 0]);
    pub const PRESSURE: Dim = Dim([-1, 1, -2, 0, 0, 0, 0]);
    pub const AREA: Dim = Dim([2, 0, 0, 0, 0, 0, 0]);
    /// Thermal resistance, K/W (used by θJC/θJB fields).
    pub const THERMAL_RESISTANCE: Dim = Dim([-2, -1, 3, 0, 1, 0, 0]);
    /// Thermal conductivity, W/(m·K).
    pub const THERMAL_CONDUCTIVITY: Dim = Dim([1, 1, -3, 0, -1, 0, 0]);

    pub fn is_none(self) -> bool {
        self == Dim::NONE
    }
    /// Dimension of a product (exponents add).
    pub fn times(self, o: Dim) -> Dim {
        let mut r = [0i8; 7];
        for (i, v) in r.iter_mut().enumerate() {
            *v = self.0[i] + o.0[i];
        }
        Dim(r)
    }
    /// Dimension of a quotient (exponents subtract).
    pub fn per(self, o: Dim) -> Dim {
        let mut r = [0i8; 7];
        for (i, v) in r.iter_mut().enumerate() {
            *v = self.0[i] - o.0[i];
        }
        Dim(r)
    }
}

impl fmt::Display for Dim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const NAMES: [&str; 7] = ["m", "kg", "s", "A", "K", "mol", "cd"];
        if self.is_none() {
            return f.write_str("1");
        }
        let mut first = true;
        for (i, e) in self.0.iter().enumerate() {
            if *e == 0 {
                continue;
            }
            if !first {
                f.write_str("·")?;
            }
            first = false;
            if *e == 1 {
                write!(f, "{}", NAMES[i])?;
            } else {
                write!(f, "{}^{}", NAMES[i], e)?;
            }
        }
        Ok(())
    }
}

/// A number in SI base units with its dimension.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quantity {
    pub value: f64,
    pub dim: Dim,
}

impl Quantity {
    pub fn new(value: f64, dim: Dim) -> Self {
        Quantity { value, dim }
    }
    pub fn scalar(value: f64) -> Self {
        Quantity {
            value,
            dim: Dim::NONE,
        }
    }
    pub fn length_m(value: f64) -> Self {
        Quantity::new(value, Dim::LENGTH)
    }
    pub fn length_mm(value: f64) -> Self {
        Quantity::new(value * 1e-3, Dim::LENGTH)
    }
    /// Relative-tolerance equality used by `=` on numbers (1e-9 relative, 1e-18 absolute).
    pub fn approx_eq(&self, o: &Quantity) -> bool {
        self.dim == o.dim
            && (self.value - o.value).abs() <= 1e-9 * self.value.abs().max(o.value.abs()) + 1e-18
    }
    /// Ordering for numeric comparison; `None` on a dimension mismatch or NaN.
    pub fn compare(&self, o: &Quantity) -> Option<Ordering> {
        if self.dim != o.dim {
            return None;
        }
        if self.approx_eq(o) {
            return Some(Ordering::Equal);
        }
        self.value.partial_cmp(&o.value)
    }
}

impl fmt::Display for Quantity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.dim.is_none() {
            write!(f, "{}", self.value)
        } else {
            write!(f, "{} {}", self.value, self.dim)
        }
    }
}

/// The value of an item attribute, as the item model reports it.
#[derive(Clone, Debug, PartialEq)]
pub enum AttrValue {
    Str(String),
    Num(Quantity),
    Bool(bool),
    /// A set-valued attribute such as the layers of a through-hole pad; `~=` tests membership.
    List(Vec<AttrValue>),
}

impl AttrValue {
    pub fn str(s: impl Into<String>) -> Self {
        AttrValue::Str(s.into())
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            AttrValue::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_quantity(&self) -> Option<Quantity> {
        match self {
            AttrValue::Num(q) => Some(*q),
            _ => None,
        }
    }
    /// Truthiness for presence tests (`[locked]`): everything except `Bool(false)` and empty lists.
    pub fn truthy(&self) -> bool {
        match self {
            AttrValue::Bool(b) => *b,
            AttrValue::List(l) => !l.is_empty(),
            _ => true,
        }
    }
    /// Text form used by string operators on non-string values.
    pub fn to_text(&self) -> String {
        match self {
            AttrValue::Str(s) => s.clone(),
            AttrValue::Num(q) => format!("{}", q.value),
            AttrValue::Bool(b) => b.to_string(),
            AttrValue::List(l) => l.iter().map(|v| v.to_text()).collect::<Vec<_>>().join(" "),
        }
    }
}

/// One unit: multiply by `scale` then add `offset` to get SI base units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UnitDef {
    pub scale: f64,
    pub offset: f64,
    pub dim: Dim,
}

/// Resolves a unit symbol (`mm`, `kohm`, `uF`, `mil`) to its SI conversion.
pub trait UnitResolver {
    fn resolve(&self, symbol: &str) -> Option<UnitDef>;
}

/// Built-in table: SI symbols with prefixes, plus `mil`, `in`, `thou`, `deg`, `rad`, `%`,
/// `ppm`, `degC`, `oz` (copper weight as thickness: 1 oz = 34.79 µm), `ohm`/`Ω`, `K/W`, `W/mK`.
#[derive(Clone, Copy, Debug, Default)]
pub struct BuiltinUnits;

const PREFIXES: &[(&str, f64)] = &[
    ("T", 1e12),
    ("G", 1e9),
    ("M", 1e6),
    ("k", 1e3),
    ("m", 1e-3),
    ("u", 1e-6),
    ("µ", 1e-6),
    ("μ", 1e-6),
    ("n", 1e-9),
    ("p", 1e-12),
    ("f", 1e-15),
];

const BASE: &[(&str, f64, Dim)] = &[
    ("m", 1.0, Dim::LENGTH),
    ("g", 1e-3, Dim::MASS),
    ("s", 1.0, Dim::TIME),
    ("A", 1.0, Dim::CURRENT),
    ("K", 1.0, Dim::TEMPERATURE),
    ("V", 1.0, Dim::VOLTAGE),
    ("ohm", 1.0, Dim::RESISTANCE),
    ("Ohm", 1.0, Dim::RESISTANCE),
    ("Ω", 1.0, Dim::RESISTANCE),
    ("S", 1.0, Dim::CONDUCTANCE),
    ("F", 1.0, Dim::CAPACITANCE),
    ("H", 1.0, Dim::INDUCTANCE),
    ("Hz", 1.0, Dim::FREQUENCY),
    ("W", 1.0, Dim::POWER),
    ("J", 1.0, Dim::ENERGY),
    ("C", 1.0, Dim::CHARGE),
    ("N", 1.0, Dim::FORCE),
    ("Pa", 1.0, Dim::PRESSURE),
];

impl UnitResolver for BuiltinUnits {
    fn resolve(&self, symbol: &str) -> Option<UnitDef> {
        let plain = |scale: f64, dim: Dim| {
            Some(UnitDef {
                scale,
                offset: 0.0,
                dim,
            })
        };
        match symbol {
            "" => return plain(1.0, Dim::NONE),
            "mil" | "thou" => return plain(25.4e-6, Dim::LENGTH),
            "in" | "inch" => return plain(25.4e-3, Dim::LENGTH),
            "deg" | "°" => return plain(std::f64::consts::PI / 180.0, Dim::NONE),
            "rad" => return plain(1.0, Dim::NONE),
            "%" => return plain(0.01, Dim::NONE),
            "ppm" => return plain(1e-6, Dim::NONE),
            "oz" => return plain(34.79e-6, Dim::LENGTH),
            "degC" | "°C" => {
                return Some(UnitDef {
                    scale: 1.0,
                    offset: 273.15,
                    dim: Dim::TEMPERATURE,
                })
            }
            "K/W" => return plain(1.0, Dim::THERMAL_RESISTANCE),
            "W/mK" | "W/(m*K)" | "W/m/K" => return plain(1.0, Dim::THERMAL_CONDUCTIVITY),
            "mm2" | "mm^2" => return plain(1e-6, Dim::AREA),
            _ => {}
        }
        // Exact base symbol first so "m" is metre, not milli-nothing.
        for (b, s, d) in BASE {
            if symbol == *b {
                return plain(*s, *d);
            }
        }
        for (p, ps) in PREFIXES {
            if let Some(rest) = symbol.strip_prefix(p) {
                for (b, s, d) in BASE {
                    if rest == *b {
                        return plain(ps * s, *d);
                    }
                }
            }
        }
        None
    }
}

/// Parses `0.2mm`, `10 kohm`, `-3.3V`, `1e-3`, `25degC` into a quantity.
pub fn parse_quantity(text: &str, units: &dyn UnitResolver) -> Result<Quantity, String> {
    let t = text.trim();
    let bytes = t.as_bytes();
    let mut end = 0;
    let mut seen_digit = false;
    while end < bytes.len() {
        let c = bytes[end] as char;
        let ok = c.is_ascii_digit()
            || c == '.'
            || ((c == '-' || c == '+') && (end == 0 || matches!(bytes[end - 1], b'e' | b'E')))
            || ((c == 'e' || c == 'E')
                && seen_digit
                && end + 1 < bytes.len()
                && (bytes[end + 1].is_ascii_digit() || matches!(bytes[end + 1], b'-' | b'+')));
        if !ok {
            break;
        }
        if c.is_ascii_digit() {
            seen_digit = true;
        }
        end += 1;
    }
    if !seen_digit {
        return Err(format!("`{t}` is not a number"));
    }
    let num: f64 = t[..end]
        .parse()
        .map_err(|_| format!("`{}` is not a number", &t[..end]))?;
    let unit = t[end..].trim();
    let def = units
        .resolve(unit)
        .ok_or_else(|| format!("unknown unit `{unit}`"))?;
    Ok(Quantity::new(num * def.scale + def.offset, def.dim))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_units() {
        let u = BuiltinUnits;
        let q = parse_quantity("0.2mm", &u).unwrap();
        assert!((q.value - 0.0002).abs() < 1e-15 && q.dim == Dim::LENGTH);
        let q = parse_quantity("10 kohm", &u).unwrap();
        assert_eq!(q.value, 10_000.0);
        assert_eq!(q.dim, Dim::RESISTANCE);
        let q = parse_quantity("4.7uF", &u).unwrap();
        assert!((q.value - 4.7e-6).abs() < 1e-18);
        let q = parse_quantity("8mil", &u).unwrap();
        assert!((q.value - 8.0 * 25.4e-6).abs() < 1e-15);
        let q = parse_quantity("25degC", &u).unwrap();
        assert!((q.value - 298.15).abs() < 1e-9);
        let q = parse_quantity("1e-3", &u).unwrap();
        assert_eq!(q.value, 1e-3);
        assert!(parse_quantity("abc", &u).is_err());
        assert!(parse_quantity("3 furlongs", &u).is_err());
        assert_eq!(parse_quantity("1m", &u).unwrap().dim, Dim::LENGTH);
        assert_eq!(parse_quantity("1mV", &u).unwrap().value, 1e-3);
    }

    #[test]
    fn compare_checks_dimension() {
        let a = Quantity::length_mm(0.2);
        let b = Quantity::length_mm(0.25);
        assert_eq!(a.compare(&b), Some(Ordering::Less));
        assert_eq!(a.compare(&Quantity::new(0.2, Dim::VOLTAGE)), None);
        assert!(Quantity::length_mm(0.1 + 0.2).approx_eq(&Quantity::length_mm(0.3)));
    }
}
