//! The units a drawing's dimensions print in, which its sheet format sets (`units = ft_in`).
//!
//! - `mm` (the default): millimetres with no unit, trailing zeros dropped (`80`, `12.5`).
//! - `inch`: decimal inches with the inch mark, to three places (`7.75"`).
//! - `ft_in`: feet and inches as building drawings write them, to the nearest 1/16 in: under a
//!   foot in inches (`7 3/4"`), else feet and inches (`9'-1 1/8"`, `40'-0"`). The title block's
//!   scale reads as inches to the foot (`1/4" = 1'-0"` for 1:48; `FULL SIZE` for 1:1).

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DrawingUnits {
    #[default]
    Mm,
    Inch,
    FtIn,
}

impl DrawingUnits {
    pub fn is_mm(&self) -> bool {
        *self == DrawingUnits::Mm
    }

    /// A length (mm) as a dimension prints it.
    pub fn length(self, mm: f64) -> String {
        match self {
            DrawingUnits::Mm => decimal(mm, 2),
            DrawingUnits::Inch => format!("{}\"", decimal(mm / 25.4, 3)),
            DrawingUnits::FtIn => feet_inches(mm / 25.4),
        }
    }

    /// A height (mm) as a level mark prints it: as [`Self::length`], except that feet and
    /// inches always carry the feet (`0'-0"`, `-0'-8"`, `10'-1 3/4"`), as elevations are written.
    pub fn elevation(self, mm: f64) -> String {
        match self {
            DrawingUnits::FtIn => {
                let inches = mm / 25.4;
                let sixteenths = (inches.abs() * 16.0).round() as u64;
                let sign = if inches < 0.0 && sixteenths > 0 { "-" } else { "" };
                format!("{sign}{}'-{}\"", sixteenths / 192, fraction(sixteenths % 192))
            }
            other => other.length(mm),
        }
    }

    /// A sheet scale `numerator : denominator` as the title block prints it.
    pub fn scale(self, (n, d): (u32, u32)) -> String {
        if self != DrawingUnits::Mm && n > 0 && n == d {
            return "FULL SIZE".to_owned();
        }
        if self == DrawingUnits::FtIn && n > 0 && d > 0 {
            // Inches on the sheet to a foot of the model.
            let sixteenths = (12.0 * 16.0 * n as f64 / d as f64).round() as u64;
            if sixteenths > 0 && sixteenths % 192 != 0 {
                return format!("{}\" = 1'-0\"", fraction(sixteenths));
            }
            if sixteenths > 0 {
                return format!("1'-0\" = {}'-0\"", d / n.max(1));
            }
        }
        format!("{n}:{d}")
    }
}

fn decimal(v: f64, places: i32) -> String {
    let k = 10f64.powi(places);
    let s = format!("{:.*}", places as usize, (v * k).round() / k);
    s.trim_end_matches('0').trim_end_matches('.').to_owned()
}

/// Sixteenths of an inch as whole inches and a reduced fraction (`7 3/4`, `1/2`, `12`).
fn fraction(sixteenths: u64) -> String {
    let (whole, mut num) = (sixteenths / 16, sixteenths % 16);
    let mut den = 16;
    while num > 0 && num % 2 == 0 {
        num /= 2;
        den /= 2;
    }
    match (whole, num) {
        (w, 0) => w.to_string(),
        (0, n) => format!("{n}/{den}"),
        (w, n) => format!("{w} {n}/{den}"),
    }
}

fn feet_inches(inches: f64) -> String {
    let sign = if inches < 0.0 { "-" } else { "" };
    let sixteenths = (inches.abs() * 16.0).round() as u64;
    if sixteenths < 12 * 16 {
        return format!("{sign}{}\"", fraction(sixteenths));
    }
    let (feet, rest) = (sixteenths / 192, sixteenths % 192);
    format!("{sign}{feet}'-{}\"", fraction(rest))
}

#[cfg(test)]
mod tests {
    use super::DrawingUnits::*;

    #[test]
    fn feet_and_inches_print_as_building_drawings_write_them() {
        assert_eq!(FtIn.length(7.75 * 25.4), "7 3/4\"");
        assert_eq!(FtIn.length(109.125 * 25.4), "9'-1 1/8\"");
        assert_eq!(FtIn.length(480.0 * 25.4), "40'-0\"");
        assert_eq!(FtIn.length(0.5 * 25.4), "1/2\"");
        // To the nearest sixteenth, carrying into the next foot.
        assert_eq!(FtIn.length(143.99 * 25.4), "12'-0\"");
        assert_eq!(Inch.length(7.75 * 25.4), "7.75\"");
        assert_eq!(Mm.length(12.5), "12.5");
    }

    #[test]
    fn an_elevation_always_carries_its_feet() {
        assert_eq!(FtIn.elevation(0.0), "0'-0\"");
        assert_eq!(FtIn.elevation(-8.0 * 25.4), "-0'-8\"");
        assert_eq!(FtIn.elevation(-22.0 * 25.4), "-1'-10\"");
        assert_eq!(FtIn.elevation(121.75 * 25.4), "10'-1 3/4\"");
        assert_eq!(super::DrawingUnits::Mm.elevation(3200.0), "3200");
    }

    #[test]
    fn an_imperial_scale_reads_as_inches_to_the_foot() {
        assert_eq!(FtIn.scale((1, 48)), "1/4\" = 1'-0\"");
        assert_eq!(FtIn.scale((1, 4)), "3\" = 1'-0\"");
        assert_eq!(FtIn.scale((1, 96)), "1/8\" = 1'-0\"");
        assert_eq!(FtIn.scale((1, 1)), "FULL SIZE");
        assert_eq!(Mm.scale((1, 48)), "1:48");
    }
}
