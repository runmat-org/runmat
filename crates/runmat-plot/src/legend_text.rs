//! Text of a field legend: the range ends at the precision the range needs, and units as they
//! are printed (`degC` as °C, `m^2` as m²).

/// How the numbers of one legend are written, chosen from its range so the two ends differ
/// and carry no digits beyond that.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LegendPrecision {
    /// Fixed point with this many decimals.
    Fixed(usize),
    /// Scientific notation with this many mantissa decimals, for very small or large values.
    Scientific(usize),
}

impl LegendPrecision {
    /// The precision for a legend from `minimum` to `maximum`: two significant digits of the
    /// span, more only when the ends would otherwise print the same.
    pub fn for_range(minimum: f64, maximum: f64) -> Self {
        let magnitude = minimum.abs().max(maximum.abs());
        if !magnitude.is_finite() || magnitude == 0.0 {
            return Self::Fixed(0);
        }
        let span = (maximum - minimum).abs();
        // A flat field has no span to resolve: three significant digits of its value.
        let resolution = if span > 0.0 && span.is_finite() {
            span
        } else {
            magnitude * 1e-3
        };
        let scientific = !(1e-3..1e5).contains(&magnitude);
        let base = if scientific {
            let exponent = magnitude.log10().floor() as i32;
            Self::Scientific(
                (exponent - resolution.log10().floor() as i32 + 1).clamp(0, 12) as usize,
            )
        } else {
            Self::Fixed((1 - resolution.log10().floor() as i32).clamp(0, 12) as usize)
        };
        let mut precision = base;
        for _ in 0..6 {
            if span == 0.0 || precision.format(minimum) != precision.format(maximum) {
                break;
            }
            precision = precision.finer();
        }
        precision
    }

    /// The precision that writes the larger end of `minimum` to `maximum` with `digits`
    /// significant digits; both ends share it.
    pub fn significant(minimum: f64, maximum: f64, digits: u8) -> Self {
        let digits = i32::from(digits.max(1));
        let magnitude = minimum.abs().max(maximum.abs());
        if !magnitude.is_finite() || magnitude == 0.0 {
            return Self::Fixed(0);
        }
        if !(1e-3..1e5).contains(&magnitude) {
            return Self::Scientific((digits - 1) as usize);
        }
        Self::Fixed((digits - 1 - magnitude.log10().floor() as i32).max(0) as usize)
    }

    fn finer(self) -> Self {
        match self {
            Self::Fixed(decimals) => Self::Fixed(decimals + 1),
            Self::Scientific(decimals) => Self::Scientific(decimals + 1),
        }
    }

    /// `value` at this precision; zero prints as `0`.
    pub fn format(self, value: f64) -> String {
        if value == 0.0 {
            return "0".into();
        }
        let text = match self {
            Self::Fixed(decimals) => format!("{value:.decimals$}"),
            Self::Scientific(decimals) => format!("{value:.decimals$e}"),
        };
        // Rounding can leave "-0.0"; the sign of a printed zero means nothing.
        if text
            .trim_start_matches('-')
            .chars()
            .all(|c| c == '0' || c == '.')
        {
            text.trim_start_matches('-').to_string()
        } else {
            text
        }
    }
}

/// A value stated by the author (a limit), with no trailing zeros: `85`, `0.25`, `1.5e-4`.
pub fn format_stated(value: f64) -> String {
    if value == 0.0 {
        return "0".into();
    }
    let magnitude = value.abs();
    if !(1e-3..1e6).contains(&magnitude) {
        let text = format!("{value:.6e}");
        let (mantissa, exponent) = text.split_once('e').unwrap_or((&text, "0"));
        let mantissa = mantissa.trim_end_matches('0').trim_end_matches('.');
        return format!("{mantissa}e{exponent}");
    }
    let text = format!("{value:.6}");
    text.trim_end_matches('0').trim_end_matches('.').to_string()
}

/// `value` with `digits` significant digits: `17.7`, `184`, `0.00412`, `1.25e8`.
pub fn format_significant(value: f64, digits: u8) -> String {
    LegendPrecision::significant(value, value, digits).format(value)
}

/// The overflow line of a fixed legend range that the field leaves: its true highest value
/// above the range (`above`) or its lowest below, such as "14.2 MPa above the range". The
/// legend draws [`OVERFLOW_ABOVE_MARK`] or [`OVERFLOW_BELOW_MARK`] before it.
pub fn overflow_line(value: f64, unit: &str, above: bool, precision: LegendPrecision) -> String {
    let side = if above { "above" } else { "below" };
    format!(
        "{} {side} the range",
        with_unit(precision.format(value), unit)
    )
}

/// Drawn before an overflow line above the range.
pub const OVERFLOW_ABOVE_MARK: char = '▲';
/// Drawn before an overflow line below the range.
pub const OVERFLOW_BELOW_MARK: char = '▼';

/// A study's acceptance criterion on the shown quantity and the run's result against it, in
/// the legend's unit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AssessedResult<'a> {
    /// A maximum (the result is a peak) or a minimum (the result is the lowest value).
    pub maximum: bool,
    /// The assessed result; `None` when the run has none.
    pub measured: Option<f64>,
    pub limit: f64,
    pub unit: &'a str,
    /// Whether the limit over the result is a meaningful factor: a quantity whose zero is
    /// physical (a stress, a displacement). A temperature states its margin instead.
    pub ratio: bool,
    /// Significant digits of the result.
    pub digits: u8,
}

/// The legend's assessment line, result first: "Peak 17.7 MPa · limit 184 MPa · factor 10.4",
/// "Lowest 2.10 mm · limit 1 mm · factor 2.10", or with a temperature "Peak 63.4 °C · limit
/// 85 °C · margin 21.6 K" ("… · 5.00 K over" when the result breaks the limit). Without a
/// result, "Limit 184 MPa".
pub fn assessment_line(result: &AssessedResult<'_>) -> String {
    let limit = with_unit(format_stated(result.limit), result.unit);
    let Some(measured) = result.measured.filter(|value| value.is_finite()) else {
        return format!("Limit {limit}");
    };
    let lead = if result.maximum { "Peak" } else { "Lowest" };
    let shown = with_unit(format_significant(measured, result.digits), result.unit);
    let (numerator, denominator) = if result.maximum {
        (result.limit, measured)
    } else {
        (measured, result.limit)
    };
    let factor = numerator / denominator;
    let tail = if result.ratio && factor.is_finite() && factor > 0.0 {
        format!("factor {}", format_significant(factor, 3))
    } else {
        let margin = if result.maximum {
            result.limit - measured
        } else {
            measured - result.limit
        };
        // A difference of absolute temperatures is a temperature difference.
        let difference = match result.unit {
            "degC" => "K",
            other => other,
        };
        let amount = with_unit(format_significant(margin.abs(), 3), difference);
        match (margin >= 0.0, result.maximum) {
            (true, _) => format!("margin {amount}"),
            (false, true) => format!("{amount} over"),
            (false, false) => format!("{amount} under"),
        }
    };
    format!("{lead} {shown} · limit {limit} · {tail}")
}

/// A unit symbol as printed on a legend: `degC` as °C, `degF` as °F, `um` as µm and the
/// exponents 2 and 3 as superscripts.
pub fn printed_unit(unit: &str) -> String {
    let mut text = match unit {
        "degC" => return "°C".into(),
        "degF" => return "°F".into(),
        "delta_degC" => return "K".into(),
        _ => unit.to_string(),
    };
    if text == "um" || text.starts_with("um/") || text.starts_with("um^") {
        text.replace_range(0..2, "µm");
    }
    text.replace("^2", "²").replace("^3", "³")
}

/// `value` and its printed unit, separated by a space unless the unit is empty.
pub fn with_unit(number: String, unit: &str) -> String {
    let unit = printed_unit(unit);
    if unit.is_empty() {
        number
    } else {
        format!("{number} {unit}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ends(minimum: f64, maximum: f64) -> (String, String) {
        let precision = LegendPrecision::for_range(minimum, maximum);
        (precision.format(minimum), precision.format(maximum))
    }

    #[test]
    fn range_precision_distinguishes_the_ends_without_extra_digits() {
        // The six-bolt busbar in °C.
        assert_eq!(
            ends(56.8142170922981, 63.3973805835758),
            ("56.8".into(), "63.4".into())
        );
        assert_eq!(
            ends(329.9642170922981, 336.5473805835758),
            ("330.0".into(), "336.5".into())
        );
        assert_eq!(ends(8.6996, 1171.548), ("9".into(), "1172".into()));
        assert_eq!(ends(0.0, 0.25), ("0".into(), "0.25".into()));
        // A narrow band far from zero still prints two different ends.
        assert_eq!(
            ends(300.001, 300.004),
            ("300.0010".into(), "300.0040".into())
        );
        assert_eq!(ends(0.0, 1.234e-5), ("0".into(), "1.2e-5".into()));
        assert_eq!(ends(2.1e7, 3.38e8), ("2.1e7".into(), "3.4e8".into()));
        assert_eq!(ends(0.0, 1.0), ("0".into(), "1.0".into()));
        let flat = LegendPrecision::for_range(293.15, 293.15);
        assert_eq!(flat.format(293.15), "293.15");
    }

    #[test]
    fn significant_digits_and_overflow_lines() {
        assert_eq!(format_significant(17.7349, 3), "17.7");
        assert_eq!(format_significant(184.0, 3), "184");
        assert_eq!(format_significant(0.0041234, 3), "0.00412");
        assert_eq!(format_significant(1.25e8, 3), "1.25e8");
        let precision = LegendPrecision::significant(0.0, 12.0, 3);
        assert_eq!(precision, LegendPrecision::Fixed(1));
        assert_eq!(
            (precision.format(0.0), precision.format(12.0)),
            ("0".into(), "12.0".into())
        );
        assert_eq!(
            overflow_line(14.2, "MPa", true, precision),
            "14.2 MPa above the range"
        );
        assert_eq!(
            overflow_line(-3.0, "degC", false, precision),
            "-3.0 °C below the range"
        );
    }

    #[test]
    fn the_assessment_line_leads_with_the_result() {
        let stress = AssessedResult {
            maximum: true,
            measured: Some(17.7),
            limit: 184.0,
            unit: "MPa",
            ratio: true,
            digits: 3,
        };
        assert_eq!(
            assessment_line(&stress),
            "Peak 17.7 MPa · limit 184 MPa · factor 10.4"
        );
        let broken = AssessedResult {
            measured: Some(200.0),
            ..stress
        };
        assert_eq!(
            assessment_line(&broken),
            "Peak 200 MPa · limit 184 MPa · factor 0.920"
        );
        let gap = AssessedResult {
            maximum: false,
            measured: Some(2.1),
            limit: 1.0,
            unit: "mm",
            ..stress
        };
        assert_eq!(
            assessment_line(&gap),
            "Lowest 2.10 mm · limit 1 mm · factor 2.10"
        );
        let busbar = AssessedResult {
            maximum: true,
            measured: Some(63.3973805835758),
            limit: 85.0,
            unit: "degC",
            ratio: false,
            digits: 3,
        };
        assert_eq!(
            assessment_line(&busbar),
            "Peak 63.4 °C · limit 85 °C · margin 21.6 K"
        );
        let hot = AssessedResult {
            measured: Some(90.0),
            ..busbar
        };
        assert_eq!(
            assessment_line(&hot),
            "Peak 90.0 °C · limit 85 °C · 5.00 K over"
        );
        let pending = AssessedResult {
            measured: None,
            ..stress
        };
        assert_eq!(assessment_line(&pending), "Limit 184 MPa");
    }

    #[test]
    fn stated_values_and_units_print_as_written() {
        assert_eq!(format_stated(85.0), "85");
        assert_eq!(format_stated(0.25), "0.25");
        assert_eq!(format_stated(2.5e8), "2.5e8");
        assert_eq!(printed_unit("degC"), "°C");
        assert_eq!(printed_unit("W/m^2"), "W/m²");
        assert_eq!(printed_unit("um"), "µm");
        assert_eq!(with_unit("56.8".into(), "degC"), "56.8 °C");
        assert_eq!(with_unit("1.0".into(), ""), "1.0");
    }
}
