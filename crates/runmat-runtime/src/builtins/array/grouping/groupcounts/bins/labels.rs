use runmat_value::NumericScalar;

use crate::builtins::array::grouping::keys::format_integer;

pub(super) fn interval(
    lower: NumericScalar,
    upper: NumericScalar,
    index: usize,
    count: usize,
    included_right: bool,
) -> String {
    let number = |value: NumericScalar| {
        value
            .into_int_value()
            .as_ref()
            .map(format_integer)
            .unwrap_or_else(|| format_number(value.materialize_f64()))
    };
    bounds(number(lower), number(upper), index, count, included_right)
}

pub(super) fn floating_interval(
    lower: f64,
    upper: f64,
    index: usize,
    count: usize,
    included_right: bool,
) -> String {
    bounds(
        format_number(lower),
        format_number(upper),
        index,
        count,
        included_right,
    )
}

fn bounds(
    lower: String,
    upper: String,
    index: usize,
    count: usize,
    included_right: bool,
) -> String {
    if included_right {
        format!("{}{lower}, {upper}]", if index == 0 { "[" } else { "(" })
    } else {
        format!(
            "[{lower}, {upper}{}",
            if index + 1 == count { "]" } else { ")" }
        )
    }
}

fn format_number(value: f64) -> String {
    if value.fract() == 0.0 {
        format!("{value:.0}")
    } else {
        value.to_string()
    }
}
