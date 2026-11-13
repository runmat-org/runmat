//! Choose double endpoints enclosing extrema compared in their original types.
use crate::{builtins::array::binning::numeric::compare, BuiltinResult};
use runmat_value::NumericScalar;
use std::cmp::Ordering;

pub(super) fn enclosing_bounds(values: &[NumericScalar]) -> BuiltinResult<(f64, f64)> {
    let mut finite = values
        .iter()
        .copied()
        .filter(|value| value.materialize_f64().is_finite());
    let first = finite.next().ok_or_else(|| {
        super::error::invalid("discretize: cannot infer equal-width bins from all-missing data")
    })?;
    let (minimum, maximum) = finite.fold((first, first), |(minimum, maximum), value| {
        (
            if compare(value, minimum) == Some(Ordering::Less) {
                value
            } else {
                minimum
            },
            if compare(value, maximum) == Some(Ordering::Greater) {
                value
            } else {
                maximum
            },
        )
    });
    let mut lower = minimum.materialize_f64();
    let mut upper = maximum.materialize_f64();
    // Converting a wide integer to double can move either endpoint inward.
    // Compare against the typed original, then expand by one representable step.
    if compare(NumericScalar::F64(lower), minimum) == Some(Ordering::Greater) {
        lower = lower.next_down();
    }
    if compare(NumericScalar::F64(upper), maximum) == Some(Ordering::Less) {
        upper = upper.next_up();
    }
    if lower == upper {
        let centre = lower;
        lower = centre - 0.5;
        upper = centre + 0.5;
        if lower == centre {
            lower = centre.next_down();
        }
        if upper == centre {
            upper = centre.next_up();
        }
        // Finite extrema need finite interpolation endpoints. At either
        // representable limit the centre itself closes that side of the range.
        if !lower.is_finite() {
            lower = centre;
        }
        if !upper.is_finite() {
            upper = centre;
        }
    }
    Ok((lower, upper))
}
