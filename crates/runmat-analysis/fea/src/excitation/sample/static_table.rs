use runmat_analysis_core::{SignalKnot, SignalKnotValue};

use super::{ordinate_value, RealizationError, SampledValue};

/// A static step has no abscissa. A table is unambiguous only when its complete sampled content
/// is constant; choosing an arbitrary knot would silently change a generated history.
pub(super) fn sample_static_table<Q>(
    knots: &[SignalKnot<Q>],
    component: &impl Fn(&Q) -> f64,
) -> Result<SampledValue, RealizationError> {
    if knots.is_empty()
        || knots.iter().any(|knot| !knot.abscissa.is_finite())
        || knots
            .windows(2)
            .any(|pair| pair[0].abscissa >= pair[1].abscissa)
    {
        return Err(RealizationError::new(
            "tabulated knots must have finite, strictly increasing abscissae",
        ));
    }
    let mut sampled = None;
    for knot in knots {
        let SignalKnotValue::Continuous(value) = &knot.value else {
            return Err(RealizationError::new(
                "static excitation cannot select a jump in a tabulated signal",
            ));
        };
        let value = ordinate_value(value, component)?;
        if sampled.as_ref().is_some_and(|previous| previous != &value) {
            return Err(RealizationError::new(
                "static excitation requires constant tabulated values",
            ));
        }
        sampled = Some(value);
    }
    Ok(sampled.expect("nonempty validated table has a sample"))
}
