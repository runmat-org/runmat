//! Same existing integer-tree branch owner, with original source validation retained.
use super::*;
/// Any finite fractional integer coordinate defines a complete disjunction.
/// Branches use exactly represented consecutive integer thresholds. Beyond
/// binary64's consecutive-integer range, stop rather than create a nonshrinking
/// or incomplete partition.
pub(super) fn numeric_branch(
    proposal: &[f64],
    integers: &[usize],
    mut progresses: impl FnMut(usize, f64, f64) -> bool,
) -> Option<(usize, f64, f64)> {
    let mut selected = None;
    let mut score = 0.0_f64;
    for &j in integers {
        let value = *proposal.get(j)?;
        if !value.is_finite() {
            return None;
        }
        let floor = value.floor();
        let ceil = value.ceil();
        if floor == ceil {
            continue;
        }
        if ceil - floor != 1.0 || floor + 1.0 != ceil || ceil - 1.0 != floor {
            return None;
        }
        if !progresses(j, floor, ceil) {
            continue;
        }
        let fraction = (value - floor).min(ceil - value);
        if fraction > score {
            score = fraction;
            selected = Some((j, floor, ceil));
        }
    }
    selected
}
pub(in crate::builtins::math::optim::intlinprog) async fn exact_branch(
    point: &[BigRational],
    integers: &[usize],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<(usize, f64, f64)>> {
    use num_traits::{ToPrimitive, Zero};
    let mut selected = None;
    let mut score = BigRational::zero();
    for &column in integers {
        let Some(value) = point.get(column) else {
            return Ok(None);
        };
        if value.is_integer() {
            continue;
        }
        let floor = value.floor();
        let ceil = value.ceil();
        if let Some((lower, upper)) = floor.to_f64().zip(ceil.to_f64()) {
            // Exact round trips, not a flintmax/domain cutoff. A different
            // representable disjunction remains usable if this one is not.
            if BigRational::from_float(lower).as_ref() == Some(&floor)
                && BigRational::from_float(upper).as_ref() == Some(&ceil)
            {
                let fraction = (value - &floor).min(&ceil - value);
                if fraction > score {
                    score = fraction;
                    selected = Some((column, lower, upper));
                }
            }
        }
        cooperation.completed(1).await?;
    }
    Ok(selected)
}
