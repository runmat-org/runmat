//! Internal helpers shared by the MATLAB-compatible degree-trig builtins
//! (`sind`, `cosd`, `tand`).
//!
//! The reduction routine maps any finite angle (in degrees) into the
//! canonical half-open range `(-180.0, 180.0]`, using a sign convention
//! that makes the exact-value tables in each builtin straightforward to
//! consult. Non-finite inputs are propagated as `None` so callers can
//! emit `NaN` without further bookkeeping.

#[inline]
pub(super) fn reduce_degrees(angle: f64) -> Option<f64> {
    if !angle.is_finite() {
        return None;
    }
    let wrapped = angle.rem_euclid(360.0);
    let reduced = if wrapped > 180.0 {
        wrapped - 360.0
    } else {
        wrapped
    };
    Some(reduced)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduce_degrees_handles_canonical_angles() {
        assert_eq!(reduce_degrees(0.0), Some(0.0));
        assert_eq!(reduce_degrees(30.0), Some(30.0));
        assert_eq!(reduce_degrees(90.0), Some(90.0));
        assert_eq!(reduce_degrees(180.0), Some(180.0));
        assert_eq!(reduce_degrees(-180.0), Some(180.0));
        assert_eq!(reduce_degrees(270.0), Some(-90.0));
        assert_eq!(reduce_degrees(-270.0), Some(90.0));
        assert_eq!(reduce_degrees(360.0), Some(0.0));
        assert_eq!(reduce_degrees(-360.0), Some(0.0));
        assert_eq!(reduce_degrees(720.0), Some(0.0));
        assert_eq!(reduce_degrees(-30.0), Some(-30.0));
    }

    #[test]
    fn reduce_degrees_rejects_nonfinite() {
        assert!(reduce_degrees(f64::NAN).is_none());
        assert!(reduce_degrees(f64::INFINITY).is_none());
        assert!(reduce_degrees(f64::NEG_INFINITY).is_none());
    }
}
