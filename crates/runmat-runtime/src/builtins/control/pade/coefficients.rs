use super::{checkpoint, numerical, BuiltinResult, RuntimeContext};

/// Fill descending-power coefficients with monic denominator normalization.
/// The ratio is formed in double precision even for native single output;
/// casting the final coefficient is separately checked by the caller.
pub(super) fn visit(
    delay: f64,
    order: usize,
    runtime: Option<&RuntimeContext>,
    mut put: impl FnMut(usize, f64, f64) -> BuiltinResult<()>,
) -> BuiltinResult<()> {
    if delay == 0.0 {
        checkpoint(runtime)?;
        return put(0, 1.0, 1.0);
    }
    let mut coefficient = 1.0_f64;
    for index in 0..=order {
        if index % 1024 == 0 {
            checkpoint(runtime)?;
        }
        let power = order - index;
        put(
            index,
            if power.is_multiple_of(2) {
                coefficient
            } else {
                -coefficient
            },
            coefficient,
        )?;
        if power != 0 {
            let k = power as f64;
            let n = order as f64;
            let ratio = (k / (n - k + 1.0)) * ((2.0 * n - k + 1.0) / delay);
            coefficient *= ratio;
            if !coefficient.is_finite() {
                return Err(numerical(
                    "normalized Padé coefficient exceeds the floating output domain",
                ));
            }
        }
    }
    checkpoint(runtime)
}
