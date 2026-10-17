//! IEEE binary32/binary64 spacing at the magnitude of a represented value.
//! The top finite binade keeps its finite spacing even at `realmax`.

/// Positive distance from `abs(value)` to the next binary64 number.
pub fn spacing_f64(value: f64) -> f64 {
    let magnitude = value.to_bits() & 0x7fff_ffff_ffff_ffff;
    let exponent = (magnitude >> 52) & 0x7ff;
    if exponent == 0x7ff {
        return f64::NAN;
    }
    if exponent == 0 {
        return f64::from_bits(1);
    }
    if exponent <= 52 {
        f64::from_bits(1_u64 << (exponent - 1))
    } else {
        f64::from_bits((exponent - 52) << 52)
    }
}

/// Positive distance from `abs(value)` to the next binary32 number.
pub fn spacing_f32(value: f32) -> f32 {
    let magnitude = value.to_bits() & 0x7fff_ffff;
    let exponent = (magnitude >> 23) & 0xff;
    if exponent == 0xff {
        return f32::NAN;
    }
    if exponent == 0 {
        return f32::from_bits(1);
    }
    if exponent <= 23 {
        f32::from_bits(1_u32 << (exponent - 1))
    } else {
        f32::from_bits((exponent - 23) << 23)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_spacing_covers_subnormal_binade_edges_and_nonfinite_values() {
        assert_eq!(spacing_f64(-0.0).to_bits(), 1);
        assert_eq!(spacing_f64(f64::MIN_POSITIVE).to_bits(), 1);
        assert_eq!(spacing_f64(-1.0), f64::EPSILON);
        assert_eq!(spacing_f64(2.0), 2.0 * f64::EPSILON);
        assert_eq!(spacing_f64(f64::MAX), f64::from_bits(1994_u64 << 52));
        assert!(spacing_f64(f64::INFINITY).is_nan());
        assert!(spacing_f64(f64::NAN).is_nan());

        assert_eq!(spacing_f32(-0.0).to_bits(), 1);
        assert_eq!(spacing_f32(f32::MIN_POSITIVE).to_bits(), 1);
        assert_eq!(spacing_f32(-1.0), f32::EPSILON);
        assert_eq!(spacing_f32(2.0), 2.0 * f32::EPSILON);
        assert_eq!(spacing_f32(f32::MAX), f32::from_bits(231_u32 << 23));
        assert!(spacing_f32(f32::INFINITY).is_nan());
        assert!(spacing_f32(f32::NAN).is_nan());
    }
}
