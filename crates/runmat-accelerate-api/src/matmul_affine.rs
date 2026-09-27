//! Ordered floating-point operations; absent phases are not identity arithmetic.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum MatmulOffset {
    Add(f64),
    Subtract(f64),
    SubtractFrom(f64),
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum MatmulAffine {
    /// Compatibility for existing direct provider callers using alpha/beta.
    #[default]
    Legacy,
    /// Exact source order: optional multiplication followed by one offset.
    /// The descriptor's legacy alpha/beta fields are ignored in this mode.
    Ordered {
        scale: Option<f64>,
        offset: Option<MatmulOffset>,
    },
}

impl MatmulAffine {
    pub fn apply(self, input: f64, legacy_alpha: f64, legacy_beta: f64) -> f64 {
        match self {
            Self::Legacy => input * legacy_alpha + legacy_beta,
            Self::Ordered { scale, offset } => {
                let value = scale.map_or(input, |scale| input * scale);
                match offset {
                    None => value,
                    Some(MatmulOffset::Add(c)) => value + c,
                    Some(MatmulOffset::Subtract(c)) => value - c,
                    Some(MatmulOffset::SubtractFrom(c)) => c - value,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordered_phases_preserve_zero_orientation_and_exceptional_values() {
        let no_op = MatmulAffine::Ordered {
            scale: None,
            offset: None,
        };
        assert_eq!(no_op.apply(-0.0, 1.0, 0.0).to_bits(), (-0.0f64).to_bits());
        for input in [f64::NEG_INFINITY, -0.0, 0.0, f64::INFINITY, f64::NAN] {
            for scalar in [-0.0, 0.0, 2.0] {
                for (offset, expected) in [
                    (MatmulOffset::Add(scalar), input + scalar),
                    (MatmulOffset::Subtract(scalar), input - scalar),
                    (MatmulOffset::SubtractFrom(scalar), scalar - input),
                ] {
                    let actual = MatmulAffine::Ordered {
                        scale: None,
                        offset: Some(offset),
                    }
                    .apply(input, 99.0, 99.0);
                    assert!(
                        actual.is_nan() && expected.is_nan()
                            || actual.to_bits() == expected.to_bits()
                    );
                }
            }
        }
    }
}
