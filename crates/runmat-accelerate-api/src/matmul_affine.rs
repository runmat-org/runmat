//! Ordered floating-point operations; absent phases are not identity arithmetic.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MatmulOffset<T = f64> {
    Add(T),
    Subtract(T),
    SubtractFrom(T),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct MatmulAffine<T = f64> {
    pub scale: Option<T>,
    pub offset: Option<MatmulOffset<T>>,
}

impl MatmulAffine {
    pub fn apply(self, input: f64) -> f64 {
        let value = self.scale.map_or(input, |scale| input * scale);
        match self.offset {
            None => value,
            Some(MatmulOffset::Add(c)) => value + c,
            Some(MatmulOffset::Subtract(c)) => value - c,
            Some(MatmulOffset::SubtractFrom(c)) => c - value,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_has_one_serialized_arithmetic_authority() {
        let descriptor = crate::MatmulEpilogue::noop();
        let value = serde_json::to_value(&descriptor).unwrap();
        assert!(value.get("alpha").is_none() && value.get("beta").is_none());
        // Only non-owning arithmetic facts round-trip. An execution epilogue
        // retains live tensor leases and cannot be acquired by deserialization.
        let mut value = value.get("affine").unwrap().clone();
        assert_eq!(
            serde_json::from_value::<MatmulAffine>(value.clone()).unwrap(),
            descriptor.affine
        );
        value
            .as_object_mut()
            .unwrap()
            .insert("alpha".into(), serde_json::json!(2.0));
        assert!(serde_json::from_value::<MatmulAffine>(value).is_err());
    }

    #[test]
    fn ordered_phases_preserve_zero_orientation_and_exceptional_values() {
        let no_op = MatmulAffine {
            scale: None,
            offset: None,
        };
        assert_eq!(no_op.apply(-0.0).to_bits(), (-0.0f64).to_bits());
        for input in [f64::NEG_INFINITY, -0.0, 0.0, f64::INFINITY, f64::NAN] {
            for scalar in [-0.0, 0.0, 2.0] {
                for (offset, expected) in [
                    (MatmulOffset::Add(scalar), input + scalar),
                    (MatmulOffset::Subtract(scalar), input - scalar),
                    (MatmulOffset::SubtractFrom(scalar), scalar - input),
                ] {
                    let actual = MatmulAffine {
                        scale: None,
                        offset: Some(offset),
                    }
                    .apply(input);
                    assert!(
                        actual.is_nan() && expected.is_nan()
                            || actual.to_bits() == expected.to_bits()
                    );
                }
            }
        }
    }
}
