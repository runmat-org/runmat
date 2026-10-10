//! Value-sign evidence is orthogonal to numerical-domain and arithmetic receipts.
//! A caller must bind this facet to the exact selected ValueId and class before
//! passing it to another operation owner. It is not provider capability.

use runmat_types::NumericClass;
use runmat_value::{NumericScalar, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum FusionSignEvidence {
    /// Every represented real element is >= 0 (either signed zero included),
    /// positive infinity, or NaN. No assertion is made about NaN payloads.
    NonnegativeOrNaN { class: NumericClass },
}

impl FusionSignEvidence {
    pub fn class(self) -> NumericClass {
        match self {
            Self::NonnegativeOrNaN { class } => class,
        }
    }
}

/// Scan actual host storage. An opaque resident or absent value has no host
/// sign proof. The returned element count belongs in admission-work telemetry.
pub fn scan_represented_sign(value: &Value, class: NumericClass) -> Option<(bool, u64)> {
    let acceptable = |value: NumericScalar| match (class, value) {
        (NumericClass::Single, NumericScalar::F32(value)) => value.is_nan() || value >= 0.0,
        (NumericClass::Double, NumericScalar::F64(value)) => value.is_nan() || value >= 0.0,
        _ => false,
    };
    match value {
        Value::Num(value) if class == NumericClass::Double => {
            Some((value.is_nan() || *value >= 0.0, 1))
        }
        Value::Tensor(tensor) => {
            if tensor.is_empty() {
                return Some((false, 0));
            }
            let mut examined = 0_u64;
            for index in 0..tensor.len() {
                examined = examined.checked_add(1)?;
                if !tensor.numeric_value_at(index).is_some_and(acceptable) {
                    return Some((false, examined));
                }
            }
            Some((true, examined))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::super::{FusionAdmissionContext, FusionAdmissionWork, FusionOperandDomainEvidence};
    use super::*;
    use runmat_accelerate_api::NumericalDomainEvidence;
    use runmat_value::Tensor;

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn represented_scan_retains_nan_infinity_signed_zero_and_class() {
        let value = Value::Tensor(
            Tensor::from_f32(vec![-0.0, f32::NAN, f32::INFINITY], vec![1, 3]).unwrap(),
        );
        assert_eq!(
            scan_represented_sign(&value, NumericClass::Single),
            Some((true, 3))
        );
        assert_eq!(
            scan_represented_sign(&value, NumericClass::Double),
            Some((false, 1))
        );
        let negative = Value::Tensor(Tensor::from_f32(vec![-1.0], vec![1, 1]).unwrap());
        assert_eq!(
            scan_represented_sign(&negative, NumericClass::Single),
            Some((false, 1))
        );
        assert_eq!(
            scan_represented_sign(&Value::Num(0.0), NumericClass::Single),
            None
        );
        assert_eq!(
            scan_represented_sign(&Value::Num(f64::NAN), NumericClass::Double),
            Some((true, 1))
        );
        let empty = Value::Tensor(Tensor::from_f32(vec![], vec![0, 1]).unwrap());
        assert_eq!(
            scan_represented_sign(&empty, NumericClass::Single),
            Some((false, 0))
        );
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn sign_facet_cannot_cross_value_id_or_real_class() {
        let fact = crate::value_fact::value_fact(&Value::Num(1.0));
        let operands = [None];
        let facts = [fact];
        let mut work = FusionAdmissionWork::None;
        let proof = FusionOperandDomainEvidence {
            value_id: 7,
            numerical_domain: NumericalDomainEvidence::Unknown,
            sign: Some(FusionSignEvidence::NonnegativeOrNaN {
                class: NumericClass::Double,
            }),
        };
        assert!(
            FusionAdmissionContext::new(&[8], &operands, &facts, &[Some(proof)], &mut work)
                .is_none()
        );
        let wrong_class = FusionOperandDomainEvidence {
            value_id: 8,
            sign: Some(FusionSignEvidence::NonnegativeOrNaN {
                class: NumericClass::Single,
            }),
            ..proof
        };
        assert!(FusionAdmissionContext::new(
            &[8],
            &operands,
            &facts,
            &[Some(wrong_class)],
            &mut work
        )
        .is_none());
    }
}
