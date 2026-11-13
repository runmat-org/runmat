//! Carrier authority from SQRT's ordinary host reconstruction and resident
//! publication. This helper proves representation, never nonnegative payloads.
use runmat_types::{NumericClass, NumericDomain, StorageFact, ValueFact};

pub(super) fn storage(
    input: &ValueFact,
    output: &ValueFact,
    class: NumericClass,
) -> Option<StorageFact> {
    let numeric = input.numeric()?;
    if numeric.domain != NumericDomain::Real
        || numeric.class != class
        || !matches!(class, NumericClass::Single | NumericClass::Double)
        || !matches!(input.storage, StorageFact::Dense | StorageFact::Scalar)
        || matches!(output.storage, StorageFact::Sparse | StorageFact::Opaque)
        || input.shape != output.shape
        || output
            .numeric()
            .is_some_and(|numeric| numeric.domain != NumericDomain::Real || numeric.class != class)
    {
        return None;
    }
    let elements = input.shape.element_count()?;
    if elements == 0 {
        return None;
    }
    if input.storage == StorageFact::Scalar && (elements != 1 || class != NumericClass::Double) {
        return None;
    }
    // host::evaluate_f32 reconstructs Tensor even for one element. F64 uses
    // common::tensor::tensor_into_value and may scalarize; a rank-preserving
    // output fact does not authorize that one-element public shape boundary.
    if class == NumericClass::Double && elements == 1 {
        return None;
    }
    Some(StorageFact::Dense)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value_fact::value_fact;
    use runmat_value::{Tensor, Value};

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn actual_host_carriers_single_one_many_double_many_and_scalar_boundary() {
        for (single, count) in [(true, 1), (true, 2), (false, 2)] {
            let tensor = if single {
                Tensor::from_f32(vec![4.0; count], vec![count, 1, 1]).unwrap()
            } else {
                Tensor::new(vec![4.0; count], vec![count, 1, 1]).unwrap()
            };
            let input = value_fact(&Value::Tensor(tensor.clone()));
            let actual = super::super::host::evaluate_tensor(tensor).unwrap();
            let output = value_fact(&actual);
            assert_eq!(
                storage(&input, &output, input.numeric().unwrap().class),
                Some(output.storage)
            );
            assert_eq!(output.storage, StorageFact::Dense);
        }
        let input = value_fact(&Value::Tensor(Tensor::new(vec![4.0], vec![1, 1]).unwrap()));
        let actual =
            super::super::host::evaluate_tensor(Tensor::new(vec![4.0], vec![1, 1]).unwrap())
                .unwrap();
        assert!(matches!(actual, Value::Num(2.0)));
        assert_eq!(
            storage(&input, &value_fact(&actual), NumericClass::Double),
            None
        );
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn unproved_and_inconsistent_carriers_fail_closed() {
        let input = value_fact(&Value::Tensor(
            Tensor::from_f32(vec![4.0; 2], vec![2, 1, 1]).unwrap(),
        ));
        for carrier in [
            StorageFact::Unknown,
            StorageFact::Sparse,
            StorageFact::Opaque,
            StorageFact::Scalar,
        ] {
            let mut bad = input.clone();
            bad.storage = carrier;
            assert_eq!(storage(&bad, &input, NumericClass::Single), None);
        }
        let negative = value_fact(
            &super::super::host::evaluate_tensor(
                Tensor::from_f32(vec![-4.0; 2], vec![2, 1, 1]).unwrap(),
            )
            .unwrap(),
        );
        assert_eq!(storage(&input, &negative, NumericClass::Single), None);
    }
}
