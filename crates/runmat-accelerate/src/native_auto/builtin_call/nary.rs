//! Shape work for an already-admitted N-ary MATLAB broadcast operation.
use runmat_value::Value;

pub(super) fn broadcast_elements(values: &[Value]) -> Option<usize> {
    let mut shape = vec![1, 1];
    for value in values {
        let dimensions = match value {
            Value::Num(_) => vec![1, 1],
            Value::Tensor(tensor) => tensor.shape().to_vec(),
            Value::GpuTensor(handle) => handle.shape().to_vec(),
            _ => return None,
        };
        shape = runmat_runtime::builtins::common::broadcast::broadcast_shapes(
            "N-ary preparation",
            &shape,
            &dimensions,
        )
        .ok()?;
    }
    if shape.contains(&0) {
        Some(0)
    } else {
        shape
            .iter()
            .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::Tensor;
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn low_rank_operands_retain_broadcast_work_cardinality() {
        let scalar = Value::Tensor(Tensor::from_f32(vec![1.0], Vec::new()).unwrap());
        assert_eq!(broadcast_elements(std::slice::from_ref(&scalar)), Some(1));
        let vector = Value::Tensor(Tensor::from_f32(vec![1.0, 2.0], vec![2]).unwrap());
        assert_eq!(broadcast_elements(&[scalar, vector]), Some(2));
        let empty = Value::Tensor(Tensor::from_f32(Vec::new(), vec![0]).unwrap());
        assert_eq!(broadcast_elements(&[empty]), Some(0));
    }

    #[test]
    fn all_operands_contribute_to_nary_broadcast_work() {
        let inputs = [
            Value::Tensor(Tensor::new(vec![0.0; 2], vec![2, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![1.0; 3], vec![1, 3]).unwrap()),
            Value::Tensor(Tensor::new(vec![1.0; 4], vec![1, 1, 4]).unwrap()),
        ];
        assert_eq!(broadcast_elements(&inputs), Some(24));
        let mut empty = inputs.to_vec();
        empty[2] = Value::Tensor(Tensor::new(vec![], vec![1, 1, 0]).unwrap());
        assert_eq!(broadcast_elements(&empty), Some(0));
    }
}
