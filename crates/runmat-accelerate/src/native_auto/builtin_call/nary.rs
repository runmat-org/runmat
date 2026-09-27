//! Shape work for an already-admitted N-ary MATLAB broadcast operation.
use runmat_value::Value;

pub(super) fn broadcast_elements(values: &[Value]) -> Option<usize> {
    let mut shape = runmat_types::ShapeFact::from(vec![Some(1), Some(1)]);
    for value in values {
        let dimensions = match value {
            Value::Num(_) => vec![1, 1],
            Value::Tensor(tensor) => tensor.shape.clone(),
            Value::GpuTensor(handle) => handle.shape.clone(),
            _ => return None,
        };
        let dimensions = if dimensions.len() == 1 {
            vec![1, dimensions[0]]
        } else {
            dimensions
        };
        let fact =
            runmat_types::ShapeFact::from(dimensions.into_iter().map(Some).collect::<Vec<_>>());
        shape = runmat_types::broadcast_shape(&shape, &fact).ok()?;
    }
    shape.element_count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::Tensor;
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
